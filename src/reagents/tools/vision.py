"""Reading an image with a vision model, as a brokered tool.

WHY THIS IS A BROKER TOOL AND NOT A LIBRARY IN THE SANDBOX
----------------------------------------------------------
`imaging` (the demigod-registry key) gives an agent scikit-image, cellpose and
friends: everything needed to *measure* an image. It gives it nothing that can
*look* at one. A segmentation mask is a number; "the third well is contaminated
and the plate is rotated ~4 degrees" is a reading, and no amount of local
morphology produces it.

So the two halves are deliberately split:

    imaging (image key)      pixels -> numbers. Runs in the sandbox.
    vision.read_image        pixels -> words.   Runs on the broker.

The credential is the reason the second one is brokered rather than a pip
install. `.env.example` states the rule plainly -- tool credentials belong to the
TOOLBOX_BROKER, never to a demigod -- and a sealed agent holding an OpenAI key
could call any OpenAI endpoint with it, including ones that carry its domain
representation out of the seal. Here it holds a lease id, the call is metered
against that lease, and the broker writes the trace.

WHY urllib AND NOT THE `openai` SDK
-----------------------------------
This tool is `ToolProvider.LOCAL`, so `DispatchPolicy` runs it INLINE in the
router replica (see broker.dispatch.INLINE_BY_DEFAULT). The router image is
`broker_image()` with no extras -- Python, pydantic, and this repo. Adding the
SDK there would put it in the base layer of every executor tier that builds on
it, to send one JSON POST. The request is a dict and the response is a dict;
stdlib covers it.

Blocking I/O in an ASGI replica is the one real hazard of running inline, and it
is handled the same way `reagents.tools.container` handles it: the POST goes to
`asyncio.to_thread`, so a 20-second vision call does not stall the demigods
sharing the box.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import urllib.error
import urllib.request
from typing import Any

from reagents.contracts import ToolAccess, ToolProvider
from reagents.tools.registry import Tool, ToolExecutionError, ToolRegistry

API_KEY_ENV = "OPENAI_API_KEY"
"""Read at CALL time, never at import. The router process is long-lived and its
secret is mounted by Modal; resolving at import would freeze whatever the value
was when the module first loaded."""

MODEL_ENV = "OPENAI_VISION_MODEL"
BASE_URL_ENV = "OPENAI_BASE_URL"

DEFAULT_MODEL = "gpt-5.5"
"""Verified present on the account's model list, and dated (2026-04-23) rather
than a moving alias, so two runs a month apart read an image with the same
model. Override per-call with `model`, or per-deployment with $OPENAI_VISION_MODEL."""

DEFAULT_BASE_URL = "https://api.openai.com/v1"

RESPONSES_PATH = "/responses"
"""The Responses API, not Chat Completions. GPT-5-class models are reasoning
models: on /chat/completions they reject `max_tokens` in favour of
`max_completion_tokens` and constrain `temperature`, and the reasoning tokens
are billed but not addressable. /responses models exactly what this tool needs
-- typed input parts, one `max_output_tokens`, and a `status` field that says
whether the answer was cut off rather than leaving it to be inferred."""

DEFAULT_MAX_OUTPUT_TOKENS = 1500
DEFAULT_TIMEOUT_S = 180.0
"""A high-detail read of a dense microscopy field is tens of seconds. The
router's own ceiling is ROUTER_TIMEOUT_S (900s), so this is the binding limit
and it is deliberately well under it."""

MAX_IMAGE_BYTES = 2_500_000
"""Decoded image bytes. THE trap this tool has, so it is enforced here with a
message rather than discovered as a truncated HTTP body.

`broker.router.MAX_BODY_BYTES` caps a request at 4 MiB and base64 costs 4/3, so
~3 MiB of image is the hard wall. 2.5 MB leaves room for the question and the
JSON envelope. A 16-bit 2048x2048 TIFF is 8 MB and will not fit -- which is
correct: it should be windowed and 8-bit PNG encoded first, and the agent has
the whole `imaging` stack to do that with.
"""

SUPPORTED_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
"""What the API accepts. Notably NOT image/tiff, which is what microscopy
actually arrives as -- hence the conversion idiom in docs/imaging.md."""

TOOL_ID = "vision.read_image"


class VisionConfigurationError(ToolExecutionError):
    """No credential where the tool runs. A deployment fact, not a bad call."""


def _api_key() -> str:
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise VisionConfigurationError(
            f"{TOOL_ID} needs ${API_KEY_ENV} in the process that executes it, "
            f"which is the TOOLBOX_BROKER router -- not the demigod sandbox and "
            f"not the caller. Locally: put {API_KEY_ENV}=sk-... in .env. On "
            f"Modal: the router picks it up from the deploying shell's "
            f"environment (see broker.service.BROKER_CREDENTIAL_ENV_VARS), so "
            f"load .env before `modal deploy -m broker.service`."
        )
    return key


def _endpoint() -> str:
    base = (os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
    return f"{base}{RESPONSES_PATH}"


def _data_url(image_base64: str, media_type: str) -> str:
    """Validate the payload before spending a lease call on a 400.

    The agent produced this string with `base64 -w0` or `b64encode` in bash, so
    the failures here are whitespace, a data-URL prefix pasted twice, and the
    size wall -- all of which are worth naming precisely, because the agent gets
    one shot per lease call at fixing them.
    """
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise ToolExecutionError(
            f"media_type {media_type!r} is not accepted by the vision API. "
            f"Use one of {list(SUPPORTED_MEDIA_TYPES)}; convert TIFF/OME-Zarr to "
            f"8-bit PNG first."
        )
    payload = "".join(image_base64.split())
    if payload.startswith("data:"):
        # Already a data URL. Take it apart rather than nesting one inside
        # another, which fails server-side with an opaque 400.
        _, _, payload = payload.partition(",")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ToolExecutionError(
            f"image_base64 is not valid base64: {exc}. Encode with "
            f"`base64 -w0 shot.png` (no line wrapping) or "
            f"`base64.b64encode(path.read_bytes()).decode()`."
        ) from exc
    if not decoded:
        raise ToolExecutionError("image_base64 decoded to zero bytes")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ToolExecutionError(
            f"image is {len(decoded) / 1e6:.1f} MB decoded; the limit is "
            f"{MAX_IMAGE_BYTES / 1e6:.1f} MB because the broker caps a request "
            f"body at 4 MiB and base64 costs a third on top. Downscale or crop "
            f"to the region your question is about and send that -- a 1024px "
            f"PNG of the right field beats a full-resolution plate."
        )
    return f"data:{media_type};base64,{payload}"


def _image_part(arguments: dict[str, Any]) -> dict[str, Any]:
    image_base64 = arguments.get("image_base64")
    image_url = arguments.get("image_url")
    if bool(image_base64) == bool(image_url):
        raise ToolExecutionError(
            "pass exactly one of `image_base64` (an image you produced or read "
            "from /run/shared) or `image_url` (a public URL the broker can "
            "fetch)."
        )
    part: dict[str, Any] = {"type": "input_image"}
    if image_base64:
        media_type = str(arguments.get("media_type") or "image/png")
        part["image_url"] = _data_url(str(image_base64), media_type)
    else:
        part["image_url"] = str(image_url)
    detail = arguments.get("detail")
    if detail:
        if detail not in {"low", "high", "auto"}:
            raise ToolExecutionError("detail must be 'low', 'high' or 'auto'")
        part["detail"] = detail
    return part


def _request_body(arguments: dict[str, Any]) -> dict[str, Any]:
    question = str(arguments.get("question") or "").strip()
    if not question:
        raise ToolExecutionError(
            "`question` is required. Ask for the specific observation your claim "
            "depends on -- 'how many wells show confluent growth, and which' "
            "gets you evidence; 'describe this image' gets you prose."
        )
    model = str(arguments.get("model") or os.environ.get(MODEL_ENV) or DEFAULT_MODEL)
    max_output_tokens = int(
        arguments.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS
    )
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": question},
        _image_part(arguments),
    ]
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
    }


def _post(body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """One blocking POST. Called via `asyncio.to_thread`, never on the loop."""
    request = urllib.request.Request(
        _endpoint(),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise ToolExecutionError(
            f"vision API returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ToolExecutionError(
            f"could not reach the vision API: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ToolExecutionError(
            f"vision API did not answer within {timeout_s:.0f}s"
        ) from exc


def _reading(response: dict[str, Any]) -> str:
    """Pull the answer text out of a Responses payload.

    Walks `output` rather than trusting `output_text`: that convenience field is
    synthesized by the SDKs, and a reasoning model's `output` array leads with a
    `reasoning` item that has no text in it at all.
    """
    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "output_text":
                chunks.append(str(block.get("text", "")))
    if chunks:
        return "\n".join(c for c in chunks if c).strip()
    text = response.get("output_text")
    return str(text).strip() if text else ""


async def read_image(arguments: dict[str, Any]) -> dict[str, Any]:
    body = _request_body(arguments)
    timeout_s = float(arguments.get("timeout_s") or DEFAULT_TIMEOUT_S)
    response = await asyncio.to_thread(_post, body, timeout_s)

    reading = _reading(response)
    status = str(response.get("status") or "")
    incomplete = response.get("incomplete_details") or {}
    if not reading:
        # An empty answer with status=incomplete means the token budget went
        # entirely to reasoning. Saying so is actionable; "no result" is not.
        raise ToolExecutionError(
            f"the vision model returned no text (status={status or 'unknown'}, "
            f"reason={incomplete.get('reason', 'unspecified')}). Raise "
            f"`max_output_tokens` or ask a narrower question."
        )
    usage = response.get("usage") or {}
    return {
        "reading": reading,
        "model": response.get("model", body["model"]),
        "status": status,
        # True when the answer was cut off mid-sentence. The agent must not
        # quote a truncated reading as a complete observation.
        "truncated": status == "incomplete",
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        },
    }


READ_IMAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["question"],
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "What to determine from the image. Be specific: name the "
                "features, counts, or comparisons your claim needs."
            ),
        },
        "image_base64": {
            "type": "string",
            "description": (
                "Base64 of the image bytes. Use this for anything on your own "
                "disk. Max ~2.5 MB decoded; downscale or crop first."
            ),
        },
        "image_url": {
            "type": "string",
            "description": (
                "Public URL, as an alternative to image_base64. Exactly one of the two."
            ),
        },
        "media_type": {
            "type": "string",
            "enum": list(SUPPORTED_MEDIA_TYPES),
            "default": "image/png",
            "description": "Required with image_base64. TIFF is not accepted.",
        },
        "detail": {
            "type": "string",
            "enum": ["low", "high", "auto"],
            "description": (
                "`high` costs more input tokens and resolves fine structure; "
                "`low` is enough for layout and gross morphology."
            ),
        },
        "model": {"type": "string", "description": f"Defaults to {DEFAULT_MODEL}."},
        "max_output_tokens": {
            "type": "integer",
            "minimum": 16,
            "default": DEFAULT_MAX_OUTPUT_TOKENS,
        },
    },
    "additionalProperties": False,
}

READ_IMAGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reading": {"type": "string", "description": "What the model saw."},
        "model": {"type": "string"},
        "status": {"type": "string"},
        "truncated": {
            "type": "boolean",
            "description": "True if the answer was cut off. Do not quote it as complete.",
        },
        "usage": {"type": "object"},
    },
}

READ_IMAGE_DESCRIPTION = """\
Look at an image and answer a question about it, using a vision model.

THE tool for any image-related task. Anything whose evidence is a picture --
micrographs, plate photographs, gels and blots, chromatograms, IHC slides,
scanned figures and plots, screenshots of instrument output -- goes through
here. Local libraries measure pixels; this is the only thing that reads them.

Send one image and one specific question. Ask for the observation your claim
depends on, not for a description. The answer is a reading, not a measurement:
quantify with your own code and use this to establish what is in the frame,
whether an artifact is present, and whether your segmentation matches what a
careful observer would say is there.

Encode with base64 (max ~2.5 MB decoded, PNG/JPEG/WEBP/GIF only -- convert TIFF
first) or pass a public image_url. Save the result with `-o`: a reading that
exists only in your transcript is not evidence.
"""


def all_tools() -> list[Tool]:
    return [
        Tool(
            id=TOOL_ID,
            namespace="vision",
            description=READ_IMAGE_DESCRIPTION,
            parameters_schema=READ_IMAGE_SCHEMA,
            output_schema=READ_IMAGE_OUTPUT_SCHEMA,
            executor=read_image,
            # LOCAL, so `DispatchPolicy` keeps it inline in the router. It is a
            # network call, but a thin one: no image, no science stack, nothing
            # an executor tier would provide. `read_image` yields the thread, so
            # inline costs the replica nothing but a socket.
            provider=ToolProvider.LOCAL,
            # READ, not WRITE: it changes nothing anywhere. A demigod does not
            # need `allow_write` on its lease to look at a picture.
            access=ToolAccess.READ,
            cost_class="remote",
            latency_class="network",
        )
    ]


def configure_vision_tools(registry: ToolRegistry) -> None:
    for tool in all_tools():
        registry.register(tool)


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODEL",
    "MAX_IMAGE_BYTES",
    "TOOL_ID",
    "VisionConfigurationError",
    "all_tools",
    "configure_vision_tools",
    "read_image",
]
