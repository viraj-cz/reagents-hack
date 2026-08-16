---
name: add-tool-to-registry
description: Add a new tool to the DEMI_GOD closed tool registry so agents can request it by key. Use when someone wants a DEMI_GOD to have a capability it does not have (a library, an API client, a CLI), when a spec fails validation with "unknown tool key", or when a tool needs to be authored once and reused across runs. Covers the registry entry, image placement, the agent-facing usage doc, and the required smoke test.
---

# Adding a tool to the DEMI_GOD registry

The registry (`src/demigod/registry/`) is a **closed set**. A DEMI_GOD can only
be given tools that exist in it; free-text tool names are rejected at spec
validation. That is the point: the GOD is a language model picking tools, and
an invented tool name should fail in milliseconds on the caller with a list of
what exists, not as an ImportError deep inside a live sandbox.

The cost of that is this procedure. Follow it in order. Tools are authored once
and reused forever, so it is worth doing properly the first time.

## Step 1 — Check it isn't already there

```bash
python -c "from demigod.registry import all_keys; print(all_keys())"
```

Then check for near-misses, because the most common failure here is adding a
second tool that does what an existing one already does:

```bash
grep -ri "<capability>" src/demigod/registry/
```

Ask whether the need is genuinely new. `polars` when `pandas` is already
registered is usually a preference, not a capability — and every added tool is
another thing the GOD can pick wrongly. If an existing entry covers it, stop
and use that key.

## Step 2 — Write the registry entry

Add a `ToolEntry` to `src/demigod/registry/entries.py`, then register it in the
`REGISTRY` dict at the bottom of that file (easy to forget — the entry is inert
until you do). Copy the shape of `PANDAS`.

```python
MY_TOOL = ToolEntry(
    key="my-tool",  # what the GOD writes in spec.tools
    display_name="My Tool",
    install=("my-tool==1.2.3",),  # PIN every version
    secrets=("MY_TOOL_API_KEY",),  # env var names, or () if offline
    apt=(),  # system packages, if any
    smoke_test=("python", "-c", "import my_tool; print('ok')"),
    doc_file="my_tool.md",
    tags=("category",),
)
```

Rules that matter:

- **Pin versions.** Unpinned installs make pre-baked images non-reproducible;
  two DEMI_GODs nominally holding the same tool end up holding different ones.
- **`key` is permanent.** It appears in every spec ever written. Lowercase,
  hyphenated, no version in the name.
- **Declare every secret.** `secrets` drives which Modal Secrets get mounted.
  Undeclared credentials mean an auth failure mid-run.

If the tool needs credentials, create the Modal Secret now, one env var per
secret, named `demigod-<env-var-lowercased-with-hyphens>` (the convention in
`runner/inside.py::_secret_name_for`):

```bash
modal secret create demigod-my-tool-api-key MY_TOOL_API_KEY=...
```

## Step 3 — Decide where it lives in the image catalog

Images are **pre-baked, never built per spawn**. Open
`src/demigod/images.py` and pick one:

**(a) It fits an existing image.** Add the key to that image's `tool_keys`:

```python
DATA = PrebakedImage(
    name="demigod-data",
    tool_keys=frozenset({"pandas", "my-tool"}),
    ...
)
```

Do this when the tool is small and commonly wanted alongside what's already in
that image.

**(b) It needs a new image.** Add a `PrebakedImage` to `CATALOG`. Do this when
the tool is heavy (CUDA, a browser, a JVM) or rarely co-requested — otherwise
every DEMI_GOD pays its pull cost.

```python
MY_IMAGE = PrebakedImage(
    name="demigod-my-thing",
    tool_keys=frozenset({"my-tool"}),
    description="What this image is for.",
)

CATALOG = (BASE, DATA, MY_IMAGE)
```

Keep the catalog **nested where possible** — a larger image containing a
smaller one's tools. `resolve_image` picks the smallest covering image, and
that only produces good choices if coverage forms a sensible hierarchy.

Watch the combinatorics: the resolver hard-errors if no single image covers a
requested set. If a plausible GOD would ask for `{pandas, my-tool}` together,
one image must contain both.

Then bake:

```bash
python scripts/bake.py --image demigod-data
```

## Step 4 — Write the agent-facing usage doc

Create `src/demigod/registry/docs/<doc_file>`. This is spliced verbatim into
the system prompt of any DEMI_GOD granted the tool, so it is the highest-leverage
part of the entry — and the part most often written as an afterthought.

Write it **for the agent, not for a human**. Read
`src/demigod/registry/docs/pandas.md` as the reference. It should be short and
contain:

- The import or invocation, exactly.
- The two or three idioms the agent will actually need, as code.
- **The traps.** This is the valuable part: the silent failures, the footguns,
  the API that returns something surprising. An agent that knows
  `merge` defaults to an inner join writes correct code; one that doesn't
  produces a plausible wrong answer.
- Where to read from (`/run/shared`, read-only) and write to (`/run/out`).

Do not restate the library's tutorial. Assume competence, supply specifics.

## Step 5 — Add and RUN the smoke test

The `smoke_test` field is argv that must exit 0 iff the tool genuinely works in
the image. Make it exercise the thing, not just import it — an import check
passes on a broken install of a package with lazy submodules.

```bash
python scripts/smoke_test.py --tool my-tool
```

**A tool is not added until this passes.** If it fails, the registry entry is
lying about what the image provides, and every DEMI_GOD given that key will
fail confusingly.

Note the limit: an offline smoke test cannot prove credentials are configured.
For a tool with `secrets`, verify separately that the Modal Secret exists.

## Step 6 — Verify end to end

```bash
python -c "from demigod.images import resolve_image; print(resolve_image(['my-tool']).name)"
```

Then dry-run a spec that requests it — this exercises validation, image
resolution, and prompt construction without creating a sandbox:

```bash
python -m demigod.spawn --spec examples/pandas-demigod.json --dry-run
```

## Checklist

- [ ] Confirmed no existing entry covers this capability
- [ ] `ToolEntry` written with **pinned** versions and declared secrets
- [ ] Added to the `REGISTRY` dict
- [ ] Modal Secret created, if the tool needs credentials
- [ ] Placed in an existing image or a new `PrebakedImage`, and baked
- [ ] Usage doc written for the agent, including the traps
- [ ] Smoke test written and **passing**
- [ ] `resolve_image([key])` returns the expected image
