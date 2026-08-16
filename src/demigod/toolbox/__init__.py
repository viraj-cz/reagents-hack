"""The DEMI_GOD half of the TOOLBOX_BROKER: wire contract, client, CLI.

Ships INSIDE the sandbox image. Everything here is stdlib + pydantic, imports no
Modal, and knows nothing about how a tool is executed -- only how to ask.

The broker half lives in `src/broker/` and is never shipped into a demigod
image. That asymmetry is the whole security model: see
`demigod/toolbox/protocol.py`.
"""

from __future__ import annotations

from demigod.toolbox.client import (
    DEFAULT_GRANT_FILE,
    ENV_GRANT_FILE,
    ENV_LEASE,
    ENV_URL,
    ToolboxClient,
    ToolboxError,
)
from demigod.toolbox.protocol import (
    PROTOCOL_VERSION,
    BrokeredTool,
    CallRequest,
    CallResponse,
    DescribeResponse,
    ErrorCode,
    ErrorResponse,
    LeaseStatus,
    ListResponse,
    ToolboxGrant,
    ToolError,
)

__all__ = [
    "DEFAULT_GRANT_FILE",
    "ENV_GRANT_FILE",
    "ENV_LEASE",
    "ENV_URL",
    "PROTOCOL_VERSION",
    "BrokeredTool",
    "CallRequest",
    "CallResponse",
    "DescribeResponse",
    "ErrorCode",
    "ErrorResponse",
    "LeaseStatus",
    "ListResponse",
    "ToolError",
    "ToolboxClient",
    "ToolboxError",
    "ToolboxGrant",
]
