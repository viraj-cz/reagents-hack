"""Read-only diagnostics for local and remote capability setup."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from typing import Any

from reagents.tools.container import runtime_mode
from reagents.tools.mcp import SPONSOR_MCP_SERVERS, configure_sponsor_mcp
from reagents.tools.registry import default_registry


COMMANDS = ("docker", "podman", "lean", "lake", "z3", "blastn", "hmmscan", "mafft", "iqtree2")
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sympy": "sympy",
    "networkx": "networkx",
    "pint": "pint",
    "cvxpy": "cvxpy",
    "z3-solver": "z3",
    "biopython": "Bio",
    "rdkit": "rdkit",
    "cobra": "cobra",
    "scanpy": "scanpy",
    "pymc": "pymc",
    "cantera": "cantera",
    "openmm": "openmm",
    "mcp": "mcp",
}


async def doctor_report(*, connect: bool = False) -> dict[str, Any]:
    commands = {name: shutil.which(name) for name in COMMANDS}
    packages = {
        display: importlib.util.find_spec(module) is not None
        for display, module in PACKAGES.items()
    }
    servers: dict[str, Any] = {}
    for config in SPONSOR_MCP_SERVERS:
        servers[config.namespace] = {
            "url": config.url,
            "credential_env": config.auth_env,
            "credential_present": bool(config.auth_env and os.environ.get(config.auth_env)),
            "connected": False,
            "tools": [],
            "error": None,
        }

    if connect:
        registry = default_registry()
        if not registry.deferred_namespaces():
            configure_sponsor_mcp(registry)
        await registry.load_deferred(strict=False)
        for namespace in registry.deferred_namespaces():
            tools = [spec.id for spec in registry.specs() if spec.namespace == namespace]
            servers[namespace]["connected"] = namespace not in registry.load_errors
            servers[namespace]["tools"] = tools
            servers[namespace]["error"] = registry.load_errors.get(namespace)

    return {
        "python": {
            "version": sys.version.split()[0],
            "supported": sys.version_info >= (3, 11),
        },
        "commands": commands,
        "packages": packages,
        # Which path a CONTAINER-provider tool would take from HERE. On a laptop
        # this says "container", and the `docker`/`podman` line above is then the
        # one that matters; inside a broker executor it says "inprocess" and the
        # `packages` line is. Reporting both without saying which is in force
        # sends people to debug the wrong half.
        "tool_runtime": runtime_mode(),
        "mcp_servers": servers,
    }
