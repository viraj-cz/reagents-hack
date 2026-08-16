"""Artifact payload validation.

The implementation lives in `demigod.schema` because it has to run INSIDE the
sandbox, and only `demigod` is shipped into the image. Re-exported here so
existing `reagents.schema` imports keep working and there is exactly one
validator rather than two that can drift.
"""

from __future__ import annotations

from demigod.schema import validate_payload

__all__ = ["validate_payload"]
