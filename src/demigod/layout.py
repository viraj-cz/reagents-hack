"""Volume layout: TWO Modal Volumes per run.

    demigod-run-<run_id>-shared        read-only input, same view for every DEMI_GOD
    demigod-run-<run_id>-out
      <name>/                          that agent's writable output dir, private to it

Enforced with Modal's per-mount options rather than by convention, so an agent
physically cannot write to shared/ or into a sibling's output:

    volumes={
      "/run/shared": shared_vol.with_mount_options(read_only=True),
      "/run/out":    out_vol.with_mount_options(sub_path=name),
    }

`sub_path` on the out volume means each sandbox sees ONLY its own output dir
mounted at /run/out -- the agent has no path by which to reach a sibling's
output. Isolation between DEMI_GODs is a mount-time property, not a prompt
instruction.

WHY TWO VOLUMES, not one with two sub_paths:
    Modal rejects it. Verified live:
        InvalidError: The same Volume cannot be mounted in multiple locations
        for the same function: /run/out, /run/shared
    The original design mounted one run volume twice (sub_path="shared"
    read-only, sub_path="out/<name>" writable). That is not permitted, so the
    read-only input and the writable output are separate Volume objects. Every
    property the one-volume design had is preserved; only the names changed.

RUNNER-INDEPENDENT. These are the paths as seen *inside the sandbox*; an
outside-the-sandbox runner would mount the same volumes locally and use the
same sub_paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import modal

# Paths inside the sandbox.
SHARED_MOUNT = "/run/shared"
OUT_MOUNT = "/run/out"
SPEC_PATH = "/run/spec.json"
"""Where the spec JSON is materialized inside the sandbox. Deliberately outside
both mounts -- it is control-plane data, not an artifact, and must not end up in
the agent's output dir where it would pollute `files`."""


@dataclass(frozen=True)
class RunLayout:
    """Resolved volumes and paths for one DEMI_GOD within one run."""

    run_id: str
    demigod_name: str

    # --- volume names ---
    #
    # TODO: add a retention/GC policy. These are cheap but not free, and a
    # hackathon will make hundreds of them.

    @property
    def shared_volume_name(self) -> str:
        """Per-run, shared by every DEMI_GOD in the run. Seeded once, then
        read-only everywhere."""
        return f"demigod-run-{self.run_id}-shared"

    @property
    def out_volume_name(self) -> str:
        """Per-run. Each DEMI_GOD gets a private sub_path within it."""
        return f"demigod-run-{self.run_id}-out"

    @property
    def out_subpath(self) -> str:
        """This agent's directory within the out volume."""
        return self.demigod_name

    def shared_volume(self, *, create_if_missing: bool = True) -> modal.Volume:
        import modal

        return modal.Volume.from_name(
            self.shared_volume_name, create_if_missing=create_if_missing
        )

    def out_volume(self, *, create_if_missing: bool = True) -> modal.Volume:
        import modal

        return modal.Volume.from_name(
            self.out_volume_name, create_if_missing=create_if_missing
        )

    def seed_shared(self, local_paths: list[Path]) -> list[str]:
        """Upload local files into the run's shared volume.

        Without this there is no way for input data to reach a DEMI_GOD: the
        spec's `files` field names paths relative to shared/, but naming them
        does not put them there. Runs before any sandbox is created -- shared/
        is mounted read-only everywhere, so it cannot be populated from inside.

        Directories are uploaded recursively, preserving their internal
        structure.
        """
        vol = self.shared_volume()
        uploaded: list[str] = []

        with vol.batch_upload(force=True) as batch:
            for local in local_paths:
                if not local.exists():
                    raise FileNotFoundError(f"--shared path does not exist: {local}")
                if local.is_dir():
                    for f in sorted(local.rglob("*")):
                        if f.is_file():
                            rel = f.relative_to(local).as_posix()
                            batch.put_file(f, f"/{rel}")
                            uploaded.append(rel)
                else:
                    batch.put_file(local, f"/{local.name}")
                    uploaded.append(local.name)

        return uploaded

    def sandbox_volumes(self) -> dict[str, modal.Volume]:
        """The `volumes=` argument for `modal.Sandbox.create`.

        Two distinct Volume objects -- see the module docstring for why one
        volume mounted twice is rejected by Modal.
        """
        return {
            SHARED_MOUNT: self.shared_volume().with_mount_options(read_only=True),
            OUT_MOUNT: self.out_volume().with_mount_options(sub_path=self.out_subpath),
        }
