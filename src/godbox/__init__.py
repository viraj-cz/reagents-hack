"""GOD in its own long-lived Modal sandbox.

The third package, and the reason it is third:

    reagents/   WHAT a demigod reasons about   (plan, transform, seal, integrate)
    demigod/    WHERE and HOW a DEMI_GOD runs  (spawn, isolate, execute, collect)
    godbox/     WHERE and HOW *GOD* runs       (this package)

`godbox` is deliberately NOT part of either existing half:

* Not `demigod/`, because it imports `reagents`, and `demigod` must never do
  that. `demigod` is the only package shipped into a DEMI_GOD's image; the
  moment it can reach `reagents`, the agent's own container contains the
  planner prompts and the inverse maps that constrain it. That import direction
  is a safety property (see `reagents/contracts.py`), not a layering
  preference.
* Not `reagents/god/`, because this is infrastructure rather than reasoning,
  and because `src/reagents` is in ruff's `extend-exclude` -- new code put
  there would ship unlinted and unformatted.

Module map:

    status.py     the modal.Dict status channel -- the ONLY cross-container
                  progress signal. Imported by both sides.
    images.py     god_image(): modal client + Anthropic SDK + all three
                  packages. NOT the demigod image, and that asymmetry is
                  load-bearing.
    launch.py     OUTSIDE the sandbox: create it, hand it a request, detach.
    entrypoint.py INSIDE the sandbox: drive God.solve(), report, self-destruct.
    cli.py        `god launch|status|watch|list|logs|followup|terminate`

The shape, end to end:

    laptop                     GOD sandbox                DEMI_GOD sandboxes
    ------                     -----------                ------------------
    god launch  ---------->    python -m godbox.entrypoint
      (returns immediately)      |-- God.solve()
                                 |     |-- plan / transform / seal
                                 |     '-- spawn ------------> N sandboxes
                                 |                             (one each, own
                                 |                              volume subpath)
                                 |-- writes modal.Dict  <-- god status (any time)
                                 '-- writes out volume  <-- artifacts, at the end

Nothing in this package is imported by `demigod` or by `reagents`. Deleting it
returns the system to the in-process GOD it had before.
"""

from godbox.status import (
    Phase,
    StatusWriter,
    dict_name,
    list_runs,
    push_followup,
    read_status,
)

__all__ = [
    "Phase",
    "StatusWriter",
    "dict_name",
    "list_runs",
    "push_followup",
    "read_status",
]
