"""THE CLOSED SET.

Exactly one real entry today: `pandas`. It is the reference implementation --
copy its shape.

To add a tool, do NOT freehand it. Follow
`.claude/skills/add-tool-to-registry/SKILL.md`, which walks the five steps that
matter: check it isn't already here, write the entry, decide whether it fits an
existing pre-baked image or needs a new one, write the agent-facing usage doc,
add and RUN the smoke test.
"""

from __future__ import annotations

from demigod.registry import ToolEntry

PANDAS = ToolEntry(
    key="pandas",
    display_name="pandas",
    install=("pandas==2.2.3", "numpy==2.1.3", "pyarrow==18.1.0"),
    secrets=(),  # offline, no credentials
    apt=(),
    smoke_test=(
        "python",
        "-c",
        "import pandas as pd; "
        "df = pd.DataFrame({'a': [1, 2, 3]}); "
        "assert df['a'].sum() == 6; "
        "import pyarrow; "
        "print('pandas', pd.__version__, 'ok')",
    ),
    doc_file="pandas.md",
    tags=("data", "tabular", "analysis"),
)

# --- Add new entries above this line, then register them below. -------------
#
# Template:
#
#   MY_TOOL = ToolEntry(
#       key="my-tool",
#       display_name="My Tool",
#       install=("my-tool==1.2.3",),      # PIN. Unpinned installs make images
#       secrets=("MY_TOOL_API_KEY",),     # non-reproducible.
#       apt=(),
#       smoke_test=("python", "-c", "import my_tool; print('ok')"),
#       doc_file="my_tool.md",
#       tags=("...",),
#   )

REGISTRY: dict[str, ToolEntry] = {
    entry.key: entry
    for entry in (
        PANDAS,
        # MY_TOOL,
    )
}
