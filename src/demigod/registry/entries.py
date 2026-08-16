"""THE CLOSED SET.

Two entries today: `pandas` (the reference implementation -- copy its shape) and
`imaging`.

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

IMAGING = ToolEntry(
    key="imaging",
    display_name="imaging (scikit-image, cellpose, tifffile, zarr)",
    # Resolved together against the `pandas` entry's pins with `uv pip compile`,
    # NOT pinned one at a time. The two entries can land in the same image, and
    # `PrebakedImage.build` concatenates their `install` lists into a single
    # `pip_install` -- so `pandas==3.0.5` here (what these packages resolve to on
    # their own) against `pandas==2.2.3` there would be an unsatisfiable
    # requirement set and a failed bake, not a version someone picked wrong.
    #
    # torch and torchvision are deliberately ABSENT even though cellpose needs
    # them. They come from `PrebakedImage.base_pip` as `+cpu` builds off the
    # PyTorch index; listed here they would resolve to the PyPI default, which
    # on linux bundles CUDA and costs several GB in an image with no GPU.
    install=(
        "numpy==2.1.3",
        "scipy==1.18.0",
        "pandas==2.2.3",
        "scikit-image==0.26.0",
        "tifffile==2026.8.16",
        "imagecodecs==2026.8.16",
        "ome-zarr==0.18.0",
        "zarr==3.3.0",
        "dask==2026.7.1",
        "matplotlib==3.11.1",
        "cellpose==4.2.1.1",
    ),
    secrets=(),  # offline. Reading an image with a MODEL is vision.read_image,
    # which is brokered precisely so no credential comes in here.
    apt=(),
    smoke_test=(
        "python",
        "-c",
        # Exercises the pipeline an agent actually runs -- threshold, label,
        # round-trip through both file formats -- rather than importing eleven
        # packages and declaring victory. Two squares in, two objects out.
        "import matplotlib; matplotlib.use('Agg'); "
        "import numpy as np, tifffile, zarr, imagecodecs, scipy, pandas, ome_zarr; "
        "import dask.array as da, matplotlib.pyplot as plt; "
        "from skimage import filters, measure; "
        "from importlib.metadata import version; "
        "img = np.zeros((64, 64), 'uint16'); "
        "img[8:24, 8:24] = 4000; img[40:56, 40:56] = 4000; "
        "n = int(measure.label(img > filters.threshold_otsu(img)).max()); "
        "assert n == 2, f'segmentation found {n} objects, expected 2'; "
        "tifffile.imwrite('/tmp/smoke.tif', img); "
        "assert tifffile.imread('/tmp/smoke.tif').shape == (64, 64); "
        "z = zarr.open(store='/tmp/smoke.zarr', mode='w', shape=(64, 64), "
        "dtype='uint16'); z[:] = img; assert int(np.asarray(z[:]).max()) == 4000; "
        "assert float(da.from_array(img, chunks=32).mean().compute()) > 0; "
        "plt.imsave('/tmp/smoke.png', img); "
        "import torch; from cellpose import models, core; core.use_gpu(); "
        "print('imaging ok', 'skimage', version('scikit-image'), "
        "'cellpose', version('cellpose'), 'torch', torch.__version__)",
    ),
    doc_file="imaging.md",
    tags=("imaging", "microscopy", "segmentation", "vision", "analysis"),
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
        IMAGING,
        # MY_TOOL,
    )
}
