## imaging

Image analysis. `scikit-image`, `cellpose`, `tifffile`, `imagecodecs`,
`ome-zarr`/`zarr`, `dask`, `matplotlib` — plus `numpy`, `scipy` and `pandas`.

**Use it by writing and running Python scripts.** Write the script into your
output directory, run it with `python`, keep the file. The script is how someone
re-runs your measurement.

`matplotlib` has no display here. Set the backend before importing pyplot or
every figure raises:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Reading — the format tells you which loader:

```python
import tifffile, zarr, numpy as np
from skimage import io

img = tifffile.imread("/run/shared/field.tif")  # TIFF/OME-TIFF, incl. stacks
img = io.imread("/run/shared/plate.png")  # PNG/JPEG
group = zarr.open("/run/shared/plate.ome.zarr", mode="r")  # OME-Zarr, lazy
level0 = np.asarray(group["0"][:])  # highest-resolution level
```

Segment and measure — `regionprops_table` gives you a DataFrame directly:

```python
import pandas as pd
from skimage import filters, measure, morphology

mask = img > filters.threshold_otsu(img)
mask = morphology.remove_small_objects(mask, 64)
labels = measure.label(mask)
props = pd.DataFrame(
    measure.regionprops_table(
        labels,
        intensity_image=img,
        properties=("label", "area", "mean_intensity", "eccentricity", "centroid"),
    )
)
props.to_csv("/run/out/objects.csv", index=False)
```

Cellpose, when thresholding is not enough (touching or non-convex cells):

```python
from cellpose import models

model = models.CellposeModel(gpu=False)  # CPU only in this sandbox
masks, flows, styles = model.eval(img, diameter=None)
print(int(masks.max()), "cells")
```

Traps that cost real time here:

- **`/run/shared` is read-only.** Writing there raises `OSError`. Copy first.
- **Cellpose downloads its weights on first use, over the network.** That is
  tens of seconds and it can fail outright — if your sandbox has a restricted
  egress allowlist it *will* fail. Try `filters.threshold_*` +
  `segmentation.watershed` first; reach for cellpose when you can say why
  thresholding was insufficient. Never let a failed download become a silent
  zero-cell result: check `masks.max()`.
- **Scientific images are usually 16-bit; viewers and PNG are 8-bit.** Do not
  cast with `.astype("uint8")` — it wraps and turns bright objects black. Use
  `skimage.exposure.rescale_intensity(img, in_range=(lo, hi), out_range="uint8")`
  and record the window you chose, because it changes what is visible.
- **TIFF axis order is not guaranteed.** A stack may be `(Z, Y, X)`, `(C, Y, X)`
  or `(T, Z, C, Y, X)`. Read `tifffile.TiffFile(path).series[0].axes` instead of
  assuming, and print `img.shape` immediately after loading.
- **`measure.label` counts background as 0**, so `labels.max()` is the object
  count — `len(np.unique(labels))` is one more than that.
- **Every measurement is in pixels until you convert it.** An area in px² is not
  a result. Find the pixel size (OME metadata via
  `tifffile.TiffFile(path).ome_metadata`, or the run's own inputs) and state the
  scale you used, or record it in `unknowns`.
- **Big images: use dask, do not load the stack.** `dask.array.from_zarr(...)`
  or `tifffile.imread(path, aszarr=True)` keeps it out of RAM. An OOM kill loses
  the whole sandbox, including work you had not written to `/run/out` yet.

### Measuring is not seeing

This stack turns pixels into numbers. It cannot tell you that a field is out of
focus, that a well is contaminated, that a blot lane is smeared, or that your
segmentation latched onto debris. For that, use the brokered `vision.read_image`
tool if it is in your lease — it is the only thing here that looks at an image
rather than computing over it. The productive loop is: segment locally, render
an overlay, and ask the vision tool whether the overlay matches what is actually
in the frame.

```python
from skimage.color import label2rgb

plt.imsave("/run/out/overlay.png", label2rgb(labels, image=img, bg_label=0))
```

```bash
python - <<'EOF' > /run/out/vision_args.json
import base64, json
print(json.dumps({
    "question": "Do the outlined regions correspond to whole cells? Name anything outlined that is debris, and any obvious cell that was missed.",
    "image_base64": base64.b64encode(open("/run/out/overlay.png","rb").read()).decode(),
    "media_type": "image/png",
    "detail": "high",
}))
EOF
toolbox call vision.read_image -i /run/out/vision_args.json -o /run/out/vision_check.json
```

Save every figure you send — the overlay is the evidence that your count means
what you say it means. When you state a number in your `claim`, the script that
produced it must be in `evidence`.
