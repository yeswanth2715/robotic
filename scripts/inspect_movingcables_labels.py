from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1] / "data" / "movingcables_sample" / "MovingCables" / "sampled_compositions_small" / "test"
rgb = next((root / "rgb_clips" / "0006").glob("*.png"))
flow = root / "flow_first_back" / "0006" / rgb.name
im = Image.open(flow)
arr = np.asarray(im)
print("rgb", rgb.name, Image.open(rgb).mode, Image.open(rgb).size)
print("flow", im.mode, im.size, arr.shape, arr.dtype, "minmax", arr.min(), arr.max())
if arr.ndim > 2:
    for i in range(arr.shape[-1]):
        print("channel", i, "unique_sample", np.unique(arr[..., i])[:12].tolist(), "nonzero", int((arr[..., i] != 0).sum()))
