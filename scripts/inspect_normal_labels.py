from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1] / "data" / "movingcables_sample" / "MovingCables" / "sampled_compositions_small" / "test"
rgb = next((root / "rgb_clips" / "0006").glob("*.png"))
for kind in ("flow_first_back", "normal_flow_first_back"):
    im = Image.open(root / kind / "0006" / rgb.name)
    arr = np.asarray(im)
    print(kind, im.mode, arr.shape, arr.dtype, arr.min(), arr.max(), [int((arr[..., i] != 0).sum()) for i in range(arr.shape[-1])])
    print([np.unique(arr[..., i])[:8].tolist() for i in range(arr.shape[-1])])
