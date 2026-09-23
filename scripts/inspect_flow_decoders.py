from pathlib import Path
import numpy as np
import imageio.v3 as iio

root = Path(__file__).resolve().parents[1] / "data" / "movingcables_sample" / "MovingCables" / "sampled_compositions_small" / "test"
name = next((root / "rgb_clips" / "0006").glob("*.png")).name
for kind in ("flow_first_back", "normal_flow_first_back"):
    arr = iio.imread(root / kind / "0006" / name)
    print(kind, arr.shape, arr.dtype, arr.min(), arr.max(), np.unique(arr[..., 2])[:10].tolist())
