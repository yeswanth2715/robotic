from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
path = root / "outputs" / "experiment" / "fastdlo_tum_adapted_results.json"
data = json.loads(path.read_text(encoding="utf-8"))
rng = np.random.default_rng(17)
results = data["frame_results"]
statistics = {}
for metric in ("precision", "recall", "f1", "iou"):
    values = np.asarray([row[metric] for row in results], dtype=float)
    samples = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    statistics[metric] = {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "bootstrap_95_ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }
data["statistics"] = statistics
path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(statistics, indent=2))
