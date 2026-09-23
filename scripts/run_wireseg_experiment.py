from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wireseg"
OUT = ROOT / "outputs" / "experiment"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 17
SIZE = 256
np.random.seed(SEED)
torch.manual_seed(SEED)


def decode_rle(encoded: str, height: int, width: int) -> np.ndarray:
    counts, pos = [], 0
    while pos < len(encoded):
        value, shift = 0, 0
        while True:
            code = ord(encoded[pos]) - 48
            pos += 1
            value |= (code & 0x1F) << shift
            if not (code & 0x20):
                break
            shift += 5
        if len(counts) > 1:
            value += counts[-2]
        counts.append(value)
    flat = np.zeros(height * width, dtype=bool)
    cursor = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[cursor:cursor + count] = True
        cursor += count
    return flat.reshape((height, width), order="F")


def load_rows():
    rows = []
    for path in sorted((ROOT / "data").glob("wireseg_rows_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["rows"]:
            row = item["row"]
            row["api_source"] = path.name
            rows.append(row)
    if len(rows) < 300:
        raise RuntimeError(f"Expected 300 WireSeg rows, found {len(rows)}")
    return rows[:300]


def materialise(row, index):
    image_path = DATA / "images" / f"{index:04d}.jpg"
    mask_path = DATA / "masks" / f"{index:04d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image_path.write_bytes(urlopen(row["image"]["src"], timeout=60).read())
    if not mask_path.exists():
        mask = np.zeros((1024, 1024), dtype=bool)
        for encoded in json.loads(row["masks_rle"]):
            mask |= decode_rle(encoded["counts"], encoded["size"][0], encoded["size"][1])
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    return image_path, mask_path


def pair(row, index):
    image_path, mask_path = materialise(row, index)
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    truth = np.asarray(Image.open(mask_path).convert("L")) > 0
    rgb = resize(rgb, (SIZE, SIZE), anti_aliasing=True, preserve_range=True).astype(np.float32)
    truth = resize(truth.astype(np.float32), (SIZE, SIZE), order=0, preserve_range=True) > 0.5
    return rgb, truth, image_path, mask_path


class WireDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        rgb, truth, _, _ = pair(*self.items[index])
        return torch.from_numpy(rgb.transpose(2, 0, 1)), torch.from_numpy(truth[None].astype(np.float32))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 12, 3, padding=1), nn.ReLU())
        self.b = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.c = nn.Sequential(nn.ConvTranspose2d(24, 12, 2, stride=2), nn.ReLU(), nn.Conv2d(12, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    def forward(self, x): return self.c(self.b(self.a(x)))


def otsu(rgb):
    gray = rgb.mean(axis=2)
    cut = threshold_otsu(gray)
    a, b = gray > cut, gray <= cut
    border_a = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean()
    border_b = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]]).mean()
    return remove_small_objects(a if border_a < border_b else b, min_size=10)


def metrics(pred, truth):
    tp = np.logical_and(pred, truth).sum(); fp = np.logical_and(pred, ~truth).sum(); fn = np.logical_and(~pred, truth).sum()
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(2 * p * r / (p + r)) if p + r else 0.0, "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0, "fp_pixels": int(fp), "fn_pixels": int(fn)}


def candidate_count(mask):
    y, x = np.where(skeletonize(mask))
    return int(min(5, len(x)))


def save_overlay(rgb, truth, pred, path, title):
    image = Image.fromarray((rgb * 255).astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0)); pix = np.asarray(overlay)
    pix[truth & ~pred] = (232, 80, 80, 125); pix[pred & ~truth] = (245, 190, 50, 125); pix[truth & pred] = (45, 160, 120, 125)
    overlay = Image.fromarray(pix, "RGBA"); image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image); draw.rectangle((3, 3, 210, 25), fill=(24, 59, 86, 220)); draw.text((8, 7), title, fill="white")
    image.save(path)


def main():
    rows = load_rows()
    groups = {}
    for index, row in enumerate(rows): groups.setdefault(row["scene_family"], []).append((row, index))
    train, test = [], []
    for family, items in sorted(groups.items()):
        cut = int(len(items) * 0.7); train.extend(items[:cut]); test.extend(items[cut:])
    loader = DataLoader(WireDataset(train), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = TinyUNet(); opt = torch.optim.Adam(model.parameters(), lr=0.002); loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(6):
        for x, y in loader:
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    torch.save(model.state_dict(), OUT / "tiny_unet_wireseg.pt")
    frame_results = []; examples = []
    model.eval()
    with torch.no_grad():
        for row, index in test:
            rgb, truth, image_path, _ = pair(row, index)
            t = time.perf_counter(); base = otsu(rgb); base_ms = (time.perf_counter() - t) * 1000
            x = torch.from_numpy(rgb.transpose(2, 0, 1)[None]); t = time.perf_counter(); pred = torch.sigmoid(model(x))[0, 0].numpy() >= 0.5; nn_ms = (time.perf_counter() - t) * 1000
            for method, output, latency in (("otsu_morphology", base, base_ms), ("tiny_unet", pred, nn_ms)):
                m = metrics(output, truth); m.update({"method": method, "scene_family": row["scene_family"], "file_name": row["file_name"], "latency_ms": latency, "candidate_count": candidate_count(output)})
                frame_results.append(m)
                if method == "tiny_unet" and len(examples) < 6:
                    overlay = OUT / "figures"; overlay.mkdir(parents=True, exist_ok=True); target = overlay / f"wireseg_{len(examples)+1:02d}.png"; save_overlay(rgb, truth, output, target, f"{row['scene_family']} | {row['file_name'].split('/')[-1]}"); examples.append(str(target.relative_to(ROOT)))
    summary = {}
    for method in ("otsu_morphology", "tiny_unet"):
        subset = [x for x in frame_results if x["method"] == method]; avg = {key: float(np.mean([x[key] for x in subset])) for key in ("precision", "recall", "f1", "iou", "latency_ms", "candidate_count")}; avg["latency_p95_ms"] = float(np.percentile([x["latency_ms"] for x in subset], 95)); avg["fps"] = float(1000 / avg["latency_ms"]); summary[method] = avg
    result = {"dataset": "DeformX/WireSeg-36K", "licence": "CC BY 4.0", "seed": SEED, "rows": len(rows), "train_rows": len(train), "test_rows": len(test), "scene_families": {k: len(v) for k, v in groups.items()}, "split": "first 70 percent per scene family for training, final 30 percent for held-out testing", "summary": summary, "frame_results": frame_results, "qualitative_figures": examples, "limitations": ["The API subset contains 300 selected rows from the 36,000-image dataset.", "WireSeg masks are synthetic foregrounds composited over real backgrounds; transfer to physical cable scenes is not established.", "Candidate validity is a geometric proxy because the dataset has no physical grasp-success labels.", "CPU latency is hardware-specific and does not establish robot-controller real-time performance."]}
    (OUT / "wireseg_experiment_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"train_rows": len(train), "test_rows": len(test), "scene_families": result["scene_families"], "summary": summary}, indent=2))


if __name__ == "__main__": main()
