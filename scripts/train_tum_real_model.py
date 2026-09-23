from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tum_dlo_dataset" / "m1690303"
OUT = ROOT / "outputs" / "experiment"
FIG = OUT / "figures"
SIZE = 256
SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def records():
    result = []
    for scenario in ("S1", "S2", "S3"):
        folder = DATA / scenario
        payload = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        by_image = {}
        for ann in payload["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for image in payload["images"]:
            mask = np.zeros((image["height"], image["width"]), np.uint8)
            for ann in by_image.get(image["id"], []):
                for polygon in ann.get("segmentation", []):
                    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [points], 255)
            path = folder / "Visualization" / image["file_name"]
            result.append({"scenario": scenario, "file_name": image["file_name"], "path": str(path), "mask": mask})
    return result


def tensor_pair(item, augment=False):
    image = np.asarray(Image.open(item["path"]).convert("RGB"), dtype=np.float32) / 255.0
    truth = item["mask"] > 0
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    truth = cv2.resize(truth.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
    if augment:
        if random.random() < 0.5:
            image = image[:, ::-1].copy(); truth = truth[:, ::-1].copy()
        if random.random() < 0.4:
            image = np.clip(image * random.uniform(0.75, 1.25), 0, 1)
        if random.random() < 0.25:
            image = np.clip(image + np.random.normal(0, 0.025, image.shape), 0, 1)
    return torch.from_numpy(image.transpose(2, 0, 1)).float(), torch.from_numpy(truth[None].astype(np.float32))


class TUMDataset(Dataset):
    def __init__(self, items, augment=False): self.items, self.augment = items, augment
    def __len__(self): return len(self.items)
    def __getitem__(self, index): return tensor_pair(self.items[index], self.augment)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 12, 3, padding=1), nn.ReLU())
        self.b = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.c = nn.Sequential(nn.ConvTranspose2d(24, 12, 2, stride=2), nn.ReLU(), nn.Conv2d(12, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    def forward(self, x): return self.c(self.b(self.a(x)))


def score(pred, truth):
    tp = np.logical_and(pred, truth).sum(); fp = np.logical_and(pred, ~truth).sum(); fn = np.logical_and(~pred, truth).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0, "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0, "fp_pixels": int(fp), "fn_pixels": int(fn)}


def overlay(image, truth, pred, path, title):
    canvas = Image.fromarray((image * 255).astype(np.uint8)).convert("RGBA")
    layer = np.zeros((SIZE, SIZE, 4), np.uint8)
    layer[truth & pred] = (45, 160, 120, 150); layer[truth & ~pred] = (220, 60, 60, 170); layer[pred & ~truth] = (230, 170, 30, 170)
    canvas.alpha_composite(Image.fromarray(layer, "RGBA"))
    ImageDraw.Draw(canvas).text((6, 6), title, fill=(24, 59, 86, 255))
    canvas.save(path)


def main():
    items = records()
    train = [x for x in items if x["scenario"] in ("S1", "S2")]
    test = [x for x in items if x["scenario"] == "S3"]
    rng = random.Random(SEED); rng.shuffle(train)
    val = train[:max(20, len(train) // 5)]; train = train[len(val):]
    model = TinyUNet()
    pretrained = OUT / "tiny_unet_wireseg_final.pt"
    if pretrained.exists(): model.load_state_dict(torch.load(pretrained, map_location="cpu"))
    loader = DataLoader(TUMDataset(train, True), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
    best, best_state = -1, None
    for epoch in range(12):
        model.train()
        for x, y in loader:
            opt.zero_grad(); logits = model(x); probs = torch.sigmoid(logits)
            dice = 1 - (2 * (probs * y).sum() + 1) / (probs.sum() + y.sum() + 1)
            loss = bce(logits, y) + dice; loss.backward(); opt.step()
        model.eval(); vals = []
        with torch.no_grad():
            for item in val:
                x, y = tensor_pair(item); pred = torch.sigmoid(model(x[None]))[0, 0].numpy() >= 0.5
                vals.append(score(pred, y[0].numpy() > 0.5)["f1"])
        vf1 = float(np.mean(vals))
        if vf1 > best: best, best_state = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "validation_f1": vf1}))
    model.load_state_dict(best_state); model.eval(); results = []; FIG.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, item in enumerate(test):
            image, truth = tensor_pair(item); start = time.perf_counter(); pred = torch.sigmoid(model(image[None]))[0, 0].numpy() >= 0.5; latency = (time.perf_counter() - start) * 1000
            m = score(pred, truth[0].numpy() > 0.5); m.update({"method": "TUM_finetuned_TinyUNet", "scenario": item["scenario"], "file_name": item["file_name"], "latency_ms": latency}); results.append(m)
            if index < 6: overlay(image.permute(1, 2, 0).numpy(), truth[0].numpy() > 0.5, pred, FIG / f"tum_finetuned_{index + 1:02d}.png", f"TUM {item['file_name']}")
    summary = {key: float(np.mean([r[key] for r in results])) for key in ("precision", "recall", "f1", "iou", "latency_ms")}
    summary["latency_p95_ms"] = float(np.percentile([r["latency_ms"] for r in results], 95)); summary["fps"] = 1000 / summary["latency_ms"]
    payload = {"dataset": "TUM Dataset for Instance Segmentation of Deformable Linear Objects", "seed": SEED, "train_scenarios": ["S1", "S2"], "validation_count": len(val), "train_count": len(train), "test_scenario": "S3", "test_count": len(test), "selection": "scenario-held-out test to prevent image-level leakage", "summary": summary, "frame_results": results, "validation_best_f1": best}
    (OUT / "tum_finetuned_tinyunet_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), OUT / "tiny_unet_tum_finetuned.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
