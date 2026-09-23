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
FAST = ROOT / "code" / "fastdlo_src" / "fastdlo-master"
OUT = ROOT / "outputs" / "experiment"
FIG = OUT / "figures"
W, H = 320, 180
SEED = 17
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
sys.path.insert(0, str(FAST))
import fastdlo.seg_net.model as network  # noqa: E402


def records():
    result = []
    for scenario in ("S1", "S2", "S3"):
        folder = DATA / scenario; payload = json.loads((folder / "annotations.json").read_text(encoding="utf-8")); by_image = {}
        for ann in payload["annotations"]: by_image.setdefault(ann["image_id"], []).append(ann)
        for image in payload["images"]:
            mask = np.zeros((image["height"], image["width"]), np.uint8)
            for ann in by_image.get(image["id"], []):
                for poly in ann.get("segmentation", []): cv2.fillPoly(mask, [np.asarray(poly, np.int32).reshape(-1, 2)], 255)
            result.append({"scenario": scenario, "file_name": image["file_name"], "path": str(folder / "Visualization" / image["file_name"]), "mask": mask})
    return result


def pair(item, augment=False):
    image = np.asarray(Image.open(item["path"]).convert("RGB"), np.float32) / 255.0; truth = item["mask"] > 0
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA); truth = cv2.resize(truth.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    if augment:
        if random.random() < 0.5: image = image[:, ::-1].copy(); truth = truth[:, ::-1].copy()
        if random.random() < 0.4: image = np.clip(image * random.uniform(0.8, 1.2), 0, 1)
    return torch.from_numpy(image.transpose(2, 0, 1)).float(), torch.from_numpy(truth[None].astype(np.float32))


class TUM(Dataset):
    def __init__(self, items, augment=False): self.items, self.augment = items, augment
    def __len__(self): return len(self.items)
    def __getitem__(self, index): return pair(self.items[index], self.augment)


def score(pred, truth):
    tp = np.logical_and(pred, truth).sum(); fp = np.logical_and(pred, ~truth).sum(); fn = np.logical_and(~pred, truth).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0, "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0, "fp_pixels": int(fp), "fn_pixels": int(fn)}


def main():
    items = records(); train = [x for x in items if x["scenario"] in ("S1", "S2")]; test = [x for x in items if x["scenario"] == "S3"]
    rng = random.Random(SEED); rng.shuffle(train); val = train[:33]; train = train[33:]
    model = network.deeplabv3plus_resnet101(num_classes=1, output_stride=16, pretrained_backbone=False); network.convert_to_separable_conv(model.classifier)
    checkpoint = FAST / "weights" / "CP_seg_extracted.pth"; model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    for parameter in model.backbone.parameters(): parameter.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0002); bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0])); loader = DataLoader(TUM(train, True), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    best, best_state = -1, None
    for epoch in range(4):
        model.train(); model.backbone.eval()
        for x, y in loader:
            opt.zero_grad(); logits = model(x); probs = torch.sigmoid(logits); dice = 1 - (2 * (probs * y).sum() + 1) / (probs.sum() + y.sum() + 1); loss = bce(logits, y) + dice; loss.backward(); opt.step()
        model.eval(); values = []
        with torch.no_grad():
            for item in val:
                x, y = pair(item); pred = torch.sigmoid(model(x[None]))[0, 0].numpy() >= 0.5; values.append(score(pred, y[0].numpy() > 0.5)["f1"])
        vf1 = float(np.mean(values)); print(json.dumps({"epoch": epoch + 1, "validation_f1": vf1}))
        if vf1 > best: best, best_state = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval(); results = []; FIG.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, item in enumerate(test):
            image, truth = pair(item); start = time.perf_counter(); pred = torch.sigmoid(model(image[None]))[0, 0].numpy() >= 0.5; latency = (time.perf_counter() - start) * 1000; m = score(pred, truth[0].numpy() > 0.5); m.update({"method": "FASTDLO_ResNet101_TUM_adapted", "scenario": "S3", "file_name": item["file_name"], "latency_ms": latency}); results.append(m)
            if index < 6:
                canvas = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8); canvas[truth[0].numpy() > 0.5] = [50, 170, 125]; canvas[np.logical_and(truth[0].numpy() > 0.5, ~pred)] = [220, 60, 60]; canvas[np.logical_and(pred, ~(truth[0].numpy() > 0.5))] = [230, 170, 30]; Image.fromarray(canvas).save(FIG / f"fastdlo_tum_adapted_{index + 1:02d}.png")
    summary = {key: float(np.mean([r[key] for r in results])) for key in ("precision", "recall", "f1", "iou", "latency_ms")}; summary["latency_p95_ms"] = float(np.percentile([r["latency_ms"] for r in results], 95)); summary["fps"] = 1000 / summary["latency_ms"]
    payload = {"dataset": "TUM Dataset for Instance Segmentation of Deformable Linear Objects", "seed": SEED, "train_scenarios": ["S1", "S2"], "train_count": len(train), "validation_count": len(val), "test_scenario": "S3", "test_count": len(test), "selection": "scenario-held-out test; published FASTDLO segmentation checkpoint with frozen ResNet-101 backbone and adapted classifier", "validation_best_f1": best, "summary": summary, "frame_results": results}
    (OUT / "fastdlo_tum_adapted_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8"); torch.save(model.state_dict(), OUT / "fastdlo_tum_adapted.pt"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
