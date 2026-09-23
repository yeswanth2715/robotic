from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from . import run_wireseg_experiment as e

def main():
    rows = e.load_rows(); groups = {}
    for i, row in enumerate(rows): groups.setdefault(row['scene_family'], []).append((row, i))
    train, test = [], []
    for family, items in sorted(groups.items()):
        cut = int(len(items) * .7); train += items[:cut]; test += items[cut:]
    loader = DataLoader(e.WireDataset(train), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(e.SEED))
    model = e.TinyUNet(); opt = torch.optim.Adam(model.parameters(), lr=.002); bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
    for epoch in range(4):
        for x, y in loader:
            opt.zero_grad(); logits = model(x); probs = torch.sigmoid(logits)
            dice = 1 - (2 * (probs * y).sum() + 1) / (probs.sum() + y.sum() + 1)
            (bce(logits, y) + dice).backward(); opt.step()
        print(f'epoch={epoch + 1}/4')
    out = e.OUT; torch.save(model.state_dict(), out / 'tiny_unet_wireseg_weighted.pt'); results=[]
    model.eval()
    with torch.no_grad():
        for row, index in test:
            rgb, truth, _, _ = e.pair(row, index); x=torch.from_numpy(rgb.transpose(2,0,1)[None]); t=time.perf_counter(); pred=torch.sigmoid(model(x))[0,0].numpy()>=.5; latency=(time.perf_counter()-t)*1000
            m=e.metrics(pred, truth); m.update({'method':'tiny_unet_weighted','scene_family':row['scene_family'],'file_name':row['file_name'],'latency_ms':latency,'candidate_count':e.candidate_count(pred)}); results.append(m)
    s={k:float(np.mean([r[k] for r in results])) for k in ('precision','recall','f1','iou','latency_ms','candidate_count')}; s['latency_p95_ms']=float(np.percentile([r['latency_ms'] for r in results],95)); s['fps']=1000/s['latency_ms']
    payload={'dataset':'DeformX/WireSeg-36K','licence':'CC BY 4.0','seed':e.SEED,'rows':len(rows),'train_rows':len(train),'test_rows':len(test),'summary':{'tiny_unet_weighted':s},'frame_results':results,'training':'BCEWithLogitsLoss(pos_weight=8)+Dice loss, 4 CPU epochs'}
    (out/'wireseg_weighted_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(s,indent=2))
if __name__ == '__main__': main()
