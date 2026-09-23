import json
import numpy as np
import torch
from pathlib import Path
from . import run_wireseg_experiment as e

root=Path(__file__).resolve().parents[1]; out=e.OUT
cache=np.load(out/'wireseg_cached_split.npz'); x=torch.from_numpy(cache['x_test']); y=cache['y_test'][:,0]>0.5
model=e.TinyUNet(); model.load_state_dict(torch.load(out/'tiny_unet_wireseg_final.pt',map_location='cpu')); model.eval()
with torch.no_grad(): scores=torch.sigmoid(model(x)).numpy()[:,0]
rows=[]
for threshold in np.arange(.05,1.0,.05):
    pred=scores>=threshold; tp=np.logical_and(pred,y).sum(); fp=np.logical_and(pred,~y).sum(); fn=np.logical_and(~pred,y).sum(); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0; i=tp/(tp+fp+fn) if tp+fp+fn else 0; rows.append({'threshold':round(float(threshold),2),'precision':float(p),'recall':float(r),'f1':float(f),'iou':float(i)})
best=max(rows,key=lambda r:r['f1']); (out/'threshold_analysis.json').write_text(json.dumps({'rows':rows,'best_by_f1':best},indent=2),encoding='utf-8'); print(json.dumps({'best_by_f1':best,'rows':rows},indent=2))
