from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from . import run_wireseg_experiment as e

def main():
    torch.set_num_threads(4)
    rows=e.load_rows(); groups={}
    for i,row in enumerate(rows): groups.setdefault(row['scene_family'],[]).append((row,i))
    train=[]; test=[]
    for _,items in sorted(groups.items()):
        cut=int(len(items)*.7); train += items[:cut]; test += items[cut:]
    def cache(items):
        xs=[]; ys=[]
        for row,index in items:
            rgb,truth,_,_=e.pair(row,index); xs.append(rgb.transpose(2,0,1)); ys.append(truth[None].astype(np.float32))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
    xtr,ytr=cache(train); xte,yte=cache(test)
    np.savez_compressed(e.OUT/'wireseg_cached_split.npz',x_train=xtr.numpy(),y_train=ytr.numpy(),x_test=xte.numpy(),y_test=yte.numpy())
    loader=DataLoader(TensorDataset(xtr,ytr),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(e.SEED))
    model=e.TinyUNet(); opt=torch.optim.Adam(model.parameters(),lr=.001); bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0]))
    for epoch in range(10):
        for x,y in loader:
            opt.zero_grad(); logits=model(x); probs=torch.sigmoid(logits); dice=1-(2*(probs*y).sum()+1)/(probs.sum()+y.sum()+1); (bce(logits,y)+dice).backward(); opt.step()
        print(f'epoch={epoch+1}/10')
    torch.save(model.state_dict(),e.OUT/'tiny_unet_wireseg_final.pt'); model.eval(); results=[]
    with torch.no_grad():
        for n,(row,index) in enumerate(test):
            rgb,truth,_,_=e.pair(row,index); x=torch.from_numpy(rgb.transpose(2,0,1)[None]); start=time.perf_counter(); pred=torch.sigmoid(model(x))[0,0].numpy()>=.5; latency=(time.perf_counter()-start)*1000; m=e.metrics(pred,truth); m.update({'method':'tiny_unet_final','scene_family':row['scene_family'],'file_name':row['file_name'],'latency_ms':latency,'candidate_count':e.candidate_count(pred)}); results.append(m)
            if n<4:
                image=Image.fromarray((rgb*255).astype(np.uint8)).convert('RGBA'); pix=np.zeros((e.SIZE,e.SIZE,4),dtype=np.uint8); pix[truth & ~pred]=(232,80,80,125); pix[pred & ~truth]=(245,190,50,125); pix[truth & pred]=(45,160,120,125); image.alpha_composite(Image.fromarray(pix,'RGBA')); ImageDraw.Draw(image).rectangle((3,3,230,25),fill=(24,59,86,220)); ImageDraw.Draw(image).text((8,7),f"WireSeg {row['scene_family']} | final",fill='white'); image.save(e.OUT/'figures'/f'wireseg_final_{n+1:02d}.png')
    s={k:float(np.mean([r[k] for r in results])) for k in ('precision','recall','f1','iou','latency_ms','candidate_count')}; s['latency_p95_ms']=float(np.percentile([r['latency_ms'] for r in results],95)); s['fps']=1000/s['latency_ms']
    payload={'dataset':'DeformX/WireSeg-36K','licence':'CC BY 4.0','seed':e.SEED,'rows':len(rows),'train_rows':len(train),'test_rows':len(test),'scene_families':{k:len(v) for k,v in groups.items()},'summary':{'tiny_unet_weighted':s},'frame_results':results,'training':'Tiny U-Net, BCEWithLogitsLoss(pos_weight=12)+Dice loss, 10 CPU epochs, cached 256px tensors'}
    (e.OUT/'wireseg_final_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(s,indent=2))
if __name__=='__main__': main()
