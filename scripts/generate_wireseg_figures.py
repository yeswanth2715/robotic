from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from . import run_wireseg_experiment as e

def main():
    rows=e.load_rows(); groups={}
    for i,row in enumerate(rows): groups.setdefault(row['scene_family'],[]).append((row,i))
    test=[]
    for _,items in sorted(groups.items()): test += items[int(len(items)*.7):]
    model=e.TinyUNet(); model.load_state_dict(torch.load(e.OUT/'tiny_unet_wireseg_weighted.pt',map_location='cpu')); model.eval()
    out=e.OUT/'figures'; out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for n,(row,index) in enumerate(test[:4],1):
            rgb,truth,_,_=e.pair(row,index); x=torch.from_numpy(rgb.transpose(2,0,1)[None]); pred=torch.sigmoid(model(x))[0,0].numpy()>=.5
            image=Image.fromarray((rgb*255).astype(np.uint8)).convert('RGBA'); overlay=Image.new('RGBA',image.size,(0,0,0,0)); pix=np.asarray(overlay).copy(); pix[truth & ~pred]=(232,80,80,125); pix[pred & ~truth]=(245,190,50,125); pix[truth & pred]=(45,160,120,125); image.alpha_composite(Image.fromarray(pix,'RGBA')); ImageDraw.Draw(image).rectangle((3,3,240,25),fill=(24,59,86,220)); ImageDraw.Draw(image).text((8,7),f"WireSeg {row['scene_family']} | GT/pred",fill='white'); image.save(out/f'wireseg_result_{n:02d}.png')
    print(f'figures={len(list(out.glob("wireseg_result_*.png")))}')
if __name__=='__main__': main()
