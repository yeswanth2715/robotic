from pathlib import Path
import torch

w=Path(__file__).resolve().parents[1]/'code'/'fastdlo_src'/'fastdlo-master'/'weights'
for name in ('CP_siam.pth','CP_seg.pth'):
    obj=torch.load(w/name,map_location='cpu')
    print(name, type(obj), list(obj.keys())[:8] if isinstance(obj,dict) else None)
