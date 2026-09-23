from pathlib import Path
import torch

w=Path(__file__).resolve().parents[1]/'code'/'fastdlo_src'/'fastdlo-master'/'weights'
obj=torch.load(w/'CP_seg.pth',map_location='cpu')
state=obj.get('model_state_dict',obj)
torch.save(state,w/'CP_seg_extracted.pth')
print(w/'CP_seg_extracted.pth')
