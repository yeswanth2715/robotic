import sys,time,json
from pathlib import Path
import cv2,numpy as np
root=Path(__file__).resolve().parents[1]; fast=root/'code'/'fastdlo_src'/'fastdlo-master'; sys.path.insert(0,str(fast)); np.int=int
_cc=cv2.connectedComponents; cv2.connectedComponents=lambda image,*a,**kw:_cc(np.asarray(image,dtype=np.uint8),*a,**kw)
from fastdlo.core import Pipeline
from . import run_wireseg_experiment as e

def main():
 base=root/'data'/'tum_dlo_dataset'/'m1690303'; records=[]
 for scenario in ('S1','S2','S3'):
  d=json.loads((base/scenario/'annotations.json').read_text()); anns={}
  for ann in d['annotations']: anns.setdefault(ann['image_id'],[]).append(ann)
  for im in d['images']: records.append((scenario,base/scenario/'Visualization'/im['file_name'],anns.get(im['id'],[]),im['width'],im['height']))
 w=fast/'weights'; p=Pipeline(str(w/'CP_siam.pth'),str(w/'CP_seg_extracted.pth'),img_w=640,img_h=360); results=[]; figures=root/'outputs'/'experiment'/'figures'; figures.mkdir(parents=True,exist_ok=True)
 for n,(scenario,path,anns,width,height) in enumerate(records):
  image=cv2.imread(str(path)); resized=cv2.resize(image,(640,360)); truth=np.zeros((height,width),np.uint8)
  for ann in anns:
   for poly in ann.get('segmentation',[]): cv2.fillPoly(truth,[np.asarray(poly,dtype=np.int32).reshape(-1,2)],255)
  truth=cv2.resize(truth,(640,360),interpolation=cv2.INTER_NEAREST)>0; start=time.perf_counter(); mask=np.asarray(p.network_seg.predict_img(resized)).astype(np.uint8); latency=(time.perf_counter()-start)*1000; pred=mask>=77; m=e.metrics(pred,truth); m.update({'method':'FASTDLO_DeepLab_segmentation','scenario':scenario,'file_name':path.name,'latency_ms':latency}); results.append(m)
  if n<6:
   overlay=resized.copy(); overlay[truth & pred]=(45,160,120); overlay[truth & ~pred]=(0,0,220); overlay[pred & ~truth]=(0,200,245); cv2.imwrite(str(figures/f'fastdlo_tum_{n+1:02d}.png'),overlay)
 s={k:float(np.mean([r[k] for r in results])) for k in ('precision','recall','f1','iou','latency_ms')}; s['latency_p95_ms']=float(np.percentile([r['latency_ms'] for r in results],95)); s['fps']=1000/s['latency_ms']; payload={'dataset':'TUM Dataset for Instance Segmentation of Deformable Linear Objects','scenarios':{x:sum(r[0]==x for r in records) for x in ('S1','S2','S3')},'images':len(records),'summary':s,'frame_results':results}; (root/'outputs'/'experiment'/'fastdlo_tum_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(s,indent=2))
if __name__=='__main__': main()
