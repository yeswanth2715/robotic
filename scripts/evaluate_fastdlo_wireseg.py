import sys,time,json
from pathlib import Path
import cv2,numpy as np
root=Path(__file__).resolve().parents[1]; fast=root/'code'/'fastdlo_src'/'fastdlo-master'; sys.path.insert(0,str(fast)); np.int=int
_cc=cv2.connectedComponents; cv2.connectedComponents=lambda image,*a,**kw:_cc(np.asarray(image,dtype=np.uint8),*a,**kw)
from fastdlo.core import Pipeline
from . import run_wireseg_experiment as e

def main():
 rows=e.load_rows(); groups={}
 for i,row in enumerate(rows): groups.setdefault(row['scene_family'],[]).append((row,i))
 test=[]
 for _,items in sorted(groups.items()): test+=items[int(len(items)*.7):]
 w=fast/'weights'; p=Pipeline(str(w/'CP_siam.pth'),str(w/'CP_seg_extracted.pth'),img_w=640,img_h=360); results=[]; fig=root/'outputs'/'experiment'/'figures'; fig.mkdir(parents=True,exist_ok=True)
 for n,(row,index) in enumerate(test):
  image_path,mask_path=e.materialise(row,index); image=cv2.resize(cv2.imread(str(image_path)),(640,360)); truth=cv2.resize(cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE),(640,360),interpolation=cv2.INTER_NEAREST)>0; start=time.perf_counter(); mask=np.asarray(p.network_seg.predict_img(image)).astype(np.uint8); latency=(time.perf_counter()-start)*1000; pred=mask>=77; m=e.metrics(pred,truth); m.update({'method':'FASTDLO_DeepLab_segmentation','scene_family':row['scene_family'],'file_name':row['file_name'],'latency_ms':latency}); results.append(m)
  if n<4:
   overlay=image.copy(); overlay[truth & pred]=(45,160,120); overlay[truth & ~pred]=(0,0,220); overlay[pred & ~truth]=(0,200,245); cv2.imwrite(str(fig/f'fastdlo_result_{n+1:02d}.png'),overlay)
 s={k:float(np.mean([r[k] for r in results])) for k in ('precision','recall','f1','iou','latency_ms')}; s['latency_p95_ms']=float(np.percentile([r['latency_ms'] for r in results],95)); s['fps']=1000/s['latency_ms']; payload={'dataset':'DeformX/WireSeg-36K','seed':e.SEED,'test_rows':len(test),'summary':{'FASTDLO_DeepLab_segmentation':s},'frame_results':results,'note':'Official FASTDLO segmentation checkpoint evaluated on the held-out WireSeg subset; full grasp-pipeline smoke test separately measured 725.25 ms on the published test image.'}; (e.OUT/'fastdlo_wireseg_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(s,indent=2))
if __name__=='__main__': main()
