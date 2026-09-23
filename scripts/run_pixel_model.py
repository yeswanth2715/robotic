from pathlib import Path
import json, time
import numpy as np
from skimage.color import rgb2hsv
from sklearn.linear_model import LogisticRegression
from . import run_wireseg_experiment as e

def feats(rgb):
    hsv=rgb2hsv(rgb); return np.concatenate([rgb,hsv],axis=2).reshape(-1,6)
def main():
    rows=e.load_rows(); groups={}
    for i,row in enumerate(rows): groups.setdefault(row['scene_family'],[]).append((row,i))
    train=[]; test=[]
    for _,items in sorted(groups.items()): cut=int(len(items)*.7); train+=items[:cut]; test+=items[cut:]
    rng=np.random.default_rng(e.SEED); xs=[]; ys=[]
    for row,index in train:
        rgb,truth,_,_=e.pair(row,index); x=feats(rgb); y=truth.reshape(-1); pos=np.flatnonzero(y); neg=np.flatnonzero(~y); n=min(len(pos),len(neg),2000); idx=np.r_[rng.choice(pos,n,replace=False),rng.choice(neg,n,replace=False)]; xs.append(x[idx]); ys.append(y[idx])
    model=LogisticRegression(max_iter=100,class_weight='balanced',solver='lbfgs'); model.fit(np.vstack(xs),np.concatenate(ys)); print('trained')
    results=[]
    for row,index in test:
        rgb,truth,_,_=e.pair(row,index); t=time.perf_counter(); pred=model.predict_proba(feats(rgb))[:,1].reshape(e.SIZE,e.SIZE)>=.5; latency=(time.perf_counter()-t)*1000; m=e.metrics(pred,truth); m.update({'method':'rgb_hsv_logistic','scene_family':row['scene_family'],'file_name':row['file_name'],'latency_ms':latency,'candidate_count':e.candidate_count(pred)}); results.append(m)
    s={k:float(np.mean([r[k] for r in results])) for k in ('precision','recall','f1','iou','latency_ms','candidate_count')}; s['latency_p95_ms']=float(np.percentile([r['latency_ms'] for r in results],95)); s['fps']=1000/s['latency_ms']; payload={'dataset':'DeformX/WireSeg-36K','seed':e.SEED,'train_rows':len(train),'test_rows':len(test),'summary':{'rgb_hsv_logistic':s},'frame_results':results}; (e.OUT/'pixel_model_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(s,indent=2))
if __name__=='__main__': main()
