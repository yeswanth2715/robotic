import json
import numpy as np
from . import run_wireseg_experiment as e

rows=e.load_rows(); groups={}
for i,row in enumerate(rows): groups.setdefault(row['scene_family'],[]).append((row,i))
test=[]
for _,items in sorted(groups.items()): test += items[int(len(items)*.7):]
results=[]
for row,index in test:
    rgb,truth,_,_=e.pair(row,index); pred=e.otsu(rgb); m=e.metrics(pred,truth); m['method']='otsu_morphology'; m['scene_family']=row['scene_family']; results.append(m)
s={k:float(np.mean([r[k] for r in results])) for k in ('precision','recall','f1','iou')}
(e.OUT/'otsu_results.json').write_text(json.dumps({'summary':s,'rows':len(results)},indent=2),encoding='utf-8')
print(json.dumps(s,indent=2))
