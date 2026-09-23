from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]/'data'/'tum_dlo_dataset'
for p in root.rglob('annotations.json'):
 d=json.loads(p.read_text()); print(p, len(d.get('images',[])),len(d.get('annotations',[])),d.get('categories'))
