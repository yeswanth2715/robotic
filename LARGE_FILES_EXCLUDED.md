# Large Files Excluded From GitHub Package

These files were intentionally left out because they are raw datasets, checkpoints, caches, or regenerated binaries.

| File category | Example local file | Reason excluded |
| --- | --- | --- |
| Model checkpoint | `outputs/checkpoint_fastdlo_tum_adapted.pt` / `outputs/experiment/fastdlo_tum_adapted.pt` | Large binary model output; should be stored outside Git or regenerated. |
| Cached arrays | `outputs/result_wireseg_cached_split.npz` / `outputs/experiment/wireseg_cached_split.npz` | Large intermediate cache; JSON results and figures are retained instead. |
| Raw TUM archive | `dataset/tum_dlo_dataset.zip` / `data/tum_dlo_dataset.zip` | Public dataset link is provided; archive should not be committed. |
| Raw MovingCables archive | `dataset/MovingCables_sample.tar` | Public dataset link is provided; archive should not be committed. |
| FASTDLO source archive/weights | `dataset/fastdlo.zip`, `code/fastdlo_src/**/weights/*` | Third-party code/weights are external dependencies. |
| Extracted dataset folders | `data/tum_dlo_dataset/`, `data/movingcables_sample/` | Recreated from public downloads; too large/noisy for GitHub. |
| Python caches | `__pycache__/`, `*.pyc` | Regenerated automatically. |
