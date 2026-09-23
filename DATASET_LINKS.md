# Dataset Public Links

The package keeps lightweight metadata and result evidence in GitHub. Full raw datasets, archives, cached arrays and model checkpoints are intentionally excluded.

| Source | Public link | Role in Project 5 |
| --- | --- | --- |
| TUM Dataset for Instance Segmentation of Deformable Linear Objects | https://mediatum.ub.tum.de/1690303 | Main real-image evaluation source; S1/S2 were used for training/validation and S3 for held-out testing. |
| TUM DOI | https://doi.org/10.14459/2022mp1690303 | Persistent identifier for the TUM dataset. |
| DeformX/WireSeg-36K | https://huggingface.co/datasets/DeformX/WireSeg-36K | Wire/cable segmentation source; sampled metadata is retained in `dataset/`. |
| DeformX project/code | https://deformx.github.io/ and https://github.com/DeformX/DeformX | Source context for WireSeg. |
| MovingCables dataset | https://zenodo.org/records/14627726 | Inspected as an alternative cable dataset; not used for the final quantitative claims. |
| MovingCables code | https://github.com/holesond/movingcables | Source repository for MovingCables tooling. |
| FASTDLO paper | https://doi.org/10.1109/LRA.2022.3189791 | Published DLO segmentation comparator used in the dissertation. |
| FASTDLO weights | https://mega.nz/file/YNsmnYwa#y9DiZEly-MQ_s8vHifSCDZLghaOe89pd4tZKQ5IOEME | External weights required by the FASTDLO comparator; not committed. |
