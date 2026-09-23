# Project 5 Professor Code And Results Package

This is the GitHub-friendly Project 5 evidence package. It keeps the files needed to inspect the dissertation evidence while excluding raw datasets, large checkpoints, cached arrays and local build noise.

## Folder Structure

- `dataset/`: dataset source metadata and lightweight WireSeg sample files.
- `scripts/`: Python scripts used for dataset inspection, model evaluation, training/adaptation, result summaries and QA.
- `outputs/`: final dissertation DOCX, JSON result files, QA evidence, dissertation figures and experiment figures.

## Main Review Files

- `outputs/final-dissertation-5-cmp.docx`: latest final dissertation document copied from the Project 5 output folder.
- `outputs/results/`: measured experiment JSON outputs used in Chapter 4.
- `outputs/figures/`, `outputs/dissertation-figures/` and `outputs/experiment-figures/`: figures and qualitative result outputs.
- `outputs/qa/`: lightweight QA and benchmark evidence.
- `DATASET_LINKS.md`: public links for datasets and external model resources.
- `LARGE_FILES_EXCLUDED.md`: list of large local files deliberately excluded from GitHub.

## Reproducibility Boundary

The package is designed for GitHub submission/review, not as a full raw-data mirror. To rerun every experiment, restore the public datasets under a local `data/` folder and restore the external FASTDLO source/weights under `code/fastdlo_src/fastdlo-master/`. Those local folders are ignored by `.gitignore`.

The retained JSON result files and figures are the lightweight evidence files that support the final dissertation without committing large binaries.
