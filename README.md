# Hierarchical Moiety Insights Drive Rational Design of High-Efficiency Lithium-Metal Electrolyte Formulations

**Code:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23467028.svg)](https://doi.org/10.5281/zenodo.23467028)

**Inference results:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23464260.svg)](https://doi.org/10.5281/zenodo.23464260)

**Datasets:** [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/JerrySuSuSu/HELENA)

Code and data accompanying **“Hierarchical Moiety Insights Drive Rational Design of High-Efficiency Lithium-Metal Electrolyte Formulations”** (*Nature Energy*, 2026).

HELENA predicts coulombic efficiency using molecular moieties and hierarchical attention.

## Setup and training

Download the datasets from [Hugging Face](https://huggingface.co/datasets/JerrySuSuSu/HELENA) and save them as `data/HELENA-CE-26.csv` (coulombic efficiency) and `data/HELENA-Viscosity-26.csv` (viscosity).

Run from the repository root:

```bash
micromamba create -p ./.venv_helena -f environment.yml
micromamba run -p ./.venv_helena python tasks/training/train_ensemble_regression.py
```

## Inference

Place your input CSV in `data/inference/` and set `DATASET_FILENAME` in `tasks/inference/run_inference.py`, then run:

```bash
micromamba run -p ./.venv_helena python tasks/inference/run_inference.py
```

Model weights are in `data/`. Figure scripts and available source data are in `figures/`.

## Data and code

- Datasets: [HELENA on Hugging Face](https://huggingface.co/datasets/JerrySuSuSu/HELENA)
- Inference results: [10.5281/zenodo.23464260](https://doi.org/10.5281/zenodo.23464260)
- Codebase: [10.5281/zenodo.23467028](https://doi.org/10.5281/zenodo.23467028)
