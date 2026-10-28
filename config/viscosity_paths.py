"""Paths for viscosity inputs, configuration, and study outputs."""

from pathlib import Path

from config.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

DATASET_PATH = Path(DATA_DIR) / "HELENA-Viscosity-26.csv"
FEATURES_DIR = Path(DATA_DIR) / "viscosity"
FOLDS_PATH = FEATURES_DIR / "viscosity_cv_folds.csv"
MOLECULE_INDEX_PATH = FEATURES_DIR / "viscosity_molecule_index.csv"
MOLECULE_FEATURES_PATH = FEATURES_DIR / "viscosity_molecule_features.npz"
TOKENIZER_MAPPING_PATH = FEATURES_DIR / "viscosity_tokenizer_mapping.json"
CONFIG_PATH = Path(PROJECT_ROOT) / "config" / "viscosity_parameters.json"
REFERENCE_RESULTS_DIR = Path(PROJECT_ROOT) / "figures" / "SI_Fig_Viscosity" / "data"
PUBLICATIONS_PATH = REFERENCE_RESULTS_DIR / "publications.csv"
RUNS_DIR = Path(RESULTS_DIR) / "viscosity" / "runs"
EVALUATION_DIR = Path(RESULTS_DIR) / "viscosity" / "evaluation"
FIGURES_OUTPUT_DIR = Path(RESULTS_DIR) / "viscosity" / "figures"
