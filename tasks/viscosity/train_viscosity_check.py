"""Check viscosity data, folds, and saved ensemble results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.viscosity_paths import (  # noqa: E402
    CONFIG_PATH, DATASET_PATH, FOLDS_PATH, MOLECULE_FEATURES_PATH,
    MOLECULE_INDEX_PATH, PUBLICATIONS_PATH, REFERENCE_RESULTS_DIR,
    TOKENIZER_MAPPING_PATH,
)
from modules.helena_viscosity.helena_viscosity_data import MOLECULE_COLUMNS, parse_cell, read_csv  # noqa: E402
from modules.helena_viscosity.helena_viscosity_training import load_folds, regression_metrics, round_half_up_median  # noqa: E402


def check_features(rows: list[dict[str, str]]) -> int:
    molecules = read_csv(MOLECULE_INDEX_PATH)
    mapping = json.loads(TOKENIZER_MAPPING_PATH.read_text())
    indices = [int(row["Molecule_Index"]) for row in molecules]
    np.testing.assert_array_equal(indices, np.arange(len(molecules)))
    occurrences: Counter[str] = Counter()
    for row in rows:
        ratios = []
        for column in MOLECULE_COLUMNS:
            if row[column].strip():
                smiles, ratio = parse_cell(row[column])
                if not np.isfinite(ratio) or ratio <= 0:
                    raise AssertionError(f"Invalid component ratio in row {row['Row_ID']}")
                ratios.append(ratio)
                occurrences[smiles] += 1
        np.testing.assert_allclose(sum(ratios), 1.0, rtol=0, atol=1e-8)
    if set(occurrences) != {row["SMILES"] for row in molecules}:
        raise AssertionError("Molecule features must cover the dataset exactly")
    with np.load(MOLECULE_FEATURES_PATH, allow_pickle=False) as features:
        tokens = features["token_ids"]
        if tokens.shape[0] != len(molecules) or tokens.max() >= mapping["final_vocab_size"]:
            raise AssertionError("Molecular tokens do not match the saved vocabulary")
        expected_graph_shape = (len(molecules), tokens.shape[1], tokens.shape[1])
        for name in ("adjacency", "distance"):
            if features[name].shape != expected_graph_shape or not np.isfinite(features[name]).all():
                raise AssertionError(f"Invalid molecular {name} matrices")
        if features["rdkit_features"].shape[0] != len(molecules) or not np.isfinite(features["rdkit_features"]).all():
            raise AssertionError("Invalid saved RDKit features")
        for row in molecules:
            index = int(row["Molecule_Index"])
            saved = json.loads(row["Moiety_Token_IDs_JSON"])
            np.testing.assert_array_equal(tokens[index][tokens[index] != 0], saved)
            if len(saved) != int(row["Token_Count"]) or len(saved) != features["lengths"][index]:
                raise AssertionError(f"Token count mismatch for {row['SMILES']}")
            if occurrences[row["SMILES"]] != int(row["Rows_Using_Molecule"]):
                raise AssertionError(f"Molecule occurrence mismatch for {row['SMILES']}")
    publications = read_csv(PUBLICATIONS_PATH)
    dois = [row["DOI"].strip().lower() for row in publications]
    if len(set(dois)) != len(dois) or set(dois) != {row["Reference"].strip().lower() for row in rows}:
        raise AssertionError("Publication metadata must cover each DOI exactly once")
    return len(molecules)


def check_results(rows: list[dict[str, str]], outer_splits, config: dict) -> dict:
    predictions = pd.read_csv(REFERENCE_RESULTS_DIR / "predictions.csv", float_precision="round_trip")
    metrics = json.loads((REFERENCE_RESULTS_DIR / "metrics.json").read_text())
    row_ids = [int(row["Row_ID"]) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise AssertionError("Dataset Row_ID values must be unique")
    np.testing.assert_array_equal(predictions["Row_ID"], row_ids)
    np.testing.assert_array_equal(predictions["Reference"], [row["Reference"] for row in rows])
    target = np.asarray([float(row["Log10_Viscosity"]) for row in rows], dtype=np.float32).astype(float)
    np.testing.assert_array_equal(predictions["Actual_Log10_Viscosity"], target)
    member_columns = [f"Member_{member}_Predicted_Log10_Viscosity" for member in range(1, 6)]
    members = predictions[member_columns].to_numpy(float)
    if not np.isfinite(members).all():
        raise AssertionError("Member predictions must be finite")
    mean = predictions["Predicted_Log10_Viscosity"].to_numpy(float)
    np.testing.assert_allclose(mean, members.mean(axis=1), rtol=1e-12, atol=1e-12)
    if metrics["rows"] != len(rows) or metrics["unique_references"] != predictions["Reference"].nunique():
        raise AssertionError("Reported dataset counts disagree with the data")
    clip = config["evaluation"]["prediction_clip_log10_for_raw_metrics"]

    def compare(actual: dict, expected) -> None:
        for key, value in actual.items():
            np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-12, err_msg=key)

    compare(regression_metrics(target, mean, clip), metrics["metrics"])
    fold_metrics = pd.read_csv(REFERENCE_RESULTS_DIR / "fold_metrics.csv").set_index("Outer_Fold")
    seed_metrics = pd.read_csv(REFERENCE_RESULTS_DIR / "seed_metrics.csv").set_index("Seed_Index")
    decisions = json.loads((REFERENCE_RESULTS_DIR / "fold_decisions.json").read_text())
    for fold, (_, test) in enumerate(outer_splits, start=1):
        np.testing.assert_array_equal(predictions.iloc[test]["Outer_Fold"], np.full(len(test), fold))
        compare(regression_metrics(target[test], mean[test], clip), fold_metrics.loc[fold])
        decision = decisions[str(fold)]
        if decision["selected_epoch"] != round_half_up_median(decision["inner_best_epochs"]):
            raise AssertionError(f"Refit epoch mismatch in outer fold {fold}")
        expected_seeds = [config["master_seed"] + 100 * fold + member for member in range(1, 6)]
        if decision["member_seeds"] != expected_seeds:
            raise AssertionError(f"Seed schedule mismatch in outer fold {fold}")
    for index in range(5):
        compare(regression_metrics(target, members[:, index], clip), seed_metrics.loc[index + 1])
    return metrics


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    rows = read_csv(DATASET_PATH)
    values = pd.DataFrame(rows)[["Temperature_C", "Temperature_K", "Viscosity_mPa_s", "Log10_Viscosity"]].astype(float)
    if not np.isfinite(values).all().all() or (values[["Temperature_K", "Viscosity_mPa_s"]] <= 0).any().any():
        raise AssertionError("Temperatures and viscosities must be finite and physically valid")
    np.testing.assert_allclose(values["Temperature_K"], values["Temperature_C"] + 273.15, rtol=0, atol=1e-9)
    np.testing.assert_allclose(values["Log10_Viscosity"], np.log10(values["Viscosity_mPa_s"]), rtol=0, atol=1e-12)
    molecules = check_features(rows)
    outer, _, _ = load_folds(rows, FOLDS_PATH)
    metrics = check_results(rows, outer, config)
    print(f"Verified {len(rows):,} rows, {molecules} molecules, {metrics['unique_references']} DOIs.")
    print("Verified outer/inner fold separation, saved predictions, pooled/fold/member metrics and refit seeds.")
    print(f"Saved predictions: R2={metrics['metrics']['r2_log10']:.4f}; RMSE={metrics['metrics']['rmse_log10']:.4f} log10.")
    print("No training, figure generation, or bootstrap rerun performed.")


if __name__ == "__main__":
    main()
