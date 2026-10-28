"""Evaluate held-out viscosity ensembles from the 25 final checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.viscosity_paths import CONFIG_PATH, DATASET_PATH, EVALUATION_DIR, FOLDS_PATH, RUNS_DIR  # noqa: E402
from modules.helena_viscosity.helena_viscosity_data import read_csv  # noqa: E402
from modules.helena_viscosity.helena_viscosity_training import cluster_bootstrap, regression_metrics  # noqa: E402

FOLDS = range(1, 6)
MEMBERS = range(1, 6)


def metric_summary(frame: pd.DataFrame, metrics: tuple[str, ...]) -> dict:
    summary = {}
    for metric in metrics:
        values = frame[metric].to_numpy(float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n": int(len(values)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--results", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    if args.results.exists():
        parser.error("Choose a new --results directory; existing results are never overwritten.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_csv(DATASET_PATH)
    row_ids = np.asarray([int(row["Row_ID"]) for row in rows], dtype=np.int64)
    references = np.asarray([row["Reference"] for row in rows], dtype=str)
    position = {int(row_id): index for index, row_id in enumerate(row_ids)}
    target = np.asarray([float(row["Log10_Viscosity"]) for row in rows], dtype=np.float32).astype(np.float64)
    folds = read_csv(FOLDS_PATH)
    if [int(row["Row_ID"]) for row in folds] != row_ids.tolist():
        raise AssertionError("viscosity_cv_folds.csv rows must match the dataset rows in order")
    expected_outer = np.asarray([int(row["Outer_Fold"]) for row in folds])

    member_matrix = np.full((len(row_ids), len(MEMBERS)), np.nan, dtype=np.float64)
    outer_by_index = np.zeros(len(row_ids), dtype=np.int64)
    decisions = {}
    for outer in FOLDS:
        for column, member in enumerate(MEMBERS):
            fit = torch.load(args.runs / "final" / f"o{outer}_m{member}.pt", map_location="cpu", weights_only=True)
            test = np.asarray([position[int(r)] for r in fit["test_row_ids"].tolist()], dtype=np.int64)
            if not np.array_equal(np.sort(test), np.flatnonzero(expected_outer == outer)):
                raise AssertionError(f"Outer fold {outer}, member {member} must predict its held-out rows exactly once")
            if int(fit["outer_fold"]) != outer or int(fit["member"]) != member:
                raise AssertionError("Final fit identity does not match its filename")
            member_matrix[test, column] = fit["test_prediction_log10"].numpy()
            outer_by_index[test] = outer
            if column == 0:
                decisions[outer] = {**fit["decision"], "member_seeds": []}
            elif fit["decision"]["selected_epoch"] != decisions[outer]["selected_epoch"]:
                raise AssertionError(f"Members of outer fold {outer} disagree on the refit epochs")
            decisions[outer]["member_seeds"].append(int(fit["seed"]))
    if not np.isfinite(member_matrix).all() or not np.all(outer_by_index > 0):
        raise AssertionError("Final fits do not predict every row exactly once")

    mean_prediction = member_matrix.mean(axis=1)
    clip = config["evaluation"]["prediction_clip_log10_for_raw_metrics"]
    predictions = pd.DataFrame(
        {
            "Row_ID": row_ids,
            "Reference": references,
            "Outer_Fold": outer_by_index,
            "Actual_Log10_Viscosity": target,
            "Predicted_Log10_Viscosity": mean_prediction,
            **{f"Member_{m}_Predicted_Log10_Viscosity": member_matrix[:, c] for c, m in enumerate(MEMBERS)},
        }
    )
    y = predictions["Actual_Log10_Viscosity"].to_numpy(float)
    p = predictions["Predicted_Log10_Viscosity"].to_numpy(float)
    master = int(config["master_seed"])
    seed_metrics = pd.DataFrame(
        [
            {
                "Seed_Index": member,
                "Seed_Schedule_JSON": json.dumps({str(o): master + 100 * o + member for o in FOLDS}, sort_keys=True),
                **regression_metrics(y, predictions[f"Member_{member}_Predicted_Log10_Viscosity"].to_numpy(float), clip),
            }
            for member in MEMBERS
        ]
    )
    fold_metrics = pd.DataFrame(
        [{"Outer_Fold": o, **regression_metrics(y[outer_by_index == o], p[outer_by_index == o], clip)} for o in FOLDS]
    )
    evaluation = config["evaluation"]
    metrics = {
        "rows": int(len(predictions)),
        "unique_references": int(predictions["Reference"].nunique()),
        "metrics": regression_metrics(y, p, clip),
        "seed_replicate_summary": metric_summary(seed_metrics, ("r2_log10", "rmse_log10", "mae_log10")),
        "outer_fold_summary": metric_summary(fold_metrics, ("r2_log10", "rmse_log10", "mae_log10")),
        "cluster_bootstrap_95": cluster_bootstrap(
            y, p, predictions["Reference"].astype(str).to_numpy(),
            int(evaluation["bootstrap_replicates"]), int(evaluation["bootstrap_seed"]), clip,
        ),
        "doi_cluster_bootstrap_replicates": int(evaluation["bootstrap_replicates"]),
        "doi_cluster_bootstrap_seed": int(evaluation["bootstrap_seed"]),
        "evaluation_design": {
            "outer_folds": 5,
            "outer_split": "groups of identical normalized composition and temperature never cross folds",
            "inner_selection": "5-fold inner CV per outer fold selects the refit epochs",
            "members_per_outer_fold": len(MEMBERS),
            "prediction": "row-wise arithmetic mean of its held-out fold's member predictions in log10 space",
            "seed_replicate": "each member index concatenated across the five held-out folds",
            "bootstrap": "resamples DOIs (Reference) with replacement",
        },
    }

    args.results.mkdir(parents=True)
    predictions.to_csv(args.results / "predictions.csv", index=False, float_format="%.17g")
    seed_metrics.to_csv(args.results / "seed_metrics.csv", index=False, float_format="%.17g")
    fold_metrics.to_csv(args.results / "fold_metrics.csv", index=False, float_format="%.17g")
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (args.results / "fold_decisions.json").write_text(
        json.dumps({str(o): decisions[o] for o in FOLDS}, indent=2) + "\n"
    )
    pooled = metrics["metrics"]
    print(f"R2 {pooled['r2_log10']:.4f}  RMSE {pooled['rmse_log10']:.4f}  MAE {pooled['mae_log10']:.4f} (log10)")


if __name__ == "__main__":
    main()
