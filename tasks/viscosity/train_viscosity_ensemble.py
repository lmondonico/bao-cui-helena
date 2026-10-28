"""Train five-member viscosity ensembles with nested five-fold cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.viscosity_paths import (  # noqa: E402
    CONFIG_PATH, DATASET_PATH, FOLDS_PATH, MOLECULE_FEATURES_PATH,
    MOLECULE_INDEX_PATH, RUNS_DIR, TOKENIZER_MAPPING_PATH,
)
from modules.helena_viscosity.helena_viscosity_data import load_tensor_bundle  # noqa: E402
from modules.helena_viscosity.helena_viscosity_model import CE_CHECKPOINT, build_model  # noqa: E402
from modules.helena_viscosity.helena_viscosity_training import (  # noqa: E402
    OuterLabelAccessGuard,
    load_folds,
    predict_log10,
    round_half_up_median,
    set_global_determinism,
    strip_targets_from_bundle,
    train_fit,
    tuning_seed,
)

def ensure_environment(config: dict) -> None:
    """Re-launch with the hash seed and cuBLAS workspace the results were produced with."""
    required = {
        "PYTHONHASHSEED": str(config["execution"]["python_hash_seed"]),
        "CUBLAS_WORKSPACE_CONFIG": str(config["execution"]["cublas_workspace_config"]),
    }
    if all(os.environ.get(key) == value for key, value in required.items()):
        return
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **required})


class Context:
    def __init__(self, config: dict, max_epochs: int | None):
        self.config = json.loads(json.dumps(config))
        if max_epochs is not None:
            self.config["optimization"]["max_epochs"] = int(max_epochs)
        self.master = int(config["master_seed"])
        self.head_mode = str(config["head_mode"])
        set_global_determinism(self.master, config)
        torch.set_num_threads(int(config["execution"]["torch_threads"]))
        self.device = torch.device(config["execution"]["training_device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Training requires a CUDA GPU")
        mapping = json.loads(TOKENIZER_MAPPING_PATH.read_text(encoding="utf-8"))
        self.cpu = load_tensor_bundle(
            DATASET_PATH,
            MOLECULE_INDEX_PATH,
            MOLECULE_FEATURES_PATH,
            int(mapping["final_vocab_size"]),
        )
        self.outer, self.inner, self.formulation = load_folds(self.cpu.rows, FOLDS_PATH)
        guard = OuterLabelAccessGuard(strip_targets_from_bundle(self.cpu))
        self.labels = {
            fold: guard.open_training_partition(fold, train, test, "inner fits and final refit")
            for fold, (train, test) in enumerate(self.outer, start=1)
        }
        self.device_bundle = self.cpu.to(self.device)


def inner_path(runs: Path, outer: int, inner: int) -> Path:
    return runs / "inner" / f"o{outer}_i{inner}.json"


def final_path(runs: Path, outer: int, member: int) -> Path:
    return runs / "final" / f"o{outer}_m{member}.pt"


def write_atomic(path: Path, save) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save(temporary)
    temporary.replace(path)


def run_inner(ctx: Context, runs: Path, outer: int, inner: int) -> None:
    path = inner_path(runs, outer, inner)
    if path.exists():
        return
    train, validation = ctx.inner[outer][inner - 1]
    seed = tuning_seed(ctx.master, outer, inner, 1)
    print(f"inner fit outer {outer} inner {inner} (seed {seed})", flush=True)
    started = time.perf_counter()
    _, history, prediction, record = train_fit(
        ctx.cpu, ctx.device_bundle, train, ctx.labels[outer], ctx.formulation,
        ctx.head_mode, seed, CE_CHECKPOINT, ctx.config, validation=validation,
    )
    payload = {
        "outer_fold": outer,
        "inner_fold": inner,
        "seed": seed,
        "best_epoch": int(record["best_epoch"]),
        "trained_epochs": len(history),
        "best_validation_mse_log10": record["best_validation_mse_log10"],
        "validation_row_ids": ctx.cpu.row_ids[validation].tolist(),
        "validation_prediction_log10": prediction.tolist(),
        "history": history,
        "minutes": (time.perf_counter() - started) / 60,
    }
    write_atomic(path, lambda target: target.write_text(json.dumps(payload, indent=1) + "\n"))
    print(f"  best epoch {payload['best_epoch']} of {payload['trained_epochs']}", flush=True)


def decision(ctx: Context, runs: Path, outer: int) -> dict:
    """Refit epochs for an outer fold from its five inner fits (and their pooled inner RMSE)."""
    fits = []
    for inner in range(1, 6):
        path = inner_path(runs, outer, inner)
        if not path.exists():
            raise FileNotFoundError(f"Run the inner fits of outer fold {outer} first: {path}")
        fits.append(json.loads(path.read_text()))
    max_epochs = int(ctx.config["optimization"]["max_epochs"])
    epochs = [int(fit["best_epoch"]) for fit in fits]
    train = ctx.outer[outer - 1][0]
    position = {int(row_id): index for index, row_id in enumerate(ctx.cpu.row_ids)}
    indices = np.asarray([position[int(r)] for fit in fits for r in fit["validation_row_ids"]], dtype=np.int64)
    if sorted(indices.tolist()) != sorted(train.tolist()):
        raise AssertionError("Inner validation folds do not cover the outer-training rows once")
    prediction = np.asarray([p for fit in fits for p in fit["validation_prediction_log10"]], dtype=np.float64)
    order = np.argsort(indices)
    target = ctx.labels[outer].values(indices[order])
    rmse = float(np.sqrt(np.mean(np.square(prediction[order] - target))))
    return {
        "outer_fold": outer,
        "selected_epoch": max(1, min(max_epochs, round_half_up_median(epochs))),
        "inner_best_epochs": epochs,
        "pooled_inner_rmse_log10": rmse,
    }


def run_final(ctx: Context, runs: Path, outer: int, member: int) -> None:
    path = final_path(runs, outer, member)
    if path.exists():
        return
    chosen = decision(ctx, runs, outer)
    train, test = ctx.outer[outer - 1]
    seed = ctx.master + 100 * outer + member
    epochs = int(chosen["selected_epoch"])
    print(f"final fit outer {outer} member {member} (seed {seed}, {epochs} epochs)", flush=True)
    started = time.perf_counter()
    checkpoint, history, _, _ = train_fit(
        ctx.cpu, ctx.device_bundle, train, ctx.labels[outer], ctx.formulation,
        ctx.head_mode, seed, CE_CHECKPOINT, ctx.config, fixed_epochs=epochs,
    )
    model = build_model(ctx.cpu.vocab_size, head_mode=ctx.head_mode)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(ctx.device)
    prep = checkpoint["preprocessing"]
    prediction = predict_log10(
        model, ctx.device_bundle, test,
        prep["temperature_mean"].to(ctx.device), prep["temperature_std"].to(ctx.device),
        float(prep["target_mean_log10"]), float(prep["target_std_log10"]),
        int(ctx.config["optimization"]["evaluation_batch_size"]),
        ctx.device.type == "cuda" and ctx.config["execution"]["precision"] == "bf16",
    )
    payload = {
        "outer_fold": outer,
        "member": member,
        "seed": seed,
        "decision": chosen,
        "model_state_dict": checkpoint["model_state_dict"],
        "preprocessing": prep,
        "test_row_ids": torch.as_tensor(ctx.cpu.row_ids[test]),
        "test_prediction_log10": torch.as_tensor(prediction, dtype=torch.float64),
        "history": history,
        "minutes": (time.perf_counter() - started) / 60,
    }
    write_atomic(path, lambda target: torch.save(payload, target))
    del model, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run every inner and final fit")
    parser.add_argument("--outer-fold", type=int, choices=range(1, 6))
    parser.add_argument("--inner-fold", type=int, choices=range(1, 6))
    parser.add_argument("--member", type=int, choices=range(1, 6))
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max-epochs", type=int, help="quick test only: cap the inner-fit epochs")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ensure_environment(config)
    if args.all == (args.outer_fold is not None) or (
        args.outer_fold is not None and (args.inner_fold is None) == (args.member is None)
    ):
        parser.error("use --all, or --outer-fold with exactly one of --inner-fold / --member")
    ctx = Context(config, args.max_epochs)
    runs = args.runs.resolve()
    if args.all:
        for outer in range(1, 6):
            for inner in range(1, 6):
                run_inner(ctx, runs, outer, inner)
            for member in range(1, 6):
                run_final(ctx, runs, outer, member)
    elif args.inner_fold is not None:
        run_inner(ctx, runs, args.outer_fold, args.inner_fold)
    else:
        run_final(ctx, runs, args.outer_fold, args.member)


if __name__ == "__main__":
    main()
