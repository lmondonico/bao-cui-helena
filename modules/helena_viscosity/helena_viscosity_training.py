"""Training, prediction and metrics for the HELENA viscosity model."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .helena_viscosity_data import MOLECULE_COLUMNS, DeviceTensorBundle, TensorBundle, temperature_statistics
from .helena_viscosity_model import HELENACETemperatureArrhenius, build_model, load_ce_backbone

MODEL_INPUT_KEYS = frozenset(
    {
        "x",
        "adjacency",
        "distance",
        "molecular_features",
        "molar_ratios",
        "molecule_types",
        "temperature_features",
    }
)

STATIC_MODEL_TENSOR_KEYS = MODEL_INPUT_KEYS - {"temperature_features"}

EXCLUDED_MODEL_FIELDS = frozenset(
    {
        "Reference",
        "Row_ID",
        "Electrolyte",
        "Viscosity_mPa_s",
        "Log10_Viscosity",
        "target_log",
        "target_raw",
    }
)
NEW_MODULE_PREFIXES = (
    "arrhenius_intercept.",
    "arrhenius_slope.",
    "arrhenius_residual.",
)


def hash_sequence(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode("utf-8")).hexdigest()


def set_global_determinism(seed: int, config: Mapping[str, Any]) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(config["execution"]["deterministic_algorithms"])
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def tuning_seed(
    master_seed: int, outer_fold: int, inner_fold: int, replica: int
) -> int:
    if replica not in (1, 2):
        raise ValueError("Tuning replica must be 1 or 2")
    return (
        int(master_seed)
        + (20_000 if replica == 1 else 30_000)
        + 100 * int(outer_fold)
        + int(inner_fold)
    )


def round_half_up_median(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("Cannot select an epoch from no values")
    return int(math.floor(float(np.median(np.asarray(values, dtype=np.float64))) + 0.5))


def canonical_signatures(row: Mapping[str, str]) -> tuple[str, str, str]:
    """Return formulation payload, formulation hash, and exact composition+temperature hash."""
    components: dict[tuple[str, str], float] = {}
    for column in MOLECULE_COLUMNS:
        cell = str(row.get(column, "")).strip()
        if not cell:
            continue
        smiles, ratio_text = cell.rsplit(":", 1)
        ratio = float(ratio_text.strip())
        if ratio <= 0:
            raise ValueError(f"Non-positive component ratio in Row_ID={row['Row_ID']}")
        role = "solvent_or_additive" if column.startswith("Solvent_") else "salt"
        key = (role, smiles.strip())
        components[key] = components.get(key, 0.0) + ratio
    total = sum(components.values())
    if total <= 0:
        raise ValueError(f"Empty formulation in Row_ID={row['Row_ID']}")
    normalized = [
        [role, smiles, round(ratio / total, 12)]
        for (role, smiles), ratio in sorted(components.items())
    ]
    formulation_payload = json.dumps(
        {"components": normalized}, sort_keys=True, separators=(",", ":")
    )
    formulation_hash = hashlib.sha256(formulation_payload.encode("utf-8")).hexdigest()
    full_payload = json.dumps(
        {
            "components": normalized,
            "temperature_C": round(float(row["Temperature_C"]), 8),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    full_hash = hashlib.sha256(full_payload.encode("utf-8")).hexdigest()
    return formulation_payload, formulation_hash, full_hash


class FoldLabelAccessor:
    """Read-only labels for one outer-training partition."""

    def __init__(self, fold: int, allowed: np.ndarray, values: np.ndarray, n_rows: int):
        self.fold = fold
        self._allowed = np.zeros(n_rows, dtype=bool)
        self._allowed[allowed] = True
        self._values = np.full(n_rows, np.nan, dtype=np.float32)
        self._values[allowed] = values

    def values(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        index = np.asarray(indices, dtype=np.int64)
        if not np.all(self._allowed[index]):
            raise PermissionError(f"Outer-fold {self.fold} test label access attempted")
        output = self._values[index]
        if not np.isfinite(output).all():
            raise AssertionError("Non-finite training label")
        return output.copy()


class OuterLabelAccessGuard:
    """In-process accidental-access guard; this is not a security boundary."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.float32).copy()
        self._released = False
        self.access_log: list[dict[str, Any]] = []

    def open_training_partition(
        self, fold: int, train: np.ndarray, test: np.ndarray, purpose: str
    ) -> FoldLabelAccessor:
        if self._released:
            raise RuntimeError(
                "Cannot open a training partition after final label release"
            )
        if np.intersect1d(train, test).size:
            raise AssertionError("Outer train/test overlap")
        self.access_log.append(
            {
                "outer_fold": fold,
                "purpose": purpose,
                "accessible_rows": int(len(train)),
                "withheld_rows": int(len(test)),
                "accessible_indices_sha256": hash_sequence(sorted(train.tolist())),
                "withheld_intersection": 0,
            }
        )
        return FoldLabelAccessor(
            fold=fold,
            allowed=train,
            values=self._values[train],
            n_rows=len(self._values),
        )


def strip_targets_from_bundle(bundle: TensorBundle) -> np.ndarray:
    """Move labels into the access guard and erase them from tensors and row dicts."""
    target = bundle.tensors.pop("target_log").numpy().astype(np.float32, copy=True)
    bundle.tensors.pop("target_raw")
    for row in bundle.rows:
        row.pop("Log10_Viscosity", None)
        row.pop("Viscosity_mPa_s", None)
    if set(bundle.tensors) & {"target_log", "target_raw"}:
        raise AssertionError("Targets remained in the model tensor bundle")
    if any({"Log10_Viscosity", "Viscosity_mPa_s"} & set(row) for row in bundle.rows):
        raise AssertionError("Target fields remained in CPU bundle row dictionaries")
    return target


def model_batch(
    bundle: DeviceTensorBundle,
    indices: np.ndarray | Sequence[int],
    temperature_mean: torch.Tensor,
    temperature_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = bundle.tensors["x"].device
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    batch = {
        key: bundle.tensors[key].index_select(0, index)
        for key in STATIC_MODEL_TENSOR_KEYS
    }
    raw_temperature = bundle.tensors["temperature_raw"].index_select(0, index)
    batch["temperature_features"] = (
        raw_temperature - temperature_mean
    ) / temperature_std
    if set(batch) != MODEL_INPUT_KEYS:
        raise AssertionError("Unexpected model-input schema")
    if set(batch) & EXCLUDED_MODEL_FIELDS:
        raise AssertionError("Excluded metadata entered the model batch")
    return batch


def initialize_model(
    vocab_size: int,
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
    head_mode: str,
) -> tuple[HELENACETemperatureArrhenius, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(vocab_size, head_mode=head_mode)
    if (
        not isinstance(model, HELENACETemperatureArrhenius)
        or model.head_mode != head_mode
    ):
        raise AssertionError("builder returned an incompatible model")
    transfer = load_ce_backbone(model, checkpoint_path)
    model.set_output_bias(0.0)
    return model.to(device), transfer


def build_temperature_pairs(
    indices: np.ndarray,
    bundle: TensorBundle,
    formulation_signatures: np.ndarray,
    maximum_per_group: int,
) -> np.ndarray:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    allowed = set(map(int, indices))
    for index in indices:
        groups[
            (str(bundle.references[index]), str(formulation_signatures[index]))
        ].append(int(index))
    pairs = []
    for values in groups.values():
        ordered = sorted(
            values,
            key=lambda i: (
                float(bundle.rows[i]["Temperature_C"]),
                int(bundle.row_ids[i]),
            ),
        )
        candidates = [
            (a, b)
            for a, b in itertools.combinations(ordered, 2)
            if float(bundle.rows[a]["Temperature_C"])
            < float(bundle.rows[b]["Temperature_C"])
        ]
        if len(candidates) > maximum_per_group:
            positions = np.linspace(
                0, len(candidates) - 1, maximum_per_group, dtype=np.int64
            )
            candidates = [candidates[p] for p in positions]
        pairs.extend(candidates)
    if any(a not in allowed or b not in allowed for a, b in pairs):
        raise AssertionError("Temperature pair escaped training partition")
    return (
        np.asarray(sorted(set(pairs)), dtype=np.int64).reshape(-1, 2)
        if pairs
        else np.empty((0, 2), dtype=np.int64)
    )


def _is_new_parameter(name: str) -> bool:
    return any(
        name == prefix.rstrip(".") or name.startswith(prefix)
        for prefix in NEW_MODULE_PREFIXES
    )


def set_backbone_frozen(model: HELENACETemperatureArrhenius, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        if not _is_new_parameter(name):
            parameter.requires_grad_(not frozen)


def optimizer_and_scheduler(
    model: HELENACETemperatureArrhenius,
    config: Mapping[str, Any],
    epochs: int,
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR, dict[str, Any]]:
    opt = config["optimization"]
    grouped = {
        name: []
        for name in ("backbone_decay", "backbone_no_decay", "new_decay", "new_no_decay")
    }
    names = {k: [] for k in grouped}
    for name, p in model.named_parameters():
        family = "new" if _is_new_parameter(name) else "backbone"
        low = name.casefold()
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "norm" in low
        key = f"{family}_{'no_decay' if no_decay else 'decay'}"
        grouped[key].append(p)
        names[key].append(name)
    if any(not v for v in grouped.values()):
        raise AssertionError("Every optimizer group must be nonempty")
    order = list(grouped)
    max_lrs = []
    parameter_groups = []
    for key in order:
        max_lr = float(
            opt["new_branch_max_learning_rate"]
            if key.startswith("new")
            else opt["backbone_max_learning_rate"]
        )
        parameter_groups.append(
            {
                "params": grouped[key],
                "group_name": key,
                "weight_decay": 0.0
                if key.endswith("no_decay")
                else float(opt["weight_decay"]),
            }
        )
        max_lrs.append(max_lr)
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=float(opt["new_branch_max_learning_rate"])
    )
    total = max(1, int(epochs) * int(steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total,
        pct_start=float(opt["one_cycle_pct_start"]),
        anneal_strategy="cos",
        div_factor=float(opt["one_cycle_div_factor"]),
        final_div_factor=float(opt["one_cycle_final_div_factor"]),
        cycle_momentum=False,
    )
    audit = {
        key: {
            "parameter_tensors": len(grouped[key]),
            "parameters": sum(p.numel() for p in grouped[key]),
            "names_sha256": hash_sequence(names[key]),
        }
        for key in order
    }
    audit["total_steps"] = total
    return optimizer, scheduler, audit


def training_model_batch(
    bundle: DeviceTensorBundle,
    indices: np.ndarray,
    temperature_mean: torch.Tensor,
    temperature_std: torch.Tensor,
    noise: float,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    batch = bundle.batch(
        indices,
        temperature_mean,
        temperature_std,
        molar_ratio_noise=noise,
        generator=generator,
        temperature_mode="dual",
    )
    if set(batch) != MODEL_INPUT_KEYS or set(batch) & EXCLUDED_MODEL_FIELDS:
        raise AssertionError("Unexpected training model-input schema")
    return batch


def predict_log10(
    model: HELENACETemperatureArrhenius,
    bundle: DeviceTensorBundle,
    indices: np.ndarray,
    temperature_mean: torch.Tensor,
    temperature_std: torch.Tensor,
    target_mean: float,
    target_std: float,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = model_batch(
                bundle,
                indices[start : start + batch_size],
                temperature_mean,
                temperature_std,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                prediction = model(batch)["log10_viscosity"].squeeze(-1)
            output.append(
                (prediction.float().cpu().numpy() * target_std + target_mean).astype(
                    np.float64
                )
            )
    return np.concatenate(output) if output else np.empty(0, dtype=np.float64)


def _pair_loss(
    model: HELENACETemperatureArrhenius,
    bundle: DeviceTensorBundle,
    pair_batch: np.ndarray,
    labels: FoldLabelAccessor,
    target_std: float,
    tmean: torch.Tensor,
    tstd: torch.Tensor,
    config: Mapping[str, Any],
    generator: torch.Generator | None,
    use_amp: bool,
) -> tuple[torch.Tensor, float, float]:
    if not len(pair_batch):
        return torch.zeros((), device=bundle.tensors["x"].device), 0.0, 0.0
    count = len(pair_batch)
    combined = np.concatenate([pair_batch[:, 0], pair_batch[:, 1]])
    batch = training_model_batch(
        bundle,
        combined,
        tmean,
        tstd,
        0.0,
        generator,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        prediction = model(batch)["log10_viscosity"].squeeze(-1).float()
    predicted = prediction[count:] - prediction[:count]
    observed = torch.as_tensor(
        (labels.values(pair_batch[:, 1]) - labels.values(pair_batch[:, 0]))
        / target_std,
        dtype=torch.float32,
        device=prediction.device,
    )
    difference = F.mse_loss(predicted, observed)
    monotonic = torch.square(
        F.relu(
            predicted
            + float(config["optimization"]["pair_monotonic_margin_standardized"])
        )
    ).mean()
    weighted = (
        float(config["optimization"]["pair_difference_weight"]) * difference
        + float(config["optimization"]["pair_monotonic_weight"]) * monotonic
    )
    return weighted, float(difference.item()), float(monotonic.item())


def train_fit(
    cpu_bundle: TensorBundle,
    device_bundle: DeviceTensorBundle,
    train: np.ndarray,
    labels: FoldLabelAccessor,
    formulation_signatures: np.ndarray,
    head_mode: str,
    seed: int,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    validation: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    if (validation is None) == (fixed_epochs is None):
        raise ValueError("Specify exactly one of validation or fixed_epochs")
    device = device_bundle.tensors["x"].device
    model, transfer = initialize_model(
        cpu_bundle.vocab_size, checkpoint_path, seed, device, head_mode
    )
    train_target = labels.values(train)
    target_mean = float(train_target.mean())
    target_std = max(float(train_target.std(ddof=0)), 1e-6)
    tmean, tstd = temperature_statistics(device_bundle, train)
    opt = config["optimization"]
    max_epochs = int(opt["max_epochs"] if fixed_epochs is None else fixed_epochs)
    steps_per_epoch = max(1, math.ceil(len(train) / int(opt["batch_size"])))
    optimizer, scheduler, grouping = optimizer_and_scheduler(
        model, config, max_epochs, steps_per_epoch
    )
    set_backbone_frozen(model, True)
    pairs = build_temperature_pairs(
        train,
        cpu_bundle,
        formulation_signatures,
        int(opt["maximum_pairs_per_doi_formulation"]),
    )
    rng = np.random.default_rng(seed)
    pair_rng = np.random.default_rng(seed + 1_000_003)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2_000_003)
    use_amp = config["execution"]["precision"] == "bf16" and device.type == "cuda"
    history = []
    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    best_prediction = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        set_backbone_frozen(model, epoch <= int(opt["backbone_freeze_epochs"]))
        model.train()
        primary_values = []
        difference_values = []
        monotonic_values = []
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled), int(opt["batch_size"])):
            indices = shuffled[start : start + int(opt["batch_size"])]
            size = min(int(opt["pair_batch_size"]), len(pairs))
            pair_batch = (
                pairs[pair_rng.choice(len(pairs), size=size, replace=False)]
                if size
                else pairs
            )
            batch = training_model_batch(
                device_bundle,
                indices,
                tmean,
                tstd,
                float(opt["molar_ratio_noise"]),
                generator,
            )
            target = torch.as_tensor(
                (labels.values(indices) - target_mean) / target_std,
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                primary = F.mse_loss(
                    model(batch)["log10_viscosity"].squeeze(-1).float(), target
                )
            auxiliary, difference, monotonic = _pair_loss(
                model,
                device_bundle,
                pair_batch,
                labels,
                target_std,
                tmean,
                tstd,
                config,
                generator,
                use_amp,
            )
            loss = primary + auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(opt["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            primary_values.append(float(primary.item()))
            difference_values.append(difference)
            monotonic_values.append(monotonic)
        row = {
            "epoch": epoch,
            "train_standardized_mse": float(np.mean(primary_values)),
            "pair_difference_mse": float(np.mean(difference_values)),
            "pair_monotonic_mse": float(np.mean(monotonic_values)),
            "backbone_frozen": epoch <= int(opt["backbone_freeze_epochs"]),
            "optimizer_steps": epoch * steps_per_epoch,
        }
        if validation is not None:
            prediction = predict_log10(
                model,
                device_bundle,
                validation,
                tmean,
                tstd,
                target_mean,
                target_std,
                int(opt["evaluation_batch_size"]),
                use_amp,
            )
            validation_mse = float(
                np.mean(np.square(prediction - labels.values(validation)))
            )
            row["validation_mse_log10"] = validation_mse
            if validation_mse < best_mse - float(
                opt["early_stopping_min_delta_validation_mse"]
            ):
                best_mse = validation_mse
                best_epoch = epoch
                stale = 0
                best_prediction = prediction.copy()
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= int(opt["early_stopping_patience"]):
            break
    if validation is not None:
        if best_state is None or best_prediction is None:
            raise RuntimeError("Inner fit produced no best state")
        model.load_state_dict(best_state)
        prediction = best_prediction
        trained_epochs = len(history)
    else:
        best_epoch = max_epochs
        trained_epochs = max_epochs
        prediction = np.empty(0, dtype=np.float64)
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    checkpoint = {
        "vocab_size": cpu_bundle.vocab_size,
        "head_mode": head_mode,
        "model_state_dict": best_state,
        "preprocessing": {
            "temperature_mean": tmean.detach().cpu(),
            "temperature_std": tstd.detach().cpu(),
            "target_mean_log10": target_mean,
            "target_std_log10": target_std,
        },
        "trained_epochs": trained_epochs,
        "selected_epoch": best_epoch,
    }
    record = {
        "seed": seed,
        "head_mode": head_mode,
        "train_rows": len(train),
        "validation_rows": 0 if validation is None else len(validation),
        "temperature_pairs": len(pairs),
        "best_epoch": best_epoch,
        "best_validation_mse_log10": None if validation is None else best_mse,
        "optimizer_grouping": grouping,
        "transfer_report": transfer,
    }
    return checkpoint, history, prediction, record


def regression_metrics(
    target_log: np.ndarray,
    prediction_log: np.ndarray,
    raw_prediction_clip_log10: Sequence[float],
) -> dict[str, float]:
    target_log = np.asarray(target_log, dtype=np.float64)
    prediction_log = np.asarray(prediction_log, dtype=np.float64)
    target_raw = np.power(10.0, target_log)
    clip_low, clip_high = map(float, raw_prediction_clip_log10)
    prediction_raw = np.power(10.0, np.clip(prediction_log, clip_low, clip_high))
    absolute_log_error = np.abs(target_log - prediction_log)
    return {
        "r2_log10": float(r2_score(target_log, prediction_log)),
        "rmse_log10": float(np.sqrt(mean_squared_error(target_log, prediction_log))),
        "mae_log10": float(mean_absolute_error(target_log, prediction_log)),
        "pearson_log10": float(np.corrcoef(target_log, prediction_log)[0, 1]),
        "spearman_log10": float(spearmanr(target_log, prediction_log).statistic),
        "geometric_mean_fold_error": float(np.power(10.0, absolute_log_error.mean())),
        "median_fold_error": float(np.power(10.0, np.median(absolute_log_error))),
        "within_factor_2": float(np.mean(absolute_log_error <= math.log10(2.0))),
        "within_factor_3": float(np.mean(absolute_log_error <= math.log10(3.0))),
        "r2_raw_mPa_s": float(r2_score(target_raw, prediction_raw)),
        "rmse_raw_mPa_s": float(
            np.sqrt(mean_squared_error(target_raw, prediction_raw))
        ),
        "mae_raw_mPa_s": float(mean_absolute_error(target_raw, prediction_raw)),
    }


def cluster_bootstrap(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
    raw_prediction_clip_log10: Sequence[float],
) -> dict[str, dict[str, float]]:
    unique_groups = np.unique(groups)
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    tracked = ("r2_log10", "rmse_log10", "mae_log10")
    values: dict[str, list[float]] = {metric: [] for metric in tracked}
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        metrics = regression_metrics(
            target[indices], prediction[indices], raw_prediction_clip_log10
        )
        for metric in tracked:
            if math.isfinite(metrics[metric]):
                values[metric].append(metrics[metric])
    return {
        metric: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
        }
        for metric, metric_values in values.items()
    }


def load_folds(
    dataset_rows: Sequence[Mapping[str, str]], folds_path: Path
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[int, list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Outer (train, test) and inner (train, validation) row-index splits from viscosity_cv_folds.csv.

    Groups are exact normalized composition plus temperature: no group may be
    split across an outer or an inner fold. Also returns the formulation hash of
    every row, which defines the training-only temperature pairs.
    """
    with folds_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["Row_ID"]) for row in rows] != [int(row["Row_ID"]) for row in dataset_rows]:
        raise AssertionError("viscosity_cv_folds.csv rows must match the dataset rows in order")
    signatures = [canonical_signatures(row) for row in dataset_rows]
    formulation = np.asarray([value[1] for value in signatures], dtype=object)
    group = np.asarray([value[2] for value in signatures], dtype=object)
    outer = np.asarray([int(row["Outer_Fold"]) for row in rows], dtype=np.int64)
    if set(outer.tolist()) != set(range(1, 6)):
        raise AssertionError("Outer folds must be numbered 1 through 5")
    outer_splits = []
    inner_splits = {}
    for fold in range(1, 6):
        test = np.flatnonzero(outer == fold)
        train = np.flatnonzero(outer != fold)
        if set(group[train]) & set(group[test]):
            raise AssertionError(f"Composition+temperature group crosses outer fold {fold}")
        inner = np.asarray([int(row[f"Inner_Fold_{fold}"]) for row in rows], dtype=np.int64)
        if np.any(inner[test] != 0) or set(inner[train].tolist()) != set(range(1, 6)):
            raise AssertionError(f"Invalid inner folds for outer fold {fold}")
        splits = []
        for inner_fold in range(1, 6):
            validation = np.flatnonzero(inner == inner_fold)
            inner_train = np.flatnonzero((inner > 0) & (inner != inner_fold))
            if set(group[inner_train]) & set(group[validation]):
                raise AssertionError(f"Group crosses inner fold {inner_fold} of outer fold {fold}")
            splits.append((inner_train, validation))
        outer_splits.append((train, test))
        inner_splits[fold] = splits
    return outer_splits, inner_splits, formulation
