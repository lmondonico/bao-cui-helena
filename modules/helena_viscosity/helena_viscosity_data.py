"""Static tensor assembly for HELENA viscosity training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SOLVENT_COLUMNS = [f"Solvent_{i}" for i in range(1, 6)]
SALT_COLUMNS = [f"Salt_{i}" for i in range(1, 4)]
MOLECULE_COLUMNS = SOLVENT_COLUMNS + SALT_COLUMNS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_cell(cell: str) -> tuple[str, float]:
    smiles, ratio = cell.rsplit(":", 1)
    return smiles.strip(), float(ratio)


@dataclass
class DeviceTensorBundle:
    tensors: dict[str, torch.Tensor]
    rows: list[dict[str, str]]
    references: np.ndarray
    row_ids: np.ndarray
    electrolytes: np.ndarray
    vocab_size: int

    def batch(
        self,
        indices: np.ndarray | torch.Tensor,
        temperature_mean: torch.Tensor,
        temperature_std: torch.Tensor,
        molar_ratio_noise: float = 0.0,
        generator: torch.Generator | None = None,
        temperature_mode: str = "dual",
    ) -> dict[str, torch.Tensor]:
        device = self.tensors["x"].device
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        batch = {
            key: value.index_select(0, index)
            for key, value in self.tensors.items()
            if key not in {"temperature_raw"}
        }
        temperature_raw = self.tensors["temperature_raw"].index_select(0, index)
        if temperature_mode == "dual":
            batch["temperature_features"] = (
                temperature_raw - temperature_mean
            ) / temperature_std
        elif temperature_mode == "none":
            batch["temperature_features"] = torch.zeros_like(temperature_raw)
        else:
            raise ValueError(f"Unsupported temperature mode: {temperature_mode}")
        if molar_ratio_noise > 0:
            ratios = batch["molar_ratios"]
            noise = torch.rand(
                ratios.shape,
                dtype=ratios.dtype,
                device=device,
                generator=generator,
            )
            multiplier = 1.0 + (2.0 * noise - 1.0) * molar_ratio_noise
            batch["molar_ratios"] = torch.where(ratios > 0, ratios * multiplier, ratios)
        return batch


@dataclass
class TensorBundle:
    tensors: dict[str, torch.Tensor]
    rows: list[dict[str, str]]
    references: np.ndarray
    row_ids: np.ndarray
    electrolytes: np.ndarray
    vocab_size: int

    def to(self, device: torch.device) -> DeviceTensorBundle:
        return DeviceTensorBundle(
            tensors={key: value.to(device) for key, value in self.tensors.items()},
            rows=self.rows,
            references=self.references,
            row_ids=self.row_ids,
            electrolytes=self.electrolytes,
            vocab_size=self.vocab_size,
        )


def load_tensor_bundle(
    dataset_path: Path,
    molecule_index_path: Path,
    molecule_features_path: Path,
    vocab_size: int,
) -> TensorBundle:
    rows = read_csv(dataset_path)
    molecule_rows = read_csv(molecule_index_path)
    molecule_to_index = {
        row["SMILES"]: int(row["Molecule_Index"]) for row in molecule_rows
    }
    features = np.load(molecule_features_path)
    molecule_tokens = features["token_ids"]
    molecule_adjacency = features["adjacency"]
    molecule_distance = features["distance"]
    if "rdkit_features" not in features:
        raise ValueError("viscosity_molecule_features.npz lacks audited rdkit_features")
    molecule_rdkit_features = features["rdkit_features"]
    n_rows = len(rows)
    n_slots = len(MOLECULE_COLUMNS)
    max_tokens = molecule_tokens.shape[1]

    x = np.zeros((n_rows, n_slots, max_tokens), dtype=np.int64)
    adjacency = np.full(
        (n_rows, n_slots, max_tokens, max_tokens), -1e9, dtype=np.float32
    )
    distance = np.full_like(adjacency, -1e9)
    molar_ratios = np.zeros((n_rows, n_slots), dtype=np.float32)
    molecule_types = np.zeros((n_rows, n_slots), dtype=np.int64)
    molecular_features = np.zeros(
        (n_rows, n_slots, molecule_rdkit_features.shape[1]), dtype=np.float32
    )

    for row_index, row in enumerate(rows):
        slot_index = 0
        for column in MOLECULE_COLUMNS:
            cell = row.get(column, "").strip()
            if not cell:
                continue
            smiles, ratio = parse_cell(cell)
            molecule_index = molecule_to_index[smiles]
            x[row_index, slot_index] = molecule_tokens[molecule_index]
            adjacency[row_index, slot_index] = molecule_adjacency[molecule_index]
            distance[row_index, slot_index] = molecule_distance[molecule_index]
            molecular_features[row_index, slot_index] = molecule_rdkit_features[molecule_index]
            molar_ratios[row_index, slot_index] = ratio
            molecule_types[row_index, slot_index] = (
                1 if column.startswith("Solvent_") else 2
            )
            slot_index += 1

    temperature_c = np.asarray([float(row["Temperature_C"]) for row in rows], dtype=np.float32)
    temperature_k = np.asarray([float(row["Temperature_K"]) for row in rows], dtype=np.float32)
    temperature_raw = np.stack([temperature_c, 1000.0 / temperature_k], axis=1)
    target_log = np.asarray([float(row["Log10_Viscosity"]) for row in rows], dtype=np.float32)
    target_raw = np.asarray([float(row["Viscosity_mPa_s"]) for row in rows], dtype=np.float32)

    tensors = {
        "x": torch.from_numpy(x),
        "adjacency": torch.from_numpy(adjacency),
        "distance": torch.from_numpy(distance),
        "molecular_features": torch.from_numpy(molecular_features),
        "molar_ratios": torch.from_numpy(molar_ratios),
        "molecule_types": torch.from_numpy(molecule_types),
        "temperature_raw": torch.from_numpy(temperature_raw),
        "target_log": torch.from_numpy(target_log),
        "target_raw": torch.from_numpy(target_raw),
    }
    return TensorBundle(
        tensors=tensors,
        rows=rows,
        references=np.asarray([row["Reference"] for row in rows], dtype=object),
        row_ids=np.asarray([int(row["Row_ID"]) for row in rows], dtype=np.int64),
        electrolytes=np.asarray([row["Electrolyte"] for row in rows], dtype=object),
        vocab_size=vocab_size,
    )


def temperature_statistics(
    bundle: DeviceTensorBundle, train_indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    device = bundle.tensors["temperature_raw"].device
    index = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    values = bundle.tensors["temperature_raw"].index_select(0, index)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std
