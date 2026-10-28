"""
Utility functions for postprocessing functions.
"""

import sys
import os

# Add project root to path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

from config.paths import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from config.datasets import MAX_SOLVENTS, MAX_SALTS, MAX_MOLECULES
from core.utils import (
    load_dataset,
    extract_training_mappings_from_checkpoint,
    feature_engineering,
)
from core.runtime_utils import get_device
from modules.helena_ce.helena_ce_model import HELENA_CE
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE
from modules.tokenizer.tokenizer_helena import Tokenizer
from modules.dataset_processing import FeatureEngineering, CollateFn
from modules.moiety_features import (
    map_atom_to_moiety,
    build_moiety_edges,
    get_adj_matrix,
    get_dist_matrix,
)
from .excluded_moieties import EXCLUDED_MOIETIES

# Dataset and model paths
DATA_PATH = os.path.join(DATA_DIR, "HELENA-CE-26.csv")
MODEL_CHECKPOINT_PATH = os.path.join(DATA_DIR, "helena_ce_trained_weights.pth")

# Postprocess output directories
POSTPROCESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "postprocess")
MOIETY_ANALYSIS_DIR = os.path.join(POSTPROCESS_RESULTS_DIR, "moiety_analysis")
MOIETY_HEATMAPS_DIR = os.path.join(MOIETY_ANALYSIS_DIR, "heatmaps")
MOIETY_VISUALIZ_DIR = os.path.join(MOIETY_ANALYSIS_DIR, "att_visualization")
MOIETY_CUSTOM_SMILES_DIR = os.path.join(MOIETY_ANALYSIS_DIR, "custom_smiles")
DISTANCE_ANALYSIS_DIR = os.path.join(POSTPROCESS_RESULTS_DIR, "distance_analysis")
BINARY_ANALYSIS_DIR = os.path.join(
    POSTPROCESS_RESULTS_DIR, "group_solvent_inference_analysis/binary"
)
# Inference postprocess directories (new pipeline)
INFERENCE_POSTPROCESS_DIR = os.path.join(DATA_DIR, "inference_postprocess")
BINARY_INTERMEDIATES_DIR = os.path.join(INFERENCE_POSTPROCESS_DIR, "binary")
TERNARY_INTERMEDIATES_DIR = os.path.join(INFERENCE_POSTPROCESS_DIR, "ternary")

TERNARY_ANALYSIS_DIR = os.path.join(
    POSTPROCESS_RESULTS_DIR, "group_solvent_inference_analysis/ternary"
)
TOLUENE_FSI_ANALYSIS_DIR = os.path.join(
    POSTPROCESS_RESULTS_DIR, "group_solvent_inference_analysis/toluene_fsi_analysis"
)
GROUP_ATT_ANALYSIS_DIR = os.path.join(POSTPROCESS_RESULTS_DIR, "group_att_analysis")


class MoeityEncoderAttentionUtils:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = MODEL_CHECKPOINT_PATH
        self.checkpoint_path = checkpoint_path
        self.device = get_device()
        self.tokenizer = Tokenizer()
        self.model, self.map_dict, self.moiety_key_map, self.key_moiety_map = (
            self.load_model()
        )

        # Update tokenizer with loaded mappings
        self.tokenizer.moiety_key_map = self.moiety_key_map
        self.tokenizer.key_moiety_map = self.key_moiety_map

        self.custom_cmap = plt.matplotlib.colors.ListedColormap(
            ["white", *plt.cm.YlOrRd(np.linspace(0.2, 1, 100))]
        )

    def load_model(self):
        """Load trained HELENA_CE model from checkpoint with embedded mappings"""
        # Extract mappings from checkpoint
        moiety_key_map, key_moiety_map, vocab_size, checkpoint = (
            extract_training_mappings_from_checkpoint(self.checkpoint_path)
        )

        # Create empty map_dict - will be populated dynamically when needed
        map_dict = {}

        # Get task type from checkpoint
        task_type = checkpoint.get("task_type", "regression")
        num_classes = checkpoint.get("num_classes", None)

        # Initialize HELENA_CE model
        model = HELENA_CE(
            **ARCHITECTURE,
            input_vocab_size=vocab_size,
            max_num_molecules=MAX_MOLECULES,
            task_type=task_type,
            num_classes=num_classes,
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        print(f"Model loaded from: {self.checkpoint_path}")
        print(f"Task type: {task_type}, Vocab size: {vocab_size}")

        return model, map_dict, moiety_key_map, key_moiety_map

    def _ensure_molecule_in_map_dict(self, smiles):
        """Add molecule to map_dict if not present"""
        if smiles in self.map_dict:
            return True

        try:
            result = self.tokenizer.get_moieties(smiles)
            if result is None:
                return False
            clique_atom_ids, edges, moiety_ids, _, is_in_ring = result

            atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
            moiety_edges = build_moiety_edges(edges, atom_to_moiety)

            node_list = list(range(len(clique_atom_ids)))
            moiety_edges_list = list(moiety_edges)

            dist_matrix = get_dist_matrix(node_list, moiety_edges_list)
            adj_matrix = get_adj_matrix(node_list, moiety_edges_list)

            self.map_dict[smiles] = {
                "adj_matrix": adj_matrix,
                "dist_matrix": dist_matrix,
                "clique_atom_ids": clique_atom_ids,
                "edges": edges,
                "moiety_ids": moiety_ids,
                "is_in_ring": is_in_ring,
            }
            return True
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return False

    def get_all_moieties(self):
        """Get all moieties detected in the dataset molecules"""
        dataset = load_dataset(DATA_PATH)
        detected_moieties = set()

        for mol_type, max_num in [
            ("Solvent", MAX_SOLVENTS),
            ("Salt", MAX_SALTS),
        ]:
            for i in range(1, max_num + 1):
                mol_col = f"{mol_type}_{i}"
                if mol_col in dataset.columns:
                    for value in dataset[mol_col].dropna().unique():
                        try:
                            # Parse SMILES:molar_ratio format
                            smiles = str(value).split(":")[0].strip()
                            result = self.tokenizer.get_moieties(smiles)
                            if result:
                                _, _, moiety_ids, *_ = result
                                moiety_names = [
                                    self.key_moiety_map.get(mid)
                                    for mid in moiety_ids
                                    if not str(
                                        self.key_moiety_map.get(mid, "")
                                    ).startswith("ring")
                                ]
                                detected_moieties.update(moiety_names)
                        except Exception as e:
                            print(f"Error processing {value}: {str(e)}")

        return sorted(list(detected_moieties))

    def get_attention_weights(self, mol_smiles, molar_ratio=1.0):
        """Extract attention weights for a molecule"""
        try:
            result = self.tokenizer.get_moieties(mol_smiles)
            if result is None:
                return None, None, None, None
            clique_atom_ids, edges, moiety_ids, *_ = result

            # Ensure molecule is in map_dict
            if not self._ensure_molecule_in_map_dict(mol_smiles):
                return None, None, None, None

            # Create temporary dataframe with new format (SMILES:molar_ratio)
            temp_data = pd.DataFrame(
                {
                    "Solvent_1": [f"{mol_smiles}:{molar_ratio}"],
                    "Salt_1": [None],
                    "LCE": [0.0],
                    "Row_ID": [0],
                    "Method": ["Cycle"],
                    "Current": [0.5],
                }
            )

            # Process using modern FeatureEngineering
            dataset = FeatureEngineering(
                temp_data,
                self.tokenizer,
                self.map_dict,
                label_field="LCE",
                molar_ratio_noise=0.0,  # No noise for inference
            )

            # Create batch
            collate_fn = CollateFn(max_num_mols=MAX_MOLECULES)
            batch = collate_fn([dataset[0]])

            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                x1 = batch["x1"]
                adj_matrix = batch["adjoin_matrix1"]
                dist_matrix = batch["dist_matrix1"]

                _, local_attn, global_attn, _ = self.model.encoder_moiety(
                    x1, adjoin_matrix=adj_matrix, dist_matrix=dist_matrix
                )

                local_attn = torch.stack(local_attn).mean(0).squeeze(0).mean(0)
                global_attn = torch.stack(global_attn).mean(0).squeeze(0).mean(0)

                return (
                    local_attn.cpu().numpy(),
                    global_attn.cpu().numpy(),
                    moiety_ids,
                    clique_atom_ids,
                )

        except Exception as e:
            import traceback

            print(f"Error processing molecule {mol_smiles}:")
            print(traceback.format_exc())
            return None, None, None, None

    def analyze_extreme_lce_formulations(self, n_top=5, n_bottom=5, results_dir=None):
        """Analyze moieties in extreme LCE formulations"""
        if results_dir is None:
            results_dir = MOIETY_ANALYSIS_DIR

        dataset = load_dataset(DATA_PATH)
        results = []

        for performance, formulations in [
            ("highest", dataset.nlargest(n_top, "LCE")),
            ("lowest", dataset.nsmallest(n_bottom, "LCE")),
        ]:
            for idx, formulation in formulations.iterrows():
                for mol_type, max_num in [
                    ("Solvent", MAX_SOLVENTS),
                    ("Salt", MAX_SALTS),
                ]:
                    for i in range(1, max_num + 1):
                        mol_col = f"{mol_type}_{i}"

                        if mol_col in formulation and pd.notna(formulation[mol_col]):
                            value = str(formulation[mol_col])
                            # Parse SMILES:molar_ratio format
                            parts = value.split(":")
                            if len(parts) != 2:
                                continue
                            smiles = parts[0].strip()
                            molar_ratio = float(parts[1].strip())

                            local_attn, global_attn, moiety_ids, _ = (
                                self.get_attention_weights(smiles, molar_ratio)
                            )

                            if local_attn is not None:
                                moiety_names = [
                                    self.key_moiety_map.get(mid, f"moiety_{mid}")
                                    for mid in moiety_ids
                                ]
                                results.extend(
                                    [
                                        {
                                            "Row_ID": formulation.get("Row_ID", idx),
                                            "formulation_id": idx,
                                            "performance": performance,
                                            "molecule_type": mol_type.lower(),
                                            "LCE": formulation["LCE"],
                                            "molecule": smiles,
                                            "molar_ratio": molar_ratio,
                                            "moiety": moiety_name,
                                            "local_attention_score": float(
                                                local_attn[i_m].mean()
                                            ),
                                            "global_attention_score": float(
                                                global_attn[i_m].mean()
                                            ),
                                        }
                                        for i_m, moiety_name in enumerate(moiety_names)
                                    ]
                                )

        if not results:
            print("No valid results found for analysis")
            return pd.DataFrame(columns=["Row_ID", "performance", "molecule", "LCE"])

        results_df = pd.DataFrame(results)
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(
            os.path.join(results_dir, "extreme_lce_moiety_analysis.csv"),
            index=False,
        )
        return results_df


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi
