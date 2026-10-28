"""
Toluene-Solvent in LiFSI Interaction Visualization Tool
Visualizes moiety co-attention scores between toluene and other solvents
"""

########################
## Parameters
TOLUENE_SMILES = "CC1=CC=CC=C1"
FSI_SALT = "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"
FSI_NAME = "FSI"

MOLECULES_TO_VISUALIZE = {
    "MBE": {"smiles": "COCCCC", "molar_ratio": 4.22},
    "MTBE": {"smiles": "COC(C)(C)(C)", "molar_ratio": 4.2},
    "DBE": {"smiles": "CCCCOCCCC", "molar_ratio": 2.96},
}

TOLUENE_MOLAR_RATIO = 4.7

# Moiety name abbreviations for cleaner heatmap labels
MOIETY_ABBREVIATIONS = {
    "secondary_carbon": "sec_C",
    "quaternary_carbon": "quat_C",
}
########################

import os
import sys

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem

from config.datasets import MAX_MOLECULES
from core.runtime_utils import get_device
from modules.dataset_processing import FeatureEngineering, CollateFn
from modules.moiety_features import (
    map_atom_to_moiety,
    build_moiety_edges,
    get_adj_matrix,
    get_dist_matrix,
)
from tasks.postprocess.postprocess_utils.postprocess_utils import (
    MoeityEncoderAttentionUtils,
    TOLUENE_FSI_ANALYSIS_DIR,
    MODEL_CHECKPOINT_PATH,
)

import warnings
warnings.filterwarnings("ignore")

TOLUENE_FSI_VIZ_DIR = os.path.join(TOLUENE_FSI_ANALYSIS_DIR, "individual_visualizations")
CONSOLIDATED_VIZ_DIR = os.path.join(TOLUENE_FSI_ANALYSIS_DIR, "consolidated_view")

for directory in [TOLUENE_FSI_ANALYSIS_DIR, TOLUENE_FSI_VIZ_DIR, CONSOLIDATED_VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

# Custom colormap that starts from light purple instead of white
CUSTOM_PURPLES = LinearSegmentedColormap.from_list(
    "custom_purples",
    ["#f0e8f8", "#9370DB", "#4B0082"]  # Lighter purple -> Medium purple -> Dark purple
)


class TolueneMoietyAttentionVisualizer(MoeityEncoderAttentionUtils):
    """
    Visualizes moiety co-attention scores between toluene and other solvent molecules
    using the trained HELENA-CE model's attention mechanism.
    """

    def __init__(self, molecules_to_visualize):
        super().__init__()
        self.molecules_to_visualize = molecules_to_visualize

        print(f"Model loaded from: {MODEL_CHECKPOINT_PATH}")

        self.toluene_mol = Chem.MolFromSmiles(TOLUENE_SMILES)
        if self.toluene_mol is None:
            raise ValueError(f"Invalid SMILES for toluene: {TOLUENE_SMILES}")

        result = self.tokenizer.get_moieties(TOLUENE_SMILES)
        if result is None:
            raise ValueError(f"Could not get moieties for toluene: {TOLUENE_SMILES}")

        self.toluene_cliques, self.toluene_edges, self.toluene_moiety_ids, self.toluene_matches, _ = result

        # Ensure toluene is in map_dict
        self._ensure_molecule_in_map_dict(TOLUENE_SMILES)

        # Ensure FSI salt is in map_dict
        self._ensure_molecule_in_map_dict(FSI_SALT)

        self.all_interaction_data = []

    def _abbreviate_moiety_name(self, name):
        """Convert long moiety names to abbreviations."""
        return MOIETY_ABBREVIATIONS.get(name, name)

    def run_analysis(self):
        """Run the analysis for all molecules with raw attention scores"""

        for name, solvent_info in self.molecules_to_visualize.items():
            if isinstance(solvent_info, dict):
                smiles = solvent_info["smiles"]
                molar_ratio = solvent_info.get("molar_ratio", 1.0)
            else:
                smiles = solvent_info
                molar_ratio = 1.0

            print(f"\nProcessing {name} ({smiles}) with molar ratio {molar_ratio}...")

            solvent_mol = Chem.MolFromSmiles(smiles)
            if solvent_mol is None:
                print(f"Invalid SMILES for {name}")
                continue

            # Ensure solvent is in map_dict
            if not self._ensure_molecule_in_map_dict(smiles):
                print(f"Failed to add {smiles} to map_dict, skipping...")
                continue

            result = self.tokenizer.get_moieties(smiles)
            if result is None:
                print(f"Could not get moieties for {smiles}")
                continue

            solvent_cliques, solvent_edges, solvent_moiety_ids, solvent_matches, _ = result

            interaction_data = self._analyze_attention(
                smiles,
                name,
                solvent_moiety_ids,
                solvent_matches,
                molar_ratio=molar_ratio,
            )
            if interaction_data is not None:
                self.all_interaction_data.append((name, smiles, interaction_data))

        if self.all_interaction_data:
            all_values = []
            for _, _, interaction_data in self.all_interaction_data:
                all_values.extend(interaction_data["raw_attention"].flatten())

            global_min = min(all_values)
            global_max = max(all_values)

            print(f"\nGlobal attention range: {global_min:.6f} to {global_max:.6f}")

            for name, smiles, interaction_data in self.all_interaction_data:
                tol_img_path = os.path.join(TOLUENE_FSI_VIZ_DIR, "toluene_highlighted.png")
                sol_img_path = os.path.join(TOLUENE_FSI_VIZ_DIR, f"{name.replace(' ', '_')}_highlighted.png")

                self._create_individual_visualization(
                    tol_img_path,
                    sol_img_path,
                    interaction_data["attention_df_raw"],
                    name,
                    vmin=global_min,
                    vmax=global_max,
                )

            self._create_consolidated_visualization(global_min, global_max)

        print("\nAnalysis complete!")

    def _analyze_attention(self, solvent_smiles, name, solvent_moiety_ids, solvent_matches, molar_ratio=1.0):
        """
        Analyze attention between toluene and solvent molecule moieties.
        Returns raw attention values without normalization.
        """
        try:
            print(f"Computing moiety-level attention for {name} with molar ratio {molar_ratio}...")

            # Create temporary dataframe with new format (SMILES:molar_ratio)
            temp_data = pd.DataFrame({
                "Solvent_1": [f"{TOLUENE_SMILES}:{TOLUENE_MOLAR_RATIO}"],
                "Solvent_2": [f"{solvent_smiles}:{molar_ratio}"],
                "Salt_1": [f"{FSI_SALT}:1.0"],
                "LCE": [0.0],
                "Row_ID": [0],
                "Method": ["Cycle"],
                "Current": [0.5],
            })

            # Process using FeatureEngineering
            dataset = FeatureEngineering(
                temp_data,
                self.tokenizer,
                self.map_dict,
                label_field="LCE",
                molar_ratio_noise=0.0,
            )

            collate_fn = CollateFn(max_num_mols=MAX_MOLECULES)
            batch = collate_fn([dataset[0]])

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                x1 = batch["x1"]
                adj_matrix1 = batch["adjoin_matrix1"]
                dist_matrix1 = batch["dist_matrix1"]

                tol_emb, _, _, _ = self.model.encoder_moiety(
                    x1, adjoin_matrix=adj_matrix1, dist_matrix=dist_matrix1
                )

                x2 = batch["x2"]
                adj_matrix2 = batch["adjoin_matrix2"]
                dist_matrix2 = batch["dist_matrix2"]

                sol_emb, _, _, _ = self.model.encoder_moiety(
                    x2, adjoin_matrix=adj_matrix2, dist_matrix=dist_matrix2
                )

                tol_mask = x1.squeeze(0) != 0
                sol_mask = x2.squeeze(0) != 0

                tol_emb_trimmed = tol_emb.squeeze(0)[tol_mask]
                sol_emb_trimmed = sol_emb.squeeze(0)[sol_mask]

                d_model = tol_emb_trimmed.size(-1)
                attention_matrix = torch.matmul(
                    sol_emb_trimmed, tol_emb_trimmed.transpose(0, 1)
                ) / (d_model ** 0.5)

                attention_scores = torch.nn.functional.softmax(attention_matrix, dim=-1)

                tol_mr = torch.tensor(TOLUENE_MOLAR_RATIO, device=self.device)
                sol_mr = torch.tensor(molar_ratio, device=self.device)
                total_mr = tol_mr + sol_mr
                sol_mr_norm = sol_mr / total_mr

                weighted_attention = attention_scores * sol_mr_norm
                moiety_attention_raw = weighted_attention.cpu().numpy()

                solvent_moiety_names_raw = [
                    self.key_moiety_map.get(mid, f"Moiety_{mid}")
                    for mid in solvent_moiety_ids
                ]
                toluene_moiety_names_raw = [
                    self.key_moiety_map.get(mid, f"Moiety_{mid}")
                    for mid in self.toluene_moiety_ids
                ]

                solvent_moiety_names = [
                    self._abbreviate_moiety_name(name) for name in solvent_moiety_names_raw
                ]
                toluene_moiety_names = [
                    self._abbreviate_moiety_name(name) for name in toluene_moiety_names_raw
                ]

                attention_df_raw = pd.DataFrame(
                    moiety_attention_raw,
                    index=solvent_moiety_names,
                    columns=toluene_moiety_names,
                )

                # Create molecule highlight images
                tol_img_path = os.path.join(TOLUENE_FSI_VIZ_DIR, "toluene_highlighted.png")
                sol_img_path = os.path.join(TOLUENE_FSI_VIZ_DIR, f"{name.replace(' ', '_')}_highlighted.png")

                solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                self.tokenizer.highlight_and_display_molecule(
                    self.toluene_mol, self.toluene_matches, save_path=tol_img_path
                )
                self.tokenizer.highlight_and_display_molecule(
                    solvent_mol, solvent_matches, save_path=sol_img_path
                )

                # Create individual visualization
                self._create_individual_visualization(
                    tol_img_path, sol_img_path, attention_df_raw, name
                )

                print(f"Successfully computed moiety-level attention for {name}")

                return {
                    "toluene_img": plt.imread(tol_img_path),
                    "solvent_img": plt.imread(sol_img_path),
                    "attention_df_raw": attention_df_raw,
                    "toluene_moiety_ids": self.toluene_moiety_ids,
                    "solvent_moiety_ids": solvent_moiety_ids,
                    "raw_attention": moiety_attention_raw,
                    "toluene_molar_ratio": TOLUENE_MOLAR_RATIO,
                    "solvent_molar_ratio": molar_ratio,
                }

        except Exception as e:
            print(f"Error analyzing attention for {name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_individual_visualization(self, toluene_img_path, solvent_img_path, attention_df, name, vmin=None, vmax=None):
        """Create and save visualization for a single molecule pair."""
        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.3, 1.6])

        toluene_img = plt.imread(toluene_img_path)
        solvent_img = plt.imread(solvent_img_path)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(toluene_img)
        ax1.set_title("Toluene")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(solvent_img)
        ax2.set_title(name)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 3])

        heatmap_kwargs = {
            "annot": True,
            "cmap": CUSTOM_PURPLES,
            "linewidths": 0.5,
            "fmt": ".4f",
            "annot_kws": {"size": 20},
        }

        if vmin is not None:
            heatmap_kwargs["vmin"] = vmin
        if vmax is not None:
            heatmap_kwargs["vmax"] = vmax

        heatmap = sns.heatmap(attention_df, ax=ax3, **heatmap_kwargs)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        ax3.set_ylabel(f"Moieties in {name}", fontsize=22)
        ax3.set_xlabel("Moieties in Toluene", fontsize=22)
        ax3.yaxis.set_label_coords(-0.22, 0.5)
        plt.setp(ax3.get_xticklabels(), rotation=0, ha="center", fontsize=20)
        # Tilted y-axis labels for solvent moieties - shifted down by half cell
        plt.setp(ax3.get_yticklabels(), rotation=45, ha="right", va="top", fontsize=20)

        plt.tight_layout()

        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        filename = os.path.join(TOLUENE_FSI_VIZ_DIR, f"toluene_{safe_name}_attention.png")
        plt.savefig(filename, dpi=900, bbox_inches="tight")
        plt.close()

        print(f"Individual visualization created for Toluene + {name}")

    def _create_consolidated_visualization(self, vmin=None, vmax=None):
        """Create a consolidated visualization showing all molecule combinations."""
        if not self.all_interaction_data:
            print("No interaction data available for consolidated view.")
            return

        num_molecules = len(self.all_interaction_data)

        fig = plt.figure(figsize=(20, 9 * num_molecules))
        fig.suptitle(
            f"Toluene-Solvent Interactions in LiFSI Electrolyte",
            fontsize=24,
            fontweight="bold",
            y=0.92,
        )

        gs = GridSpec(
            num_molecules,
            4,
            width_ratios=[1, 1, 0.3, 1.6],
            height_ratios=[1] * num_molecules,
            hspace=0.4,
            wspace=0.1,
            figure=fig,
        )

        for i, (name, smiles, interaction_data) in enumerate(self.all_interaction_data):
            ax_toluene = fig.add_subplot(gs[i, 0])
            ax_toluene.imshow(interaction_data["toluene_img"])
            ax_toluene.set_title("Toluene")
            ax_toluene.axis("off")

            ax_solvent = fig.add_subplot(gs[i, 1])
            ax_solvent.imshow(interaction_data["solvent_img"])
            ax_solvent.set_title(name)
            ax_solvent.axis("off")

            ax_heatmap = fig.add_subplot(gs[i, 3])

            raw_attention = interaction_data["raw_attention"]
            attention_df = pd.DataFrame(
                raw_attention,
                index=interaction_data["attention_df_raw"].index,
                columns=interaction_data["attention_df_raw"].columns,
            )

            heatmap = sns.heatmap(
                attention_df,
                annot=True,
                cmap=CUSTOM_PURPLES,
                linewidths=0.5,
                fmt=".4f",
                ax=ax_heatmap,
                vmin=vmin,
                vmax=vmax,
                annot_kws={"size": 20},
            )

            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)

            ax_heatmap.set_ylabel(f"Moieties in {name}", fontsize=22)
            ax_heatmap.set_xlabel("Moieties in Toluene", fontsize=22)
            ax_heatmap.yaxis.set_label_coords(-0.22, 0.5)
            plt.setp(ax_heatmap.get_xticklabels(), rotation=0, ha="center", fontsize=20)
            # Tilted y-axis labels for solvent moieties - shifted down by half cell
            plt.setp(ax_heatmap.get_yticklabels(), rotation=45, ha="right", va="top", fontsize=20)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = os.path.join(CONSOLIDATED_VIZ_DIR, "toluene_fsi_summary.png")
        plt.savefig(filename, dpi=900, bbox_inches="tight")
        plt.close()

        print(f"\nConsolidated visualization created with {num_molecules} molecule combinations")


def main():
    """Main function to run the visualizer"""
    print("\n" + "+" * 50)
    print("TOLUENE-SOLVENT IN LiFSI INTERACTION VISUALIZATION")
    print("+" * 50 + "\n")

    if not MOLECULES_TO_VISUALIZE:
        print("Error: No molecules to visualize defined!")
        return

    try:
        visualizer = TolueneMoietyAttentionVisualizer(MOLECULES_TO_VISUALIZE)
        visualizer.run_analysis()

        print(f"\nVisualization complete.\nResults saved to: {TOLUENE_FSI_ANALYSIS_DIR}\n")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
