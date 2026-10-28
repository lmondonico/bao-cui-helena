"""
Moiety attention heatmap analysis from HELENA model
Extracts and visualizes both local and global attention weights and creates summary heatmaps
"""

########################
## Parameters
summary_best_molecules = 60
summary_worst_molecules = 60

# Dictionary of SMILES to analyze with custom labels
SMILES_TO_ANALYZE = {
    "DME": "COCCOC",
    "BTMSP": "C[Si](C)(C)OCCCO[Si](C)(C)C",
    "BFEM": "FCCOCCOCCF",
    "LiTFTFB": "[Li+].FC(F)(F)CO[B-](F)(F)F",
}
########################

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
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

from config.datasets import MAX_MOLECULES
from modules.dataset_processing import FeatureEngineering, CollateFn
from tasks.postprocess.postprocess_utils.postprocess_utils import (
    MoeityEncoderAttentionUtils,
    MOIETY_HEATMAPS_DIR,
)
from tasks.postprocess.postprocess_utils.excluded_moieties import EXCLUDED_MOIETIES

import warnings

warnings.filterwarnings("ignore")

os.makedirs(MOIETY_HEATMAPS_DIR, exist_ok=True)
MOIETY_HEATMAPS_SUMMARY_DIR = os.path.join(MOIETY_HEATMAPS_DIR, "summary")
os.makedirs(MOIETY_HEATMAPS_SUMMARY_DIR, exist_ok=True)
MOIETY_HEATMAPS_MULTIHEAD_DIR = os.path.join(MOIETY_HEATMAPS_DIR, "multihead")
os.makedirs(MOIETY_HEATMAPS_MULTIHEAD_DIR, exist_ok=True)


class MoietyEncoderAttentionHeatmaps(MoeityEncoderAttentionUtils):
    """Extended analyzer class for moiety attention visualization"""

    def __init__(self):
        super().__init__()
        self.show_device_used = True

    def _draw_molecule_image(self, smiles, width=400, height=300):
        """Draw molecule structure and return as PIL Image"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().padding = 0.02
        drawer.drawOptions().bondLineWidth = 6  # thick bonds
        drawer.drawOptions().multipleBondOffset = 0.18
        drawer.drawOptions().minFontSize = 50  # much larger atom labels
        drawer.drawOptions().maxFontSize = 80
        drawer.drawOptions().baseFontSize = 1.0
        drawer.drawOptions().annotationFontScale = 1.5
        # no fixedBondLength - let it scale to fill the canvas
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        img_data = drawer.GetDrawingText()
        return Image.open(io.BytesIO(img_data))

    def visualize_multihead_attention_combined(
        self, smiles_dict, molar_ratio=1.0, output_dir=None
    ):
        """
        Visualize attention weights for each molecule in smiles_dict.
        Creates a single combined image per molecule with:
        - Molecule structure centered at the top
        - Local attention heads below (labeled 'a')
        - Global attention heads at the bottom (labeled 'b')

        Args:
            smiles_dict: Dictionary of {label: smiles}
            molar_ratio: Molar ratio for the molecules
            output_dir: Directory to save outputs (default: MOIETY_HEATMAPS_MULTIHEAD_DIR)
        """
        if output_dir is None:
            output_dir = MOIETY_HEATMAPS_MULTIHEAD_DIR

        for label, mol_smiles in tqdm(smiles_dict.items(), desc="Processing molecules"):
            base_attn_info = self.get_attention_weights(mol_smiles, molar_ratio)
            if base_attn_info[0] is None:
                print(f"Skipping {label}: could not get attention weights")
                continue

            try:
                # Ensure molecule is in map_dict
                self._ensure_molecule_in_map_dict(mol_smiles)

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
                    molar_ratio_noise=0.0,
                )

                # Create batch
                collate_fn = CollateFn(max_num_mols=MAX_MOLECULES)
                batch = collate_fn([dataset[0]])

                # Move to device
                processed_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.no_grad():
                    _, local_attn, global_attn, _ = self.model.encoder_moiety(
                        processed_batch["x1"],
                        adjoin_matrix=processed_batch["adjoin_matrix1"],
                        dist_matrix=processed_batch["dist_matrix1"],
                    )

                    local_attn = torch.stack(local_attn).mean(0).squeeze(0)
                    global_attn = torch.stack(global_attn).mean(0).squeeze(0)

                    _, _, moiety_ids, _ = base_attn_info
                    moiety_names = [
                        self.tokenizer.key_moiety_map.get(mid) for mid in moiety_ids
                    ]
                    n_heads = local_attn.size(0)

                    # Calculate grid layout
                    # We want heads in a single row or 2 rows max
                    if n_heads <= 4:
                        head_cols = n_heads
                        head_rows = 1
                    else:
                        head_cols = 4
                        head_rows = int(np.ceil(n_heads / 4))

                    # Create figure with GridSpec for precise layout
                    # Top: molecule image (spanning all columns)
                    # Middle: local attention heads (labeled 'a')
                    # Bottom: global attention heads (labeled 'b')
                    fig_height = (
                        8 + 8 * head_rows * 2
                    )  # molecule + local + global (larger)
                    fig_width = 9 * head_cols  # wider

                    fig = plt.figure(figsize=(fig_width, fig_height))

                    # Create GridSpec with 5 rows: molecule, spacer, local heads, spacer, global heads
                    from matplotlib.gridspec import GridSpec

                    gs = GridSpec(
                        5,  # molecule row, spacer, local row, spacer, global row
                        head_cols,
                        figure=fig,
                        height_ratios=[
                            1.2,
                            0.35,
                            1.5,
                            1.2,
                            1.5,
                        ],  # larger molecule, bigger gap below molecule, heads, big spacer between heatmap rows, heads
                        hspace=0.12,
                        wspace=0.5,  # more horizontal space between heatmaps
                    )

                    # Draw molecule at top, centered (no title/label)
                    ax_mol = fig.add_subplot(gs[0, :])
                    mol_img = self._draw_molecule_image(
                        mol_smiles, width=2400, height=700
                    )
                    if mol_img is not None:
                        ax_mol.imshow(mol_img)
                    ax_mol.axis("off")

                    # Draw local attention heads (row 2, after molecule spacer)
                    for head in range(n_heads):
                        col_idx = head % head_cols
                        ax = fig.add_subplot(gs[2, col_idx])

                        sns.heatmap(
                            local_attn[head].cpu().numpy(),
                            xticklabels=moiety_names,
                            yticklabels=moiety_names,
                            cmap="YlOrRd",
                            ax=ax,
                            vmin=0,
                            cbar=True,
                            cbar_kws={"shrink": 0.8},
                        )
                        ax.set_title(f"Head {head + 1}", fontsize=26, fontweight="bold")
                        ax.set_xticklabels(
                            ax.get_xticklabels(), fontsize=18, rotation=45, ha="right"
                        )
                        ax.set_yticklabels(
                            ax.get_yticklabels(), fontsize=18, rotation=0
                        )

                        # Add 'a' label on first head only
                        if head == 0:
                            ax.text(
                                -0.5,
                                1.25,
                                "a",
                                transform=ax.transAxes,
                                fontsize=42,
                                fontweight="bold",
                                va="bottom",
                                ha="left",
                            )

                    # Draw global attention heads (row 4, after spacer)
                    for head in range(n_heads):
                        col_idx = head % head_cols
                        ax = fig.add_subplot(gs[4, col_idx])

                        sns.heatmap(
                            global_attn[head].cpu().numpy(),
                            xticklabels=moiety_names,
                            yticklabels=moiety_names,
                            cmap="Blues",
                            ax=ax,
                            vmin=0,
                            cbar=True,
                            cbar_kws={"shrink": 0.8},
                        )
                        ax.set_title(f"Head {head + 1}", fontsize=26, fontweight="bold")
                        ax.set_xticklabels(
                            ax.get_xticklabels(), fontsize=18, rotation=45, ha="right"
                        )
                        ax.set_yticklabels(
                            ax.get_yticklabels(), fontsize=18, rotation=0
                        )

                        # Add 'b' label on first head only
                        if head == 0:
                            ax.text(
                                -0.5,
                                1.25,
                                "b",
                                transform=ax.transAxes,
                                fontsize=42,
                                fontweight="bold",
                                va="bottom",
                                ha="left",
                            )

                    # Save figure
                    safe_label = label.replace("/", "_").replace("\\", "_")
                    output_path = os.path.join(
                        output_dir, f"multihead_attention_{safe_label}.png"
                    )
                    plt.savefig(
                        output_path, dpi=300, bbox_inches="tight", facecolor="white"
                    )
                    plt.close(fig)
                    print(f"Saved: {output_path}")

            except Exception as e:
                print(
                    f"Error visualizing attention for {label} ({mol_smiles}): {str(e)}"
                )
                import traceback

                traceback.print_exc()

    def visualize_head_attention(
        self, mol_smiles, molar_ratio=1.0, performance_dir=None
    ):
        """Visualize attention weights for each attention head separately (legacy method)"""
        base_attn_info = self.get_attention_weights(mol_smiles, molar_ratio)
        if base_attn_info[0] is None:
            return

        try:
            # Ensure molecule is in map_dict
            self._ensure_molecule_in_map_dict(mol_smiles)

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
                molar_ratio_noise=0.0,
            )

            # Create batch
            collate_fn = CollateFn(max_num_mols=MAX_MOLECULES)
            batch = collate_fn([dataset[0]])

            # Move to device
            processed_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                _, local_attn, global_attn, _ = self.model.encoder_moiety(
                    processed_batch["x1"],
                    adjoin_matrix=processed_batch["adjoin_matrix1"],
                    dist_matrix=processed_batch["dist_matrix1"],
                )

                local_attn = torch.stack(local_attn).mean(0).squeeze(0)
                global_attn = torch.stack(global_attn).mean(0).squeeze(0)

                _, _, moiety_ids, _ = base_attn_info
                moiety_names = [
                    self.tokenizer.key_moiety_map.get(mid) for mid in moiety_ids
                ]
                n_heads = local_attn.size(0)
                rows = int(np.ceil(n_heads / 4))

                # Create visualization
                fig_local, axes_local = plt.subplots(rows, 4, figsize=(20, 5 * rows))
                fig_global, axes_global = plt.subplots(rows, 4, figsize=(20, 5 * rows))

                for head in range(n_heads):
                    row, col = head // 4, head % 4

                    # Plot local attention
                    ax_local = axes_local[row, col] if rows > 1 else axes_local[col]
                    sns.heatmap(
                        local_attn[head].cpu().numpy(),
                        xticklabels=moiety_names,
                        yticklabels=moiety_names,
                        cmap="YlOrRd",
                        ax=ax_local,
                        vmin=0,
                    )
                    ax_local.set_title(f"Local Attention Head {head}")
                    ax_local.set_xticklabels(
                        ax_local.get_xticklabels(), fontsize=11, rotation=45, ha="right"
                    )
                    ax_local.set_yticklabels(
                        ax_local.get_yticklabels(), fontsize=11, rotation=0
                    )

                    # Plot global attention
                    ax_global = axes_global[row, col] if rows > 1 else axes_global[col]
                    sns.heatmap(
                        global_attn[head].cpu().numpy(),
                        xticklabels=moiety_names,
                        yticklabels=moiety_names,
                        cmap="Blues",
                        ax=ax_global,
                        vmin=0,
                    )
                    ax_global.set_title(f"Global Attention Head {head}")
                    ax_global.set_xticklabels(
                        ax_global.get_xticklabels(),
                        fontsize=11,
                        rotation=45,
                        ha="right",
                    )
                    ax_global.set_yticklabels(
                        ax_global.get_yticklabels(), fontsize=11, rotation=0
                    )

                # Save plots
                save_dir = performance_dir if performance_dir else MOIETY_HEATMAPS_DIR
                for fig, att_type in [(fig_local, "local"), (fig_global, "global")]:
                    plt.figure(fig.number)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(
                        os.path.join(
                            save_dir, f"head_attention_{att_type}_{mol_smiles[:30]}.png"
                        ),
                        dpi=600,
                    )
                plt.close("all")

        except Exception as e:
            print(f"Error visualizing attention for {mol_smiles}: {str(e)}")

    def create_summary_heatmaps(
        self, results_df, excluded_moieties=None, performances=("highest", "lowest")
    ):
        """Create summary heatmaps of moiety interactions"""
        if excluded_moieties is None:
            excluded_moieties = []

        for performance in performances:
            perf_df = results_df[results_df["performance"] == performance]
            moiety_pair_attentions = defaultdict(list)
            moiety_set = set()

            for _, row in perf_df.iterrows():
                local_attn, global_attn, moiety_ids, _ = self.get_attention_weights(
                    row["molecule"], row["molar_ratio"]
                )
                if local_attn is None:
                    continue

                moiety_names = [
                    self.tokenizer.key_moiety_map.get(mid) for mid in moiety_ids
                ]
                valid_indices = [
                    i
                    for i, name in enumerate(moiety_names)
                    if name not in excluded_moieties
                ]

                if valid_indices:
                    filtered_names = [moiety_names[i] for i in valid_indices]
                    filtered_local = local_attn[valid_indices][:, valid_indices]
                    filtered_global = global_attn[valid_indices][:, valid_indices]

                    moiety_set.update(filtered_names)
                    for i, m1 in enumerate(filtered_names):
                        for j, m2 in enumerate(filtered_names):
                            moiety_pair_attentions[(m1, m2)].append(
                                {
                                    "local": filtered_local[i, j],
                                    "global": filtered_global[i, j],
                                }
                            )

            # Create average attention matrices
            moiety_list = sorted(list(moiety_set))
            n_moieties = len(moiety_list)
            local_matrix = np.zeros((n_moieties, n_moieties))
            global_matrix = np.zeros((n_moieties, n_moieties))

            for i, m1 in enumerate(moiety_list):
                for j, m2 in enumerate(moiety_list):
                    attentions = moiety_pair_attentions.get((m1, m2), [])
                    if attentions:
                        local_matrix[i, j] = np.min(
                            [att["local"] for att in attentions]
                        )
                        global_matrix[i, j] = np.min(
                            [att["global"] for att in attentions]
                        )

            # Plot heatmaps
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            for ax, matrix, title, cmap in [
                (
                    ax1,
                    local_matrix,
                    f"Average Local Attention ({performance} LCE)",
                    self.custom_cmap,
                ),
                (
                    ax2,
                    global_matrix,
                    f"Average Global Attention ({performance} LCE)",
                    "Blues",
                ),
            ]:
                sns.heatmap(
                    matrix,
                    xticklabels=moiety_list,
                    yticklabels=moiety_list,
                    cmap=cmap,
                    annot=True,
                    fmt=".2f",
                    ax=ax,
                    vmin=0,
                    mask=matrix == 0,
                )
                ax.set_title(title, fontsize=20)
                ax.set_xlabel("Reference Moiety", fontsize=16)
                ax.set_ylabel("Target Moiety", fontsize=16)
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=45, ha="right", fontsize=14
                )
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(
                os.path.join(
                    MOIETY_HEATMAPS_SUMMARY_DIR, f"summary_heatmap_{performance}.png"
                ),
                bbox_inches="tight",
                dpi=600,
            )
            plt.close()

    def create_all_moieties_summary_heatmaps(
        self, results_df, excluded_moieties=None, performances=("highest", "lowest")
    ):
        """Create separate summary heatmaps for highest and lowest LCE moieties"""
        all_moieties = self.get_all_moieties()
        if excluded_moieties:
            all_moieties = [m for m in all_moieties if m not in excluded_moieties]
        print(f"Processing {len(all_moieties)} moieties...")

        # Process each performance level separately
        for performance in performances:
            perf_df = results_df[results_df["performance"] == performance]

            n_moieties = len(all_moieties)
            local_matrix = np.zeros((n_moieties, n_moieties))
            global_matrix = np.zeros((n_moieties, n_moieties))
            counts = np.zeros((n_moieties, n_moieties))

            for _, row in perf_df.iterrows():
                local_attn, global_attn, moiety_ids, _ = self.get_attention_weights(
                    row["molecule"], row["molar_ratio"]
                )
                if local_attn is not None:
                    for i, mid1 in enumerate(moiety_ids):
                        m1 = self.tokenizer.key_moiety_map.get(mid1)
                        if m1 in all_moieties:
                            idx1 = all_moieties.index(m1)
                            for j, mid2 in enumerate(moiety_ids):
                                m2 = self.tokenizer.key_moiety_map.get(mid2)
                                if m2 in all_moieties:
                                    idx2 = all_moieties.index(m2)
                                    local_matrix[idx1, idx2] += local_attn[i, j]
                                    global_matrix[idx1, idx2] += global_attn[i, j]
                                    counts[idx1, idx2] += 1

            # Average matrices
            mask = counts > 0
            local_matrix[mask] /= counts[mask]
            global_matrix[mask] /= counts[mask]

            # Plot heatmaps for this performance level
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

            for ax, matrix, title, cmap in [
                (
                    ax1,
                    local_matrix,
                    f"Average Local Attention ({performance} LCE)",
                    self.custom_cmap,
                ),
                (
                    ax2,
                    global_matrix,
                    f"Average Global Attention ({performance} LCE)",
                    "Blues",
                ),
            ]:
                sns.heatmap(
                    matrix,
                    xticklabels=all_moieties,
                    yticklabels=all_moieties,
                    cmap=cmap,
                    ax=ax,
                    annot=False,
                    fmt=".2f",
                    vmin=0,
                )
                ax.set_title(title, fontsize=20)
                ax.set_xlabel("Reference Moiety", fontsize=16)
                ax.set_ylabel("Target Moiety", fontsize=16)
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=45, ha="right", fontsize=12
                )
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            suffix = "_filtered" if excluded_moieties else ""
            plt.savefig(
                os.path.join(
                    MOIETY_HEATMAPS_SUMMARY_DIR,
                    f"summary_heatmap_all_moieties_{performance}{suffix}.png",
                ),
                bbox_inches="tight",
                dpi=600,
            )
            plt.close()

    def create_stacked_publication_figures(self):
        """
        Create publication-ready figures that stack:
        - Top (a): summary_heatmap_highest/lowest
        - Bottom (b): summary_heatmap_all_moieties_highest/lowest_filtered

        Creates two figures: one for highest LCE and one for lowest LCE.
        """
        from matplotlib.gridspec import GridSpec

        for performance in ["highest", "lowest"]:
            # Load the two images
            summary_path = os.path.join(
                MOIETY_HEATMAPS_SUMMARY_DIR, f"summary_heatmap_{performance}.png"
            )
            all_moieties_path = os.path.join(
                MOIETY_HEATMAPS_SUMMARY_DIR, f"summary_heatmap_all_moieties_{performance}_filtered.png"
            )

            # Check if files exist
            if not os.path.exists(summary_path):
                print(f"Warning: {summary_path} not found, skipping stacked figure for {performance}")
                continue
            if not os.path.exists(all_moieties_path):
                print(f"Warning: {all_moieties_path} not found, skipping stacked figure for {performance}")
                continue

            # Load images and crop whitespace
            img_top = Image.open(summary_path)
            img_bottom = Image.open(all_moieties_path)

            # Crop whitespace from images
            img_top_array = np.array(img_top)
            img_bottom_array = np.array(img_bottom)

            # Find non-white rows for top image (crop bottom whitespace)
            non_white_rows_top = np.where(np.any(img_top_array < 250, axis=(1, 2)))[0]
            if len(non_white_rows_top) > 0:
                img_top = img_top.crop((0, 0, img_top.width, non_white_rows_top[-1] + 10))

            # Find non-white rows for bottom image (crop top whitespace)
            non_white_rows_bottom = np.where(np.any(img_bottom_array < 250, axis=(1, 2)))[0]
            if len(non_white_rows_bottom) > 0:
                img_bottom = img_bottom.crop((0, max(0, non_white_rows_bottom[0] - 10), img_bottom.width, img_bottom.height))

            # Create figure with two rows
            fig = plt.figure(figsize=(20, 20))
            gs = GridSpec(
                2, 1,
                figure=fig,
                height_ratios=[img_top.height, img_bottom.height],
                hspace=-0.05
            )

            # Top plot (a) - summary heatmap
            ax_top = fig.add_subplot(gs[0, 0])
            ax_top.imshow(img_top)
            ax_top.axis("off")
            ax_top.text(
                -0.02, 1.02, "a",
                transform=ax_top.transAxes,
                fontsize=36,
                fontweight="bold",
                va="bottom",
                ha="left",
                fontfamily="sans-serif"
            )

            # Bottom plot (b) - all moieties heatmap
            ax_bottom = fig.add_subplot(gs[1, 0])
            ax_bottom.imshow(img_bottom)
            ax_bottom.axis("off")
            ax_bottom.text(
                -0.02, 1.02, "b",
                transform=ax_bottom.transAxes,
                fontsize=36,
                fontweight="bold",
                va="bottom",
                ha="left",
                fontfamily="sans-serif"
            )

            # Save publication figure
            output_path = os.path.join(
                MOIETY_HEATMAPS_SUMMARY_DIR, f"publication_stacked_{performance}.png"
            )
            plt.savefig(
                output_path,
                bbox_inches="tight",
                dpi=600,
                facecolor="white"
            )
            plt.close(fig)
            print(f"Saved publication figure: {output_path}")


def main():
    analyzer = MoietyEncoderAttentionHeatmaps()
    print("+" * 50)
    print("POST-PROCESSING\nMoeity Encoder Attention Heatmaps")
    print("+" * 50)

    # Generate multihead attention heatmaps for the specified SMILES dictionary
    print("\nGenerating MULTI-HEAD attention heatmaps for specified molecules...")
    print(f"Molecules to analyze: {list(SMILES_TO_ANALYZE.keys())}")
    analyzer.visualize_multihead_attention_combined(SMILES_TO_ANALYZE)

    # Optional: Also generate summary heatmaps
    print("\nAnalyzing extreme LCE formulations for summary heatmaps...\n")
    results_df = analyzer.analyze_extreme_lce_formulations(
        n_top=summary_best_molecules, n_bottom=summary_worst_molecules
    )

    if not results_df.empty:
        analyzer.create_summary_heatmaps(
            results_df, excluded_moieties=EXCLUDED_MOIETIES
        )
        print("Generating summary heatmaps for filtered moieties...\n")
        analyzer.create_all_moieties_summary_heatmaps(results_df, EXCLUDED_MOIETIES)

        # Create publication-ready stacked figures
        print("Generating publication-ready stacked figures...\n")
        analyzer.create_stacked_publication_figures()

    print(f"\nResults saved to: {MOIETY_HEATMAPS_DIR}")
    print(f"Multihead heatmaps saved to: {MOIETY_HEATMAPS_MULTIHEAD_DIR}")
    print(f"Publication figures saved to: {MOIETY_HEATMAPS_SUMMARY_DIR}")


if __name__ == "__main__":
    main()
