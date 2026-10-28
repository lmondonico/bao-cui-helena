"""
Simple analyzer to visualize self-attention patterns for custom SMILES molecules.
Shows local self-attention scores from the moiety encoder for selected moieties.
All moieties in the image share the same color scale for comparison.
"""

########################
## ALL OPTIONS - each is a dict with prefix, smiles, and moieties

ALL_OPTIONS = [
    {
        "prefix": "phosphates",
        "smiles": ["COP(=O)(OC)OC", "CCOP(=O)(OCC)OCC", "CCCOP(=O)(OCCC)OCCC"],
        "moieties": ["phosphoryl", "ether"],
    },
    {
        "prefix": "sulfonamides",
        "smiles": ["FS(=O)(=O)N(C)C", "FS(=O)(=O)N(CC)CC", "FS(=O)(=O)N(CCC)CCC"],
        "moieties": ["trivalent_nitrogen"],
    },
    {
        "prefix": "sulfites",
        "smiles": ["CCOS(C)(=O)=O", "CCCOS(C)(=O)=O", "CCCCOS(C)(=O)=O"],
        "moieties": ["sulfonyl"],
    },
    {
        "prefix": "amides",
        "smiles": ["CC(=O)N(C)C", "CC(=O)N(CC)CC", "CC(=O)N(CCC)CCC"],
        "moieties": ["amide", "carbonyl"],
    },
    {
        "prefix": "fluorinated_carbonates",
        "smiles": ["FC(F)OC(=O)OCC", "FC(F)(F)OC(=O)OCC", "FC(F)(F)C(F)(F)OC(=O)OCC"],
        "moieties": ["carbonate_ester", "fluoro"],
    },
    {
        "prefix": "siloxanes",
        "smiles": [
            "C[Si](C)(C)O[Si](C)(C)C",
            "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C",
            "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C",
        ],
        "moieties": ["silyl"],
    },
    {
        "prefix": "dioxolanes",
        "smiles": ["C1COCO1", "CC1OCOC1", "CC1OCOC1C"],
        "moieties": ["ether"],
    },
    {
        "prefix": "carbonates",
        "smiles": [
            "COC(=O)OC",
            "CCOC(=O)OC",
            "O=C(OCC)OCC",
            "C1COC(=O)O1",
            "CC1COC(=O)O1",
        ],
        "moieties": ["carbonate_ester"],
    },
    {
        "prefix": "fluorinated_ethers",
        "smiles": [
            "CCOCCOCC",
            "C(F)COCCOCC",
            "C(F)COCCOCC(F)",
            "C(F)COCCOCC(F)(F)",
            "FC(F)(F)COCCOCC",
            "FC(F)COCCOCC(F)(F)",
            "FC(F)COCCOCC(F)(F)(F)",
            "FC(F)(F)COCCOCC(F)(F)(F)",
        ],
        "moieties": ["ether"],
    },
    {
        "prefix": "fluorinated_dec",
        "smiles": [
            "O=C(OCC)OCC",
            "O=C(OCC(F)(F)(F))OCC",
            "O=C(OCC(F)(F)(F))OCC(F)(F)(F)",
        ],
        "moieties": ["carbonate_ester"],
    },
    {
        "prefix": "acyclic_ethers",
        "smiles": [
            "COCOC",  # Dimethoxymethane
            "CCOCOCC",  # Diethoxymethane
            "CCCOCOCCC",  # Dipropoxymethane
            "CCCCOCOCCCC",  # Dibutoxymethane
            "COCCOC",  # 1,2-Dimethoxyethane (DME)
            "COCC(C)OC",  # 1,2-Dimethoxypropane
            "CCOCCOCC",  # 1,2-Diethoxyethane (DEE)
            "CCOCC(C)OCC",  # 1,2-Diethoxypropane
            "CC(C)OCCOC(C)C",  # 1,2-Diisopropoxyethane
        ],
        "moieties": ["ether"],
    },
    {
        "prefix": "rok_choi_ethers",
        "smiles": [
            "COCCOCC",
            "COCCOCC(F)",
            "COCCOCC(F)(F)",
            "COCCOCC(F)(F)(F)",
        ],
        "moieties": ["ether"],
    },
]

SALT = "LiFSI"
########################

import os
import sys

# Add project root to path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from tasks.postprocess.postprocess_utils.postprocess_utils import (
    MOIETY_CUSTOM_SMILES_DIR,
)
from tasks.postprocess.moiety_analysis.moiety_encod_att_heatmaps import (
    MoietyEncoderAttentionHeatmaps,
)

import warnings

warnings.filterwarnings("ignore")

os.makedirs(MOIETY_CUSTOM_SMILES_DIR, exist_ok=True)


class SimpleAttentionAnalyzer:
    def __init__(self):
        self.heatmap_analyzer = MoietyEncoderAttentionHeatmaps()
        self.local_cmap = sns.color_palette("YlOrRd", as_cmap=True)

    def analyze_molecules_vertically(
        self, smiles_list, output_dir, show_moieties=None, image_prefix=None
    ):
        """
        Process molecules and visualize self-attention for selected moieties.
        All moieties share the same color scale within the image.
        """
        print(
            f"\nProcessing molecules{' (highlighted moieties: ' + ', '.join(show_moieties) + ')' if show_moieties else ''}..."
        )

        molecules_data = []

        # First pass: collect all self-attention scores for the selected moieties
        all_self_attention_scores = []

        for smiles in smiles_list:
            try:
                local_attn, global_attn, moiety_ids, clique_atom_ids = (
                    self.heatmap_analyzer.get_attention_weights(smiles, 1.0)
                )
                if local_attn is None:
                    print(f"Could not process {smiles}, skipping")
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                if mol.GetNumConformers() == 0:
                    rdDepictor.Compute2DCoords(mol)

                moiety_names = [
                    self.heatmap_analyzer.tokenizer.key_moiety_map.get(mid)
                    for mid in moiety_ids
                ]

                # Get self-attention scores (diagonal) for selected moieties
                # Group by moiety name and average scores for same moiety type
                from collections import defaultdict

                moiety_type_scores = defaultdict(list)
                moiety_type_indices = defaultdict(list)

                for idx, name in enumerate(moiety_names):
                    if show_moieties is None or name in show_moieties:
                        # Self-attention = diagonal element
                        self_attn = float(local_attn[idx, idx])
                        moiety_type_scores[name].append(self_attn)
                        moiety_type_indices[name].append(idx)

                # Average self-attention for each moiety type
                moiety_type_avg = {
                    name: np.mean(scores) for name, scores in moiety_type_scores.items()
                }

                # Assign averaged score to all indices of that moiety type
                self_attn_scores = {}
                for name, indices in moiety_type_indices.items():
                    avg_score = moiety_type_avg[name]
                    for idx in indices:
                        self_attn_scores[idx] = avg_score
                    all_self_attention_scores.append(avg_score)

                molecules_data.append(
                    {
                        "smiles": smiles,
                        "mol": mol,
                        "moiety_ids": moiety_ids,
                        "moiety_names": moiety_names,
                        "clique_atom_ids": clique_atom_ids,
                        "self_attn_scores": self_attn_scores,
                    }
                )

            except Exception as e:
                print(f"Error processing {smiles}: {str(e)}")
                continue

        if not molecules_data or not all_self_attention_scores:
            print("No valid molecules to display")
            return

        # Compute global min/max for color scale across all moieties in this image
        global_min = min(all_self_attention_scores)
        global_max = max(all_self_attention_scores)
        score_range = global_max - global_min if global_max > global_min else 1.0

        # Set up figure layout
        n_mols = len(molecules_data)
        col1_mols = min(4, n_mols)
        col2_mols = min(5, max(0, n_mols - col1_mols))
        max_rows = max(col1_mols, col2_mols)

        fig, axes = plt.subplots(max_rows, 2, figsize=(8, 3.5 * max_rows))
        axes = np.atleast_2d(axes)

        axes[0, 0].set_title("Local Self-Attention (Col 1)", fontsize=12)
        axes[0, 1].set_title("Local Self-Attention (Col 2)", fontsize=12)

        # Second pass: create visualizations with unified color scale
        for idx, data in enumerate(molecules_data):
            if idx < col1_mols:
                row = idx
                col = 0
            else:
                row = idx - col1_mols
                col = 1

            mol = data["mol"]
            clique_atom_ids = data["clique_atom_ids"]
            self_attn_scores = data["self_attn_scores"]

            # Create atom colors using global scale
            atom_colors = {}
            highlight_atoms = set()

            for moiety_idx, self_attn in self_attn_scores.items():
                # Normalize using global min/max
                normalized = (self_attn - global_min) / score_range
                rgb_color = self.local_cmap(normalized)[:3]

                for atom_idx in clique_atom_ids[moiety_idx]:
                    atom_colors[atom_idx] = rgb_color
                    highlight_atoms.add(atom_idx)

            # Get bonds to highlight
            highlight_bonds = []
            bond_colors = {}
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_idx = bond.GetIdx()

                if begin_idx in highlight_atoms or end_idx in highlight_atoms:
                    highlight_bonds.append(bond_idx)
                    if begin_idx in atom_colors:
                        bond_colors[bond_idx] = atom_colors[begin_idx]
                    elif end_idx in atom_colors:
                        bond_colors[bond_idx] = atom_colors[end_idx]

            # Draw molecule
            drawer = rdMolDraw2D.MolDraw2DCairo(600, 300)
            drawer.drawOptions().bondLineWidth = 2
            drawer.drawOptions().continuousHighlight = True
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(atom_colors.keys()),
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()

            img = Image.open(BytesIO(drawer.GetDrawingText()))
            ax = axes[row, col]
            ax.imshow(np.array(img))
            ax.set_title(data["smiles"], fontsize=9)
            ax.axis("off")

        # Hide unused axes
        for row in range(max_rows):
            if row >= col1_mols:
                axes[row, 0].axis("off")
                axes[row, 0].set_visible(False)
            if row >= col2_mols:
                axes[row, 1].axis("off")
                axes[row, 1].set_visible(False)

        # Add colorbar with actual min/max values
        cbar_ax = fig.add_axes([0.17, 0.05, 0.7, 0.02])
        norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd"),
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.set_label(
            f"Local Self-Attention (min={global_min:.3f}, max={global_max:.3f})"
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        # Generate filename
        prefix = image_prefix if image_prefix else "custom_smiles"
        moiety_suffix = "_" + "_".join(show_moieties) if show_moieties else "_all"
        filename = f"{prefix}{moiety_suffix}_local_self_attention.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Results saved to {output_dir}/{filename}")


def main():
    print()
    print("+" * 50)
    print("POST-PROCESSING\nMoiety Self-Attention Analysis for custom SMILES")
    print("+" * 50)

    analyzer = SimpleAttentionAnalyzer()

    for option in ALL_OPTIONS:
        prefix = option["prefix"]
        smiles_list = option["smiles"]
        moieties = option["moieties"]

        print(f"\n{'=' * 50}")
        print(f"Processing: {prefix}")
        print(f"SMILES count: {len(smiles_list)}")
        print(f"Moieties: {moieties}")
        print("=" * 50)

        # Image 1: All moieties colored
        analyzer.analyze_molecules_vertically(
            smiles_list,
            MOIETY_CUSTOM_SMILES_DIR,
            show_moieties=None,
            image_prefix=prefix,
        )

        # Image 2: Only selected moieties colored
        analyzer.analyze_molecules_vertically(
            smiles_list,
            MOIETY_CUSTOM_SMILES_DIR,
            show_moieties=moieties,
            image_prefix=prefix,
        )

    print(f"\nAll analyses saved to: {MOIETY_CUSTOM_SMILES_DIR}\n")


if __name__ == "__main__":
    main()
