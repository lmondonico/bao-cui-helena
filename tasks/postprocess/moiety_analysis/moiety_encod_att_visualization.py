"""
Visualization of moiety attention scores over molecular graphs
Shows local and global attention patterns for extreme LCE molecules
"""

########################
## Parameters
N_EXTREME_TOP_MOLS = 7
N_EXTREME_BOTTOM_MOLS = 18
VISUALIZATION_DPI = 900
MOL_IMG_SIZE = (800, 500)
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm

from tasks.postprocess.postprocess_utils.postprocess_utils import (
    MoeityEncoderAttentionUtils,
    MOIETY_VISUALIZ_DIR,
    canonicalize_smiles,
)
from tasks.postprocess.postprocess_utils.excluded_moieties import (
    EXCLUDED_VISUALIZATION_MOIETIES,
    EXCLUDED_VISUALIZATION_MOLECULES,
)

import warnings

warnings.filterwarnings("ignore")


class MoietyAttentionVisualizer(MoeityEncoderAttentionUtils):
    def __init__(self):
        super().__init__()
        self.show_device_used = True

        os.makedirs(MOIETY_VISUALIZ_DIR, exist_ok=True)

        # Create separate directories for local and global attention
        self.local_dir = os.path.join(MOIETY_VISUALIZ_DIR, "local_attention")
        self.global_dir = os.path.join(MOIETY_VISUALIZ_DIR, "global_attention")
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        # Define colormaps similar to heatmaps
        self.local_cmap = sns.color_palette("YlOrRd", as_cmap=True)
        self.global_cmap = sns.color_palette("Blues", as_cmap=True)

    def _create_atom_colors(
        self, attention_scores, clique_atom_ids, moiety_ids, attention_type="local"
    ):
        """
        Create smooth color gradients for atoms based on attention scores.
        NOTE: If an attention score is provided as an array, its mean is used.
        (That is, each moiety's attention is computed as the mean of its attention values.)
        """
        atom_colors = {}
        valid_indices = []
        valid_atoms = []
        valid_scores = []

        # Filter moieties
        for idx, (moiety_id, atoms) in enumerate(zip(moiety_ids, clique_atom_ids)):
            moiety_name = self.tokenizer.key_moiety_map.get(moiety_id)
            if moiety_name not in EXCLUDED_VISUALIZATION_MOIETIES:
                valid_indices.append(idx)
                valid_atoms.extend(atoms)
                valid_scores.append(attention_scores[idx])

        if not valid_scores:
            return {}

        # Flatten the valid_scores and compute the range
        attention_scores = np.array(valid_scores).flatten()
        max_score = np.max(attention_scores)
        min_score = np.min(attention_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        cmap = self.local_cmap if attention_type == "local" else self.global_cmap

        for score_idx, moiety_idx in enumerate(valid_indices):
            # Compute the mean if the score is an array
            score = float(np.array(attention_scores[score_idx]).mean())
            normalized_score = (score - min_score) / score_range
            rgb_color = cmap(normalized_score)[:3]
            for atom_idx in clique_atom_ids[moiety_idx]:
                atom_colors[atom_idx] = rgb_color

        return atom_colors

    def visualize_attention(self, smiles, save_prefix, molar_ratio=1.0):
        """
        Visualize both local and global attention patterns for a molecule.
        The individual molecule image is saved without an embedded colorbar.
        Returns a dict mapping attention_type to a tuple (vmin, vmax) for the valid attention scores.
        """
        result_ranges = {}
        try:
            # Get attention weights and molecule information
            local_attn, global_attn, moiety_ids, clique_atom_ids = (
                self.get_attention_weights(smiles, molar_ratio)
            )

            if local_attn is None or global_attn is None:
                print(f"No attention weights available for {smiles}")
                return None

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Could not parse SMILES: {smiles}")
                return None

            if not mol.GetNumConformers():
                rdDepictor.Compute2DCoords(mol)

            # Process for both local and global attention
            for attention_type, attention_scores, cmap in [
                ("local", local_attn, self.local_cmap),
                ("global", global_attn, self.global_cmap),
            ]:
                atom_colors = self._create_atom_colors(
                    attention_scores, clique_atom_ids, moiety_ids, attention_type
                )
                if not atom_colors:
                    print(f"No valid moieties found for {smiles}")
                    continue

                # Set up drawing (without a colorbar)
                drawer = rdMolDraw2D.MolDraw2DCairo(MOL_IMG_SIZE[0], MOL_IMG_SIZE[1])
                drawer.drawOptions().addAtomIndices = False
                drawer.drawOptions().padding = 0.2
                drawer.drawOptions().bondLineWidth = 2

                drawer.DrawMolecule(
                    mol,
                    highlightAtoms=list(atom_colors.keys()),
                    highlightAtomColors=atom_colors,
                )
                drawer.FinishDrawing()

                output_dir = (
                    self.local_dir if attention_type == "local" else self.global_dir
                )
                base_filename = f"{save_prefix}_attention.png"
                output_path = os.path.join(output_dir, base_filename)
                drawer.WriteDrawingText(output_path)

                # Compute valid attention scores for this attention type
                valid_scores = [
                    float(np.array(score).mean())
                    for idx, score in enumerate(attention_scores)
                ]
                vmin = min(valid_scores)
                vmax = max(valid_scores)
                # Fix identical limits to avoid singular transformation warnings
                if vmin == vmax:
                    vmin = vmin - 0.1
                    vmax = vmax + 0.1

                result_ranges[attention_type] = (vmin, vmax)

        except Exception as e:
            print(f"Error visualizing molecule {smiles}: {str(e)}")
            import traceback

            traceback.print_exc()
        return result_ranges


def main():
    visualizer = MoietyAttentionVisualizer()
    print("+" * 50)
    print("POST-PROCESSING\nMoiety Attention Visualization")
    print("+" * 50)

    # Dictionary to collect (vmin, vmax) for each attention type across molecules
    global_ranges = {"local": [], "global": []}

    # Analyze extreme LCE formulations
    print(
        f"\nAnalyzing {N_EXTREME_TOP_MOLS} highest and {N_EXTREME_BOTTOM_MOLS} lowest LCE formulations..."
    )
    results_df = visualizer.analyze_extreme_lce_formulations(
        n_top=N_EXTREME_TOP_MOLS, n_bottom=N_EXTREME_BOTTOM_MOLS
    )

    if not results_df.empty:
        print("\nGenerating attention visualizations...")
        for performance in ["highest", "lowest"]:
            perf_df = results_df[results_df["performance"] == performance]

            if not perf_df.empty:
                excluded_set = set(
                    canonicalize_smiles(smi) for smi in EXCLUDED_VISUALIZATION_MOLECULES
                )
                perf_df = perf_df[
                    ~perf_df["molecule"].apply(
                        lambda x: canonicalize_smiles(x) in excluded_set
                    )
                ]

                molecules = perf_df.groupby("molecule")["LCE"].mean()
                selected_mols = (
                    molecules.nlargest(N_EXTREME_TOP_MOLS)
                    if performance == "highest"
                    else molecules.nsmallest(N_EXTREME_BOTTOM_MOLS)
                )

                for smiles in tqdm(
                    selected_mols.index, desc=f"Processing {performance} LCE mols"
                ):
                    if canonicalize_smiles(smiles) not in excluded_set:
                        mol_info = perf_df[perf_df["molecule"] == smiles].iloc[0]
                        save_prefix = f"{performance}_lce_{mol_info['Row_ID']}"
                        ret = visualizer.visualize_attention(
                            smiles, save_prefix, molar_ratio=mol_info["molar_ratio"]
                        )
                        if ret is not None:
                            for attn_type, rng in ret.items():
                                global_ranges[attn_type].append(rng)

        # Create a single combined figure with local (a) and global (b) attention stacked
        local_highest = sorted(
            [f for f in os.listdir(visualizer.local_dir) if "highest_lce" in f]
        )
        local_lowest = sorted(
            [f for f in os.listdir(visualizer.local_dir) if "lowest_lce" in f]
        )
        global_highest = sorted(
            [f for f in os.listdir(visualizer.global_dir) if "highest_lce" in f]
        )
        global_lowest = sorted(
            [f for f in os.listdir(visualizer.global_dir) if "lowest_lce" in f]
        )

        n_cols = max(
            len(local_highest), len(local_lowest),
            len(global_highest), len(global_lowest)
        )
        if n_cols == 0:
            print("No images found to create composite figure.")
        else:
            # 4 rows: local highest, local lowest, global highest, global lowest
            fig, axes = plt.subplots(4, n_cols + 1, figsize=(4 * n_cols + 1.5, 12),
                                     gridspec_kw={'width_ratios': [0.15] + [1] * n_cols})

            # Helper to plot a row
            def plot_row(row_idx, files, directory, row_label):
                # Panel label in leftmost column
                axes[row_idx, 0].axis("off")
                for j in range(n_cols):
                    ax = axes[row_idx, j + 1]
                    if j < len(files):
                        img = plt.imread(os.path.join(directory, files[j]))
                        ax.imshow(img)
                    ax.axis("off")

            # Row 0: Local - Highest LCE
            plot_row(0, local_highest, visualizer.local_dir, "Highest LCE")
            # Row 1: Local - Lowest LCE
            plot_row(1, local_lowest, visualizer.local_dir, "Lowest LCE")
            # Row 2: Global - Highest LCE
            plot_row(2, global_highest, visualizer.global_dir, "Highest LCE")
            # Row 3: Global - Lowest LCE
            plot_row(3, global_lowest, visualizer.global_dir, "Lowest LCE")

            # Add panel labels a and b at top left of each section
            # a for local attention (row 0)
            axes[0, 0].text(0.0, 1.0, "a", fontsize=36, fontweight="bold",
                           ha="left", va="top", transform=axes[0, 0].transAxes)
            # b for global attention (row 2)
            axes[2, 0].text(0.0, 1.0, "b", fontsize=36, fontweight="bold",
                           ha="left", va="top", transform=axes[2, 0].transAxes)

            # Add row labels on the right side
            for row_idx, label in [(0, "Highest LCE"), (1, "Lowest LCE"),
                                   (2, "Highest LCE"), (3, "Lowest LCE")]:
                axes[row_idx, -1].annotate(label, xy=(1.02, 0.5),
                                           xycoords="axes fraction",
                                           fontsize=16, fontweight="bold",
                                           ha="left", va="center",
                                           rotation=-90)

            # Adjust layout
            fig.subplots_adjust(
                left=0.02,
                bottom=0.08,
                right=0.85,
                top=0.92,
                wspace=0.02,
                hspace=0.08,
            )

            # Add colorbars for local and global attention
            # Local colorbar (for panel a)
            if global_ranges["local"]:
                local_vmin = min(x for (x, y) in global_ranges["local"])
                local_vmax = max(y for (x, y) in global_ranges["local"])
            else:
                local_vmin, local_vmax = 0, 1
            local_norm = Normalize(vmin=local_vmin, vmax=local_vmax)
            local_sm = ScalarMappable(cmap=visualizer.local_cmap, norm=local_norm)
            local_sm.set_array([])
            local_cbar_ax = fig.add_axes([0.88, 0.52, 0.02, 0.38])
            local_cbar = fig.colorbar(local_sm, cax=local_cbar_ax)
            local_cbar.ax.tick_params(labelsize=14)

            # Global colorbar (for panel b)
            if global_ranges["global"]:
                global_vmin = min(x for (x, y) in global_ranges["global"])
                global_vmax = max(y for (x, y) in global_ranges["global"])
            else:
                global_vmin, global_vmax = 0, 1
            global_norm = Normalize(vmin=global_vmin, vmax=global_vmax)
            global_sm = ScalarMappable(cmap=visualizer.global_cmap, norm=global_norm)
            global_sm.set_array([])
            global_cbar_ax = fig.add_axes([0.88, 0.08, 0.02, 0.38])
            global_cbar = fig.colorbar(global_sm, cax=global_cbar_ax)
            global_cbar.ax.tick_params(labelsize=14)

            # Add section titles (larger font, closer to images)
            fig.text(0.44, 0.91, "Local Attention", fontsize=20, fontweight="bold", ha="center")
            fig.text(0.44, 0.46, "Global Attention", fontsize=20, fontweight="bold", ha="center")

            composite_path = os.path.join(
                MOIETY_VISUALIZ_DIR, "moiety_attention_composite.png"
            )
            plt.savefig(composite_path, dpi=VISUALIZATION_DPI, bbox_inches="tight")
            plt.close()
            print(f"\nCombined figure saved to: {composite_path}")

    print(f"\nVisualizations saved to:")
    print(f"Local attention: {visualizer.local_dir}")
    print(f"Global attention: {visualizer.global_dir}")


if __name__ == "__main__":
    main()
