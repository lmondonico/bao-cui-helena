"""
Ternary Formulation Visualization.
Creates visualizations for ternary solvent formulations from pkl intermediates.
Requires generate_intermediates.py to be run first.
"""

########################
## Parameters
SALT_NAMES = {
    "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F": "FSI",
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F": "TFSI",
    "[Li+].F[P-](F)(F)(F)(F)F": "PF6",
    "[Li+].[B-]1(OC(=O)C(=O)O1)(F)F": "DFOB",
}
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

import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image

from tasks.postprocess.postprocess_utils.postprocess_utils import (
    TERNARY_INTERMEDIATES_DIR,
    TERNARY_ANALYSIS_DIR,
)

import warnings

warnings.filterwarnings("ignore")

# Output directories
SINGLE_SALT_DIR = os.path.join(TERNARY_ANALYSIS_DIR, "tern_single_salt")
BINARY_SALT_DIR = os.path.join(TERNARY_ANALYSIS_DIR, "tern_binary_salt")
STATS_DIR = os.path.join(TERNARY_ANALYSIS_DIR, "statistics")

os.makedirs(SINGLE_SALT_DIR, exist_ok=True)
os.makedirs(BINARY_SALT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


class TernaryFormulationVisualizer:
    """
    Visualizes ternary solvent formulation moiety patterns from pkl files.
    """

    def __init__(self):
        self.salt_combinations = {}
        self.all_salts = []
        self.moiety_triplet_data = {}
        self._load_intermediates()

    def _get_salt_name(self, salt_smiles):
        """Convert salt SMILES to a short name if available."""
        return SALT_NAMES.get(
            salt_smiles,
            salt_smiles[:15] + "..." if len(salt_smiles) > 15 else salt_smiles,
        )

    def _load_intermediates(self):
        """Load intermediate data from pickle files."""
        salt_comb_path = os.path.join(
            TERNARY_INTERMEDIATES_DIR, "salt_combinations.pkl"
        )
        all_salts_path = os.path.join(TERNARY_INTERMEDIATES_DIR, "all_salts.pkl")
        moiety_triplet_path = os.path.join(
            TERNARY_INTERMEDIATES_DIR, "moiety_triplet_data.pkl"
        )

        if not os.path.exists(salt_comb_path):
            print(f"ERROR: Intermediate files not found in {TERNARY_INTERMEDIATES_DIR}")
            print("Please run generate_intermediates.py first.")
            sys.exit(1)

        with open(salt_comb_path, "rb") as f:
            self.salt_combinations = pickle.load(f)
        with open(all_salts_path, "rb") as f:
            self.all_salts = pickle.load(f)
        with open(moiety_triplet_path, "rb") as f:
            self.moiety_triplet_data = pickle.load(f)

        print(f"Loaded intermediate data from {TERNARY_INTERMEDIATES_DIR}")
        print(f"  - {len(self.salt_combinations)} salt combinations")
        print(f"  - {len(self.moiety_triplet_data)} moiety triplet datasets")

    def create_visualizations(self):
        """Create all visualizations from the loaded data."""
        if not self.salt_combinations and not self.moiety_triplet_data:
            print("No data available for visualization.")
            return

        if self.salt_combinations and self.all_salts:
            print("\nCreating salt statistics visualizations...")
            self._create_salt_histograms()

        if self.moiety_triplet_data:
            print("\nAnalyzing moiety triplet occurrences...")
            for combo_key, triplets in tqdm(
                self.moiety_triplet_data.items(), desc="Creating visualizations"
            ):
                if len(triplets) < 5:
                    continue

                salt_combo, salt_key, display_key = combo_key
                if salt_combo not in ["single", "binary"]:
                    continue

                self._create_moiety_distribution_plot(triplets, salt_combo, display_key)
                self._create_publication_figures(triplets, salt_combo, display_key)

        print("\nAll visualizations created successfully!")

    def _create_salt_histograms(self):
        """Create histograms for salt statistics."""
        combo_counts = {}
        for (combo, key), indices in self.salt_combinations.items():
            if combo not in ["single", "binary"]:
                continue

            if combo == "single":
                display_key = self._get_salt_name(key)
            else:
                parts = key.split("_")
                display_key = (
                    f"{self._get_salt_name(parts[0])} + {self._get_salt_name(parts[1])}"
                )
            combo_counts[display_key] = len(indices)

        if not combo_counts:
            return

        plt.figure(figsize=(12, 8))
        sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]
        labels = [combo for combo, _ in sorted_combos]
        values = [count for _, count in sorted_combos]

        plt.title("Salt Combinations in Ternary Formulations", fontsize=18)
        bars = plt.bar(range(len(labels)), values, color="steelblue")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
        plt.ylabel("Number of Formulations", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height):,}",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(STATS_DIR, "salt_combinations_histogram.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_moiety_distribution_plot(self, triplets, salt_combo, display_key):
        """Create bar plots showing moiety distribution in each position."""
        all_moieties1 = []
        all_moieties2 = []
        all_moieties3 = []

        for triplet in triplets:
            all_moieties1.extend(triplet["moieties1"])
            all_moieties2.extend(triplet["moieties2"])
            all_moieties3.extend(triplet["moieties3"])

        moiety_counts = [
            Counter(all_moieties1),
            Counter(all_moieties2),
            Counter(all_moieties3),
        ]
        labels = ["Solvent 1", "Solvent 2", "Solvent 3"]

        plt.figure(figsize=(15, 10))

        for idx, (counts, label) in enumerate(zip(moiety_counts, labels)):
            ax = plt.subplot(3, 1, idx + 1)
            top_moieties = counts.most_common(12)

            if not top_moieties:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
                continue

            moiety_names = [m[0] for m in top_moieties]
            moiety_freqs = [m[1] for m in top_moieties]

            sorted_indices = np.argsort(moiety_freqs)
            sorted_names = [moiety_names[i] for i in sorted_indices]
            sorted_freqs = [moiety_freqs[i] for i in sorted_indices]

            bars = ax.barh(
                sorted_names, sorted_freqs, color=plt.cm.Blues(0.5 + idx / 6.0)
            )
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + max(sorted_freqs) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    va="center",
                )

            ax.set_title(f"Top Moieties in {label}", fontsize=12)
            ax.set_xlabel("Occurrence Count", fontsize=10)
            ax.grid(axis="x", linestyle="--", alpha=0.7)

        title = f"Moiety Distribution in Ternary Solvents with {display_key}"
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        safe_key = "".join(c if c.isalnum() else "" for c in str(display_key))
        save_dir = SINGLE_SALT_DIR if salt_combo == "single" else BINARY_SALT_DIR
        plt.savefig(
            os.path.join(save_dir, f"moiety_dist_{salt_combo}_{safe_key}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_publication_figures(self, triplets, salt_combo, display_key):
        """Create two publication-ready figures with stacked plots."""
        # Build triplet counts
        triplet_counts = defaultdict(int)
        for triplet in triplets:
            for m1 in triplet["moieties1"]:
                for m2 in triplet["moieties2"]:
                    for m3 in triplet["moieties3"]:
                        triplet_counts[(m1, m2, m3)] += 1

        if not triplet_counts:
            return

        # Save patterns to CSV
        top_patterns = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]
        data = [[m1, m2, m3, count] for (m1, m2, m3), count in top_patterns]
        df = pd.DataFrame(
            data,
            columns=[
                "Moiety in Solvent 1",
                "Moiety in Solvent 2",
                "Moiety in Solvent 3",
                "Count",
            ],
        )
        safe_key = "".join(c if c.isalnum() else "" for c in str(display_key))
        save_dir = SINGLE_SALT_DIR if salt_combo == "single" else BINARY_SALT_DIR
        df.to_csv(
            os.path.join(save_dir, f"moiety_patterns_{salt_combo}_{safe_key}.csv"),
            index=False,
        )

        # Prepare co-occurrence data for all three pairs
        heatmap_data = {}
        for label1, label2, idx1, idx2 in [
            ("1", "2", 0, 1),
            ("1", "3", 0, 2),
            ("2", "3", 1, 2),
        ]:
            cooccur = defaultdict(int)
            for (m1, m2, m3), count in triplet_counts.items():
                moieties = [m1, m2, m3]
                cooccur[(moieties[idx1], moieties[idx2])] += count

            unique_m1 = sorted(set(pair[0] for pair in cooccur.keys()))[:15]
            unique_m2 = sorted(set(pair[1] for pair in cooccur.keys()))[:15]

            if not unique_m1 or not unique_m2:
                continue

            matrix = np.zeros((len(unique_m1), len(unique_m2)))
            for i, m1_val in enumerate(unique_m1):
                for j, m2_val in enumerate(unique_m2):
                    matrix[i, j] = cooccur.get((m1_val, m2_val), 0)

            heatmap_data[f"{label1}_{label2}"] = {
                "matrix": matrix,
                "unique_m1": unique_m1,
                "unique_m2": unique_m2,
                "label1": label1,
                "label2": label2,
            }

        # Generate temporary heatmap images (without titles)
        heatmap_images = {}
        for key, data_dict in heatmap_data.items():
            plt.figure(figsize=(12, 10))
            mask = data_dict["matrix"] == 0
            sns.heatmap(
                data_dict["matrix"],
                xticklabels=data_dict["unique_m2"],
                yticklabels=data_dict["unique_m1"],
                cmap="BuGn",
                mask=mask,
                linewidths=0,
            )
            plt.ylabel(f"Moieties in Solvent {data_dict['label1']}", fontsize=12)
            plt.xlabel(f"Moieties in Solvent {data_dict['label2']}", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.yticks(fontsize=12, rotation=0)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=600, bbox_inches="tight")
            buf.seek(0)
            heatmap_images[key] = np.array(Image.open(buf))
            plt.close()

        # Generate 3D plot to buffer
        plot_3d_img = self._create_3d_scatter_plot_to_buffer(
            triplets, salt_combo, display_key
        )

        # Figure 1: Stack heatmaps 1-2 (a) and 1-3 (b)
        if "1_2" in heatmap_images and "1_3" in heatmap_images:
            fig, axes = plt.subplots(2, 1, figsize=(14, 20))

            axes[0].imshow(heatmap_images["1_2"])
            axes[0].axis("off")

            axes[1].imshow(heatmap_images["1_3"])
            axes[1].axis("off")

            plt.tight_layout()

            # Add letters with fixed horizontal position
            fig.text(
                0.02, 0.98, "a", fontsize=24, fontweight="bold", ha="left", va="top"
            )
            fig.text(
                0.02, 0.50, "b", fontsize=24, fontweight="bold", ha="left", va="top"
            )

            plt.savefig(
                os.path.join(save_dir, f"{safe_key}_1.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Figure 2: Stack heatmap 2-3 (c) and 3D plot (d)
        if "2_3" in heatmap_images and plot_3d_img is not None:
            fig, axes = plt.subplots(2, 1, figsize=(14, 20))

            axes[0].imshow(heatmap_images["2_3"])
            axes[0].axis("off")

            axes[1].imshow(plot_3d_img)
            axes[1].axis("off")

            plt.tight_layout()

            # Add letters with fixed horizontal position
            fig.text(
                0.02, 0.98, "c", fontsize=24, fontweight="bold", ha="left", va="top"
            )
            fig.text(
                0.02, 0.50, "d", fontsize=24, fontweight="bold", ha="left", va="top"
            )

            plt.savefig(
                os.path.join(save_dir, f"{safe_key}_2.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _create_3d_scatter_plot_to_buffer(self, triplets, salt_combo, display_key):
        """Create 3D scatter plot and return as image array (no file output)."""
        # Build triplet counts
        triplet_counts = defaultdict(int)
        for triplet in triplets:
            for m1 in triplet["moieties1"]:
                for m2 in triplet["moieties2"]:
                    for m3 in triplet["moieties3"]:
                        triplet_counts[(m1, m2, m3)] += 1

        # Get top 50 patterns
        top_patterns = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)[
            :50
        ]

        if len(top_patterns) < 3:
            return None

        # Extract unique moieties for each position
        unique_m1 = sorted(set(triplet[0] for triplet, _ in top_patterns))
        unique_m2 = sorted(set(triplet[1] for triplet, _ in top_patterns))
        unique_m3 = sorted(set(triplet[2] for triplet, _ in top_patterns))

        # Create mappings
        m1_to_idx = {m: i for i, m in enumerate(unique_m1)}
        m2_to_idx = {m: i for i, m in enumerate(unique_m2)}
        m3_to_idx = {m: i for i, m in enumerate(unique_m3)}

        # Filter and extract data
        filtered_patterns = [
            ((m1, m2, m3), count)
            for (m1, m2, m3), count in top_patterns
            if m1 in m1_to_idx and m2 in m2_to_idx and m3 in m3_to_idx
        ]

        if not filtered_patterns:
            return None

        counts = np.array([count for _, count in filtered_patterns])
        x = np.array([m1_to_idx[m1] for (m1, m2, m3), _ in filtered_patterns])
        y = np.array([m2_to_idx[m2] for (m1, m2, m3), _ in filtered_patterns])
        z = np.array([m3_to_idx[m3] for (m1, m2, m3), _ in filtered_patterns])

        # Create figure
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Plot scatter
        scatter = ax.scatter(x, y, z, c=counts, cmap="viridis", s=50, alpha=0.85)

        # Set axes limits
        ax.set_xlim(-0.5, len(unique_m1) - 0.5)
        ax.set_ylim(-0.5, len(unique_m2) - 0.5)
        ax.set_zlim(-0.5, len(unique_m3) - 0.5)

        # Set ticks
        ax.set_xticks(range(len(unique_m1)))
        ax.set_yticks(range(len(unique_m2)))
        ax.set_zticks(range(len(unique_m3)))

        # Tick label styling
        ticksize = 10
        ax.set_xticklabels(
            unique_m1, rotation=0, ha="center", va="top", fontsize=ticksize
        )
        ax.set_yticklabels(
            unique_m2, rotation=0, ha="left", va="center", fontsize=ticksize
        )
        ax.set_zticklabels(
            unique_m3, rotation=0, ha="left", va="center", fontsize=ticksize
        )

        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", pad=1, labelright=False, labelleft=True)
        ax.tick_params(axis="z", pad=1, labelright=False, labelleft=True)

        # Set axis labels
        label_fontsize = 14
        max_y_tick_len = max(len(label) for label in unique_m2) if unique_m2 else 1
        max_z_tick_len = max(len(label) for label in unique_m3) if unique_m3 else 1

        ax.set_xlabel(
            "Solvent 1 Moieties", fontsize=label_fontsize, labelpad=max_y_tick_len * 2
        )
        ax.set_ylabel(
            "Solvent 2 Moieties", fontsize=label_fontsize, labelpad=max_y_tick_len * 4
        )
        ax.set_zlabel(
            "Solvent 3 Moieties", fontsize=label_fontsize, labelpad=max_z_tick_len * 4
        )

        # Add colorbar
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cbar_height = bbox.height * 0.07
        cbar_ax = fig.add_axes([0.88, 0.15, 0.01, cbar_height])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Occurrence Count", fontsize=14)

        # Save to buffer and return as array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        plt.close()

        return img_array


def main():
    print("\n" + "+" * 50)
    print("TERNARY FORMULATION VISUALIZATION")
    print("+" * 50 + "\n")

    visualizer = TernaryFormulationVisualizer()
    visualizer.create_visualizations()

    print(f"\nVisualization complete.")
    print(f"Results saved to: {TERNARY_ANALYSIS_DIR}")
    print(f"Single salt results in: {SINGLE_SALT_DIR}")
    print(f"Binary salt results in: {BINARY_SALT_DIR}\n")


if __name__ == "__main__":
    main()
