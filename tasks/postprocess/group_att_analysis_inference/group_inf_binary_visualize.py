"""
Binary Formulation Visualization.
Creates visualizations for binary solvent formulations from pkl intermediates.
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
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from tasks.postprocess.postprocess_utils.postprocess_utils import (
    BINARY_INTERMEDIATES_DIR,
    BINARY_ANALYSIS_DIR,
)

import warnings
warnings.filterwarnings("ignore")

# Output directories
SINGLE_SALT_DIR = os.path.join(BINARY_ANALYSIS_DIR, "bin_single_salt")
BINARY_SALT_DIR = os.path.join(BINARY_ANALYSIS_DIR, "bin_binary_salt")
STATS_DIR = os.path.join(BINARY_ANALYSIS_DIR, "salt_statistics")

os.makedirs(SINGLE_SALT_DIR, exist_ok=True)
os.makedirs(BINARY_SALT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


class BinaryFormulationVisualizer:
    """
    Visualizes binary solvent formulation moiety patterns from pkl files.
    """

    def __init__(self):
        self.salt_combinations = {}
        self.all_salts = []
        self.moiety_pair_data = {}
        self._load_intermediates()

    def _get_salt_name(self, salt_smiles):
        """Convert salt SMILES to a short name if available."""
        return SALT_NAMES.get(salt_smiles, salt_smiles[:15] + "..." if len(salt_smiles) > 15 else salt_smiles)

    def _load_intermediates(self):
        """Load intermediate data from pickle files."""
        salt_comb_path = os.path.join(BINARY_INTERMEDIATES_DIR, "salt_combinations.pkl")
        all_salts_path = os.path.join(BINARY_INTERMEDIATES_DIR, "all_salts.pkl")
        moiety_pair_path = os.path.join(BINARY_INTERMEDIATES_DIR, "moiety_pair_data.pkl")

        if not os.path.exists(salt_comb_path):
            print(f"ERROR: Intermediate files not found in {BINARY_INTERMEDIATES_DIR}")
            print("Please run generate_intermediates.py first.")
            sys.exit(1)

        with open(salt_comb_path, "rb") as f:
            self.salt_combinations = pickle.load(f)
        with open(all_salts_path, "rb") as f:
            self.all_salts = pickle.load(f)
        with open(moiety_pair_path, "rb") as f:
            self.moiety_pair_data = pickle.load(f)

        print(f"Loaded intermediate data from {BINARY_INTERMEDIATES_DIR}")
        print(f"  - {len(self.salt_combinations)} salt combinations")
        print(f"  - {len(self.moiety_pair_data)} moiety pair datasets")

    def create_visualizations(self):
        """Create all visualizations from the loaded data."""
        if not self.salt_combinations or not self.moiety_pair_data:
            print("No data available for visualization.")
            return

        # Create salt statistics
        combo_counts = {}
        for (combo, key), indices in self.salt_combinations.items():
            if combo == "single":
                display_key = self._get_salt_name(key)
            elif combo == "binary":
                parts = key.split("_")
                display_key = f"{self._get_salt_name(parts[0])} + {self._get_salt_name(parts[1])}"
            else:
                display_key = key
            combo_counts[display_key] = len(indices)

        print("\nCreating salt statistics histograms...")
        self._create_salt_histograms(
            combo_counts,
            {self._get_salt_name(salt): count for salt, count in Counter(self.all_salts).items()},
        )

        # Create moiety pair heatmaps
        print("Creating moiety pair heatmaps...")
        self._create_moiety_pair_heatmaps()

        print("\nAll visualizations created successfully!")

    def _create_salt_histograms(self, combo_counts, salt_counts):
        """Create histograms for salt statistics."""
        # Salt combinations histogram
        plt.figure(figsize=(12, 8))

        sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [combo for combo, _ in sorted_combos]
        values = [count for _, count in sorted_combos]

        plt.title("Binary salt combinations in high CE formulations", fontsize=18)
        bars = plt.bar(range(len(labels)), values, color="steelblue")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
        plt.ylabel("Number of Formulations", fontsize=14)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(STATS_DIR, "binary_salt_combinations.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Unique salts histogram
        if salt_counts:
            plt.figure(figsize=(12, 8))
            sorted_salts = sorted(salt_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [salt for salt, _ in sorted_salts]
            values = [count for _, count in sorted_salts]

            plt.title("Unique salt molecules in high CE formulations", fontsize=18)
            bars = plt.bar(range(len(labels)), values, color="teal")
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
            plt.ylabel("Number of Occurrences", fontsize=14)

            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.tight_layout()
            plt.savefig(
                os.path.join(STATS_DIR, "unique_salts_combinations.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _create_moiety_pair_heatmaps(self):
        """Create heatmaps for each salt combination's moiety pairs."""
        for (salt_combo, salt_key, display_key), moiety_pairs in self.moiety_pair_data.items():
            if not moiety_pairs:
                continue

            pairs_df = pd.DataFrame(moiety_pairs)
            pair_counts = (
                pairs_df.groupby(["moiety1", "moiety2"])
                .size()
                .reset_index(name="count")
            )
            pivot_table = pair_counts.pivot(
                index="moiety1", columns="moiety2", values="count"
            ).fillna(0)

            self._generate_heatmap(pivot_table, salt_combo, display_key)

            # Save raw data
            safe_key = "".join(c if c.isalnum() else "" for c in str(display_key))
            save_dir = SINGLE_SALT_DIR if salt_combo == "single" else BINARY_SALT_DIR
            raw_data_file = os.path.join(save_dir, f"moiety_pairs_{salt_combo}_{safe_key}.csv")
            pairs_df.to_csv(raw_data_file, index=False)

    def _generate_heatmap(self, pivot_table, salt_combo, display_key):
        """Generate and save a heatmap for moiety pairs."""
        if pivot_table.empty:
            print(f"No data to generate heatmap for {salt_combo} salt '{display_key}'")
            return

        if salt_combo == "single":
            title = f"Moiety Pairs for Binary Solvents with {display_key} Salt"
        else:
            title = f"Moiety Pairs for Binary Solvents with {display_key} Salts"

        plt.figure(figsize=(16, 14))
        mask = pivot_table == 0
        heatmap = sns.heatmap(
            pivot_table,
            annot=False,
            cmap="Purples",
            linewidths=0,
            mask=mask,
            cbar_kws={"label": "Co-occurrence Count"},
        )

        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel("Co-occurrence Count", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        plt.title(title, fontsize=18, pad=20)
        plt.ylabel("Moieties in Solvent Molecule 1", fontsize=16)
        plt.xlabel("Moieties in Solvent Molecule 2", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12, rotation=0)
        plt.tight_layout()

        safe_key = "".join(c if c.isalnum() else "" for c in str(display_key))
        save_dir = SINGLE_SALT_DIR if salt_combo == "single" else BINARY_SALT_DIR
        filename = os.path.join(save_dir, f"moiety_pairs_{salt_combo}_{safe_key}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Created: {os.path.basename(filename)}")


def main():
    print()
    print("+" * 50)
    print("BINARY SOLVENT FORMULATION VISUALIZATION")
    print("+" * 50 + "\n")

    visualizer = BinaryFormulationVisualizer()
    visualizer.create_visualizations()

    print(f"\nVisualization complete.")
    print(f"Results saved to: {BINARY_ANALYSIS_DIR}\n")


if __name__ == "__main__":
    main()
