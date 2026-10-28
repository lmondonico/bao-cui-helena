"""
Distance-attention analysis for moiety interactions in HELENA model.
Analyzes relationships between moiety distances and attention weights.
"""

########################
## Parameters
N_TOP_MOLECULES = 30
N_BOTTOM_MOLECULES = 30
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from config.datasets import MAX_MOLECULES
from modules.dataset_processing import FeatureEngineering, CollateFn
from tasks.postprocess.postprocess_utils.postprocess_utils import (
    MoeityEncoderAttentionUtils,
    MOIETY_ANALYSIS_DIR,
    DISTANCE_ANALYSIS_DIR,
)
from tasks.postprocess.postprocess_utils.excluded_moieties import EXCLUDED_MOIETIES

import warnings

warnings.filterwarnings("ignore")

os.makedirs(MOIETY_ANALYSIS_DIR, exist_ok=True)
os.makedirs(DISTANCE_ANALYSIS_DIR, exist_ok=True)
os.makedirs(os.path.join(DISTANCE_ANALYSIS_DIR, "highest"), exist_ok=True)
os.makedirs(os.path.join(DISTANCE_ANALYSIS_DIR, "lowest"), exist_ok=True)


class DistanceAttentionAnalyzer(MoeityEncoderAttentionUtils):
    def __init__(self):
        super().__init__()
        self.results_dir = DISTANCE_ANALYSIS_DIR

    def collect_distance_attention_data(self, results_df, performance):
        """
        Collect data about distances, attention weights, and LCE values
        for each moiety pair in the molecules.
        """
        distance_attention_data = []
        perf_df = results_df[results_df["performance"] == performance]

        for _, row in perf_df.iterrows():
            mol_smiles = row["molecule"]
            lce = row["LCE"]
            molar_ratio = row["molar_ratio"]

            # Get attention and distance information
            local_attn, global_attn, moiety_ids, _ = self.get_attention_weights(
                mol_smiles, molar_ratio
            )

            if local_attn is None:
                continue

            # Ensure molecule is in map_dict and get distance matrix
            self._ensure_molecule_in_map_dict(mol_smiles)

            # Create temporary dataframe with new format (SMILES:molar_ratio)
            temp_data = pd.DataFrame(
                {
                    "Solvent_1": [f"{mol_smiles}:{molar_ratio}"],
                    "Salt_1": [None],
                    "LCE": [0.0],
                    "Row_ID": [int(row["Row_ID"])],
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

            dist_matrix = dataset[0]["dist_matrix1"]

            # Get moiety names
            moiety_names = [
                self.tokenizer.key_moiety_map.get(mid) for mid in moiety_ids
            ]

            # Collect data for each moiety pair
            for i, moiety1 in enumerate(moiety_names):
                for j, moiety2 in enumerate(moiety_names):
                    if (
                        i != j
                        and moiety1 not in EXCLUDED_MOIETIES
                        and moiety2 not in EXCLUDED_MOIETIES
                        and dist_matrix[i, j] < 1e8  # Exclude disconnected moieties (salts)
                    ):
                        distance_attention_data.append(
                            {
                                "moiety1": moiety1,
                                "moiety2": moiety2,
                                "distance": dist_matrix[i, j],
                                "local_attention": local_attn[i, j],
                                "global_attention": global_attn[i, j],
                                "lce": lce,
                            }
                        )

        return pd.DataFrame(distance_attention_data)

    def create_distance_slice_heatmaps(self, distance_data, performance):
        """
        Create side-by-side heatmaps showing moiety-moiety attention at different distances.
        Only shows moieties actually present in the molecules for that performance level.
        """
        save_dir = os.path.join(self.results_dir, performance)
        unique_distances = sorted(distance_data["distance"].unique())

        present_moieties_1 = sorted(distance_data["moiety1"].unique())
        present_moieties_2 = sorted(distance_data["moiety2"].unique())
        present_moieties = sorted(set(present_moieties_1) | set(present_moieties_2))

        for distance in unique_distances:
            # Filter data for current distance
            distance_slice = distance_data[distance_data["distance"] == distance]

            # Create pivot tables only for present moieties
            local_matrix = pd.pivot_table(
                distance_slice[
                    distance_slice["moiety1"].isin(present_moieties)
                    & distance_slice["moiety2"].isin(present_moieties)
                ],
                values="local_attention",
                index="moiety1",
                columns="moiety2",
                aggfunc="mean",
            )

            global_matrix = pd.pivot_table(
                distance_slice[
                    distance_slice["moiety1"].isin(present_moieties)
                    & distance_slice["moiety2"].isin(present_moieties)
                ],
                values="global_attention",
                index="moiety1",
                columns="moiety2",
                aggfunc="mean",
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

            # Plot local attention
            sns.heatmap(
                local_matrix,
                cmap="YlOrRd",
                annot=True,
                fmt=".2f",
                ax=ax1,
                cbar_kws={
                    "label": "Local Attention Weight",
                },
            )
            ax1.set_title(
                f"Local Attention at Distance {distance}", fontsize=18, pad=25
            )
            ax1.set_xlabel("Reference Moiety", fontsize=14, labelpad=13)
            ax1.set_ylabel("Target Moiety", fontsize=14, labelpad=13)
            ax1.set_xticklabels(
                ax1.get_xticklabels(), rotation=45, ha="right", fontsize=12
            )
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=12)

            # Plot global attention
            sns.heatmap(
                global_matrix,
                cmap="Blues",
                annot=True,
                fmt=".2f",
                ax=ax2,
                cbar_kws={
                    "label": "Global Attention Weight",
                },
            )
            for ax in [ax1, ax2]:
                cbar = ax.collections[0].colorbar
                cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=14)
                cbar.ax.tick_params(labelsize=12)

            ax2.set_title(
                f"Global Attention at Distance {distance}", fontsize=18, pad=25
            )
            ax2.set_xlabel("Reference Moiety", fontsize=14, labelpad=12)
            ax2.set_ylabel("Target Moiety", fontsize=14, labelpad=12)
            ax2.set_xticklabels(
                ax2.get_xticklabels(), rotation=45, ha="right", fontsize=12
            )
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=12)

            plt.suptitle(
                f"{performance.capitalize()} LCE: Moiety-Moiety Attention Patterns at Distance {distance}",
                fontsize=18,
                fontweight="bold",
            )

            plt.subplots_adjust(wspace=2)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(
                os.path.join(save_dir, f"heatmap_distance_{distance}.png"),
                bbox_inches="tight",
                dpi=600,
            )
            plt.close()

    def create_stacked_publication_figures(self, distance_data, performance):
        """
        Create publication-ready figures by vertically stacking heatmaps at different distances.
        Maximum 3 heatmaps per figure. Starts from distance 1.
        """
        from matplotlib.gridspec import GridSpec
        from PIL import Image

        save_dir = os.path.join(self.results_dir, performance)
        pub_dir = os.path.join(save_dir, "publication")
        os.makedirs(pub_dir, exist_ok=True)

        # Get unique distances starting from 1, sorted
        unique_distances = sorted([d for d in distance_data["distance"].unique() if d >= 1])

        if len(unique_distances) == 0:
            print(f"No distances >= 1 found for {performance}")
            return

        # Group distances into sets of 3
        distance_groups = [
            unique_distances[i : i + 3] for i in range(0, len(unique_distances), 3)
        ]

        total_figures = len(distance_groups)

        for fig_idx, distance_group in enumerate(distance_groups):
            n_distances = len(distance_group)

            # Load the individual heatmap images for this group
            images = []
            for distance in distance_group:
                img_path = os.path.join(save_dir, f"heatmap_distance_{distance}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    # Crop whitespace
                    img_array = np.array(img)
                    non_white_rows = np.where(np.any(img_array < 250, axis=(1, 2)))[0]
                    if len(non_white_rows) > 0:
                        img = img.crop(
                            (0, max(0, non_white_rows[0] - 5), img.width, non_white_rows[-1] + 5)
                        )
                    images.append((distance, img))

            if not images:
                continue

            # Create stacked figure
            fig_height = 8 * n_distances
            fig = plt.figure(figsize=(22, fig_height))
            gs = GridSpec(
                n_distances,
                1,
                figure=fig,
                height_ratios=[1] * n_distances,
                hspace=0.08,
            )

            for idx, (distance, img) in enumerate(images):
                ax = fig.add_subplot(gs[idx, 0])
                ax.imshow(img)
                ax.axis("off")

            # Create figure label indicating set
            start_dist = distance_group[0]
            end_dist = distance_group[-1]
            fig_label = f"Figure {fig_idx + 1}/{total_figures} (Distances {start_dist}-{end_dist})"

            output_path = os.path.join(
                pub_dir, f"publication_stacked_{performance}_fig{fig_idx + 1}.png"
            )
            plt.savefig(
                output_path,
                bbox_inches="tight",
                dpi=600,
                facecolor="white",
            )
            plt.close(fig)
            print(f"Saved: {output_path}")

    def analyze_optimal_distances(self, distance_data, performance):
        """
        Analyze optimal distances between moiety pairs based on attention and LCE.
        """
        optimal_distances = defaultdict(list)

        # Group by moiety pairs
        for (moiety1, moiety2), group in distance_data.groupby(["moiety1", "moiety2"]):
            if moiety1 == moiety2:
                continue
            group = group.copy()
            group["attention_lce_score"] = group["local_attention"] * group["lce"]
            optimal_row = group.loc[group["attention_lce_score"].idxmax()]

            optimal_distances[(moiety1, moiety2)].append(
                {
                    "optimal_distance": optimal_row["distance"],
                    "attention_score": optimal_row["local_attention"],
                    "lce": optimal_row["lce"],
                }
            )

        optimal_df = pd.DataFrame(
            [
                {
                    "moiety1": pair[0],
                    "moiety2": pair[1],
                    "optimal_distance": data[0]["optimal_distance"],
                    "attention_score": data[0]["attention_score"],
                    "lce": data[0]["lce"],
                }
                for pair, data in optimal_distances.items()
            ]
        )

        save_dir = os.path.join(self.results_dir, performance)
        optimal_df.to_csv(os.path.join(save_dir, "optimal_distances.csv"), index=False)
        return optimal_df


def main():
    analyzer = DistanceAttentionAnalyzer()
    print("+" * 50)
    print("POST-PROCESSING\nDistance-Attention Analysis")
    print("+" * 50)

    results_df = analyzer.analyze_extreme_lce_formulations(
        n_top=N_TOP_MOLECULES, n_bottom=N_BOTTOM_MOLECULES
    )

    if not results_df.empty:
        for performance in ["highest", "lowest"]:
            print(f"\nProcessing {performance} LCE molecules...")

            print("Collecting distance-attention data...")
            distance_data = analyzer.collect_distance_attention_data(
                results_df, performance
            )

            print("Generating distance slice heatmaps...")
            analyzer.create_distance_slice_heatmaps(distance_data, performance)

            print("Generating publication-ready stacked figures...")
            analyzer.create_stacked_publication_figures(distance_data, performance)

            print("Analyzing optimal distances...")
            optimal_distances = analyzer.analyze_optimal_distances(
                distance_data, performance
            )

            n_molecules = (
                N_TOP_MOLECULES if performance == "highest" else N_BOTTOM_MOLECULES
            )
            print(f"\nAnalysis Summary for {performance.capitalize()} LCE:")
            print(f"Number of molecules analyzed: {n_molecules}")
            print(f"Unique distances found: {len(distance_data['distance'].unique())}")
            print(f"Unique moiety pairs analyzed: {len(optimal_distances)}")

        print(f"\nResults saved to: {analyzer.results_dir}")


if __name__ == "__main__":
    main()
