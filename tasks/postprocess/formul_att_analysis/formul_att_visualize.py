"""
Script to visualize the attention weights in the HELENA-CE model.
Extracts weights from:
- MoietyEncoder (local/global attention with Wq/Wk/Wv)
- GroupAttention (nn.MultiheadAttention with in_proj_weight)
- FormulationAttention (molecular-level cross-attention)
"""

import os
import sys

# Add project root to path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tasks.postprocess.postprocess_utils.postprocess_utils import (
    MoeityEncoderAttentionUtils,
    POSTPROCESS_RESULTS_DIR,
    MODEL_CHECKPOINT_PATH,
)

import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(POSTPROCESS_RESULTS_DIR, "group_formulation_attention")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AttentionWeightVisualizer(MoeityEncoderAttentionUtils):
    """Class to visualize attention weights in the HELENA-CE model"""

    def __init__(self):
        super().__init__()
        print(f"Model loaded from: {MODEL_CHECKPOINT_PATH}\n")

    def extract_and_visualize_moiety_encoder_weights(self):
        """
        Extract and visualize the weight matrices from MoietyEncoder.
        MoietyEncoder uses custom MultiHeadAttention with explicit Wq/Wk/Wv.
        """
        try:
            encoder = self.model.encoder_moiety

            # Get weights from all encoder layers
            layer_weights = []
            for layer_idx, layer in enumerate(encoder.encoder_layers):
                # Local attention (multi_head_att_1)
                local_weights = {
                    "query": layer.multi_head_att_1.Wq.weight.detach().cpu().numpy(),
                    "key": layer.multi_head_att_1.Wk.weight.detach().cpu().numpy(),
                    "value": layer.multi_head_att_1.Wv.weight.detach().cpu().numpy(),
                }
                # Global attention (multi_head_att_2)
                global_weights = {
                    "query": layer.multi_head_att_2.Wq.weight.detach().cpu().numpy(),
                    "key": layer.multi_head_att_2.Wk.weight.detach().cpu().numpy(),
                    "value": layer.multi_head_att_2.Wv.weight.detach().cpu().numpy(),
                }
                layer_weights.append(
                    {
                        "layer": layer_idx,
                        "local": local_weights,
                        "global": global_weights,
                    }
                )

            # Visualize the first layer's weights
            if layer_weights:
                first_layer = layer_weights[0]
                _, axes = plt.subplots(2, 3, figsize=(18, 12))
                weight_types = ["query", "key", "value"]

                for i, weight_type in enumerate(weight_types):
                    # Local attention weights
                    local_matrix = first_layer["local"][weight_type]
                    vmin_local = np.percentile(local_matrix, 5)
                    vmax_local = np.percentile(local_matrix, 95)

                    sns.heatmap(
                        local_matrix,
                        cmap="YlOrRd",
                        ax=axes[0, i],
                        vmin=vmin_local,
                        vmax=vmax_local,
                    )
                    axes[0, i].set_title(
                        f"Local {weight_type.capitalize()} Weights", fontsize=18
                    )
                    axes[0, i].set_xlabel("Output Features", fontsize=14)
                    axes[0, i].set_ylabel("Input Features", fontsize=14)

                    # Global attention weights
                    global_matrix = first_layer["global"][weight_type]
                    vmin_global = np.percentile(global_matrix, 5)
                    vmax_global = np.percentile(global_matrix, 95)

                    sns.heatmap(
                        global_matrix,
                        cmap="Blues",
                        ax=axes[1, i],
                        vmin=vmin_global,
                        vmax=vmax_global,
                    )
                    axes[1, i].set_title(
                        f"Global {weight_type.capitalize()} Weights", fontsize=18
                    )
                    axes[1, i].set_xlabel("Output Features", fontsize=14)
                    axes[1, i].set_ylabel("Input Features", fontsize=14)

                plt.suptitle("MoietyEncoder Attention Weights", fontsize=20)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_DIR, "moiety_encoder_attention_weights.png"),
                    dpi=900,
                )
                plt.close()

            return layer_weights

        except Exception as e:
            print(f"Error extracting moiety encoder weights: {str(e)}")
            self.find_linear_layers(self.model.encoder_moiety, prefix="encoder_moiety")
            return None

    def extract_and_visualize_solvent_group_attention_weights(self):
        """
        Extract and visualize the weight matrices from SolventGroup.
        SolventGroup uses nn.MultiheadAttention which has combined in_proj_weight.
        Note: SaltGroup has no attention mechanism (uses mean pooling only).
        """
        try:
            solvent_group = self.model.solvent_group

            # Get weights from all layers
            layer_weights = []
            for layer_idx, attn in enumerate(solvent_group.layers):
                # nn.MultiheadAttention stores QKV combined in in_proj_weight
                # Shape: (3 * d_model, d_model) where rows are [Q, K, V]
                in_proj = attn.in_proj_weight.detach().cpu().numpy()
                d_model = in_proj.shape[1]

                # Split into Q, K, V
                weights = {
                    "query": in_proj[:d_model, :],
                    "key": in_proj[d_model : 2 * d_model, :],
                    "value": in_proj[2 * d_model :, :],
                }
                layer_weights.append({"layer": layer_idx, "weights": weights})

            # Visualize first layer
            if layer_weights:
                first_layer = layer_weights[0]["weights"]
                _, axes = plt.subplots(1, 3, figsize=(20, 6))

                for i, (weight_type, weight_matrix) in enumerate(first_layer.items()):
                    vmin = np.percentile(weight_matrix, 5)
                    vmax = np.percentile(weight_matrix, 95)

                    sns.heatmap(
                        weight_matrix,
                        cmap="viridis",
                        ax=axes[i],
                        vmin=vmin,
                        vmax=vmax,
                    )
                    axes[i].set_title(
                        f"Solvent Group Attention {weight_type.capitalize()}", fontsize=16
                    )
                    axes[i].set_xlabel("Input Features", fontsize=12)
                    axes[i].set_ylabel("Output Features", fontsize=12)

                plt.suptitle("Solvent Group Attention Weights", fontsize=18)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_DIR, "solvent_group_attention_weights.png"),
                    dpi=900,
                )
                plt.close()

            return layer_weights

        except Exception as e:
            print(f"Error extracting solvent group attention weights: {str(e)}")
            self.find_linear_layers(
                self.model.solvent_group, prefix="solvent_group"
            )
            return None

    def extract_and_visualize_formulation_attention_weights(self):
        """
        Extract and visualize the formulation-level attention weights.
        FormulationAttention also uses nn.MultiheadAttention.
        """
        try:
            formulation_attn = self.model.molecular_attention

            # Get combined QKV weights from nn.MultiheadAttention
            in_proj = formulation_attn.attention.in_proj_weight.detach().cpu().numpy()
            d_model = in_proj.shape[1]

            weights = {
                "query": in_proj[:d_model, :],
                "key": in_proj[d_model : 2 * d_model, :],
                "value": in_proj[2 * d_model :, :],
            }

            _, axes = plt.subplots(1, 3, figsize=(20, 6))
            weight_types = ["query", "key", "value"]

            for i, weight_type in enumerate(weight_types):
                weight_matrix = weights[weight_type]
                vmin = np.percentile(weight_matrix, 5)
                vmax = np.percentile(weight_matrix, 95)

                sns.heatmap(
                    weight_matrix, cmap="Purples", ax=axes[i], vmin=vmin, vmax=vmax
                )
                axes[i].set_title(
                    f"Formulation Attention {weight_type.capitalize()}", fontsize=20
                )
                axes[i].set_xlabel("Input Features", fontsize=16)
                axes[i].set_ylabel("Output Features", fontsize=16)

            plt.suptitle("Formulation Attention Weights", fontsize=18)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, "formulation_attention_weights.png"),
                dpi=900,
            )
            plt.close()

            return weights

        except Exception as e:
            print(f"Error extracting formulation attention weights: {str(e)}")
            self.find_linear_layers(
                self.model.molecular_attention, prefix="molecular_attention"
            )
            return None

    def find_linear_layers(self, module, prefix=""):
        """Recursively search for linear layers in the model to help diagnose structure"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                print(f"Found Linear layer: {full_name}, shape: {child.weight.shape}")
            elif isinstance(child, nn.MultiheadAttention):
                print(f"Found MultiheadAttention: {full_name}")
                if hasattr(child, "in_proj_weight"):
                    print(f"  - in_proj_weight shape: {child.in_proj_weight.shape}")
                if hasattr(child, "out_proj"):
                    print(f"  - out_proj.weight shape: {child.out_proj.weight.shape}")
            else:
                self.find_linear_layers(child, full_name)


def main():
    print()
    print("+" * 50)
    print("POST-PROCESSING\nExtracting HELENA-CE Attention Weights")
    print("+" * 50)

    visualizer = AttentionWeightVisualizer()

    print("\nExtracting MoietyEncoder attention weights...")
    visualizer.extract_and_visualize_moiety_encoder_weights()

    print("Extracting SolventGroup attention weights...")
    visualizer.extract_and_visualize_solvent_group_attention_weights()

    print("Extracting FormulationAttention weights...")
    visualizer.extract_and_visualize_formulation_attention_weights()

    print(f"\nVisualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
