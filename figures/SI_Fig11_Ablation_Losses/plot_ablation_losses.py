"""Plot saved ablation loss curves for outer fold 4, model 4."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


variants = [
    ("helena_baseline.csv", "HELENA (Baseline)"),
    ("no_moiety_encoder.csv", "No Moiety Encoder"),
    ("no_salt_group.csv", "No Salt Group"),
    ("no_solvent_group_attention.csv", "No Solvent Group"),
    ("no_molar_ratio.csv", "No Molar Ratio"),
    ("no_formulation_attention.csv", "No Formulation Attention"),
    ("no_current.csv", "No Current"),
    ("no_method.csv", "No Method"),
]

figure_dir = Path(__file__).resolve().parent
loss_dir = figure_dir / "data"


outer_fold = 4
model_idx = 4


def smooth_curve(values, weight=0.6):
    """Exponential moving average smoothing for visualization."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed



fig, axes = plt.subplots(4, 2, figsize=(10, 16))
axes = axes.flatten()

for idx, (csv_file, title) in enumerate(variants):
    ax = axes[idx]


    df = pd.read_csv(loss_dir / csv_file)


    model_data = df[(df["outer_fold"] == outer_fold) & (df["model_idx"] == model_idx)]

    if len(model_data) > 0:
        epochs = list(range(1, len(model_data) + 1))
        train_raw = model_data["train_loss"].tolist()
        val_raw = model_data["val_loss"].tolist()
        train_smooth = smooth_curve(train_raw)
        val_smooth = smooth_curve(val_raw)


        ax.plot(epochs, train_raw, color="orange", alpha=0.3, linewidth=1)
        ax.plot(epochs, val_raw, color="blue", alpha=0.3, linewidth=1)

        ax.plot(epochs, train_smooth, color="orange", label="Train", linewidth=2)
        ax.plot(epochs, val_smooth, color="blue", label="Val", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig(figure_dir / "ablation_losses.png", dpi=300)
plt.close()

print(f"Saved: {figure_dir / 'ablation_losses.png'}")
