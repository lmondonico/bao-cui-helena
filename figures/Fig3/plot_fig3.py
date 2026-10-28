"""Regenerate Figure 3 plots from the adjacent saved results.

Run from the repository root with its Python environment. Use --output-dir to
write into a new directory; published images are never overwritten by default.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


# Original Fig3 Seaborn bright palette: indices 0, 2, 3, 4, 5, 6, 9, 1.
# Pooled test scores from manuscript Figure 3 / SI Table 8.
MODELS = {
    "LinReg": ("#023eff", 0.459, 0.292),
    "SVM": ("#1ac938", 0.539, 0.270),
    "RF": ("#e8000b", 0.609, 0.249),
    "XGBoost": ("#8b2be2", 0.611, 0.248),
    "MPNN": ("#9f4800", 0.612, 0.248),
    "AttentiveFP": ("#f14cc1", 0.635, 0.240),
    "MolFormer": ("#00d7ff", 0.607, 0.249),
    "HELENA": ("#ff7c00", 0.693, 0.220),
}


def load_comparisons():
    dataset = pd.read_csv(ROOT / "data/HELENA-CE-26.csv")
    truth = dataset.set_index("Row_ID")["LCE"]
    # Classical baseline exporters use positions after their seed-142 shuffle.
    shuffled_ids = dataset.sample(frac=1, random_state=142)["Row_ID"].reset_index(drop=True)
    predictions, members, checks = {}, {}, []
    for name, (_, expected_r2, expected_rmse) in MODELS.items():
        folder = HERE / "data" / name
        pred = pd.read_csv(folder / "nested_cv_predictions.csv")
        pred = pred.rename(columns={"elf_code": "row_id"})
        if name in {"LinReg", "SVM", "RF", "XGBoost"}:
            if set(pred["row_id"]) != set(range(len(dataset))):
                raise ValueError(f"{name}: unexpected shuffled-position indices")
            pred["row_id"] = pred["row_id"].map(shuffled_ids)
        pred = pred.set_index("row_id").sort_index()
        if not pred.index.is_unique or set(pred.index) != set(truth.index):
            raise ValueError(f"{name}: predictions do not cover the 1,000 unique dataset rows")
        if not pred["n_predictions"].eq(8).all():
            raise ValueError(f"{name}: expected eight held-out predictions per formulation")
        np.testing.assert_allclose(pred["actual_lce"], truth.loc[pred.index], atol=1e-6)
        residual = pred["predicted_lce"] - pred["actual_lce"]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        r2 = float(1 - np.sum(residual ** 2) / np.sum(
            (pred["actual_lce"] - pred["actual_lce"].mean()) ** 2
        ))
        if (round(r2, 3), round(rmse, 3)) != (expected_r2, expected_rmse):
            raise ValueError(f"{name}: saved predictions do not match published rounded scores")
        metrics = pd.read_csv(folder / "individual_model_test_metrics.csv")
        pairs = set(zip(metrics["outer_fold"], metrics["model_idx_in_fold"]))
        if len(metrics) != 40 or pairs != {(f, m) for f in range(1, 6) for m in range(1, 9)}:
            raise ValueError(f"{name}: incomplete five-fold, eight-member metric records")
        predictions[name], members[name] = pred, metrics
        checks.append(dict(model=name, formulations=len(pred), members=len(metrics),
                           r2=r2, rmse=rmse, matches_published_3dp=True))
    return predictions, members, pd.DataFrame(checks)


def save(fig, output, name, dpi=900, bbox_inches="tight"):
    fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def plot_distributions(members, output):
    # Original Figure 3 style, including HSV lightening.
    colors = np.array([mcolors.to_rgb(model[0]) for model in MODELS.values()])
    hsv = mcolors.rgb_to_hsv(colors)
    hsv[:, 1] *= 0.7
    palette = {name: sns.desaturate(color, 0.75)
               for name, color in zip(MODELS, mcolors.hsv_to_rgb(hsv))}
    style = {
        "boxprops": {"edgecolor": "black"},
        "medianprops": {"color": "black"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
        "showfliers": True,
        "flierprops": {"marker": "*", "markersize": 5},
    }
    for metric, label in [("test_rmse", "RMSE"), ("test_r2", "R² Score")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        rng = np.random.default_rng(142)
        data = pd.concat([members[name][[metric]].assign(Model=name) for name in MODELS])
        sns.boxplot(data=data, x="Model", y=metric, hue="Model", order=list(MODELS),
                    palette=palette, saturation=1, width=0.5, legend=False, zorder=1,
                    patch_artist=True, ax=ax, **style)
        fliers = [line for line in ax.lines if line.get_marker() == "*"]
        for flier, color in zip(fliers, palette.values()):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(np.asarray(color) * 0.65)
        for index, name in enumerate(MODELS):
            values = members[name][metric].to_numpy()
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = q3 - q1
            non_outliers = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
            ax.scatter(index + rng.normal(0, 0.05, len(non_outliers)), non_outliers,
                       color=palette[name], edgecolors=np.asarray(palette[name]) * 0.65,
                       linewidths=0.4, s=16, alpha=0.9, zorder=3)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=16)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(False)
        ymin, ymax = ax.get_ylim()
        # Keep the original section headings clear of the current 40-member data.
        ymax += 0.12 * (ymax - ymin)
        ax.axvspan(-0.4, 3.4, color=(1.00, 0.98, 0.81), alpha=0.4, zorder=0)
        ax.axvspan(3.6, 7.6, color=(1.00, 0.88, 0.93), alpha=0.25, zorder=0)
        title_y = ymax * (0.96 if metric == "test_rmse" else 0.95)
        for x, title in [(1.5, "Traditional ML"), (5.5, "Molecule ML")]:
            ax.text(x, title_y, title, ha="center", va="bottom", fontsize=15)
        ax.set_ylim(ymin, ymax)
        fig.tight_layout()
        save(fig, output, f"{metric.removeprefix('test_')}_metrics_plot")


def parity_panel(ax, marginal_x, marginal_y, name, pred, check):
    # Original JointGrid density style with the paper's pooled-test annotation.
    color = MODELS[name][0]
    sns.kdeplot(x=pred["actual_lce"], ax=marginal_x, fill=True,
                color=color, alpha=0.3, bw_adjust=0.75)
    sns.kdeplot(y=pred["predicted_lce"], ax=marginal_y, fill=True,
                color=color, alpha=0.3, bw_adjust=0.75)
    marginal_x.set_axis_off()
    marginal_y.set_axis_off()
    ax.scatter(pred["actual_lce"], pred["predicted_lce"], s=18, alpha=0.3,
               color=color, edgecolors="none")
    ax.plot([0.5, 3], [0.5, 3], "k--", linewidth=1.5, alpha=0.8)
    ax.set(xlim=(0.5, 3), ylim=(0.5, 3))
    ax.set_xlabel("Ground Truth LCE", fontsize=16)
    ax.set_ylabel("Predicted LCE", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.text(0.98, 0.05, f"{name}\nR² = {check.r2:.3f}\nRMSE = {check.rmse:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", color=color, fontsize=16)
    handle = mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                          markersize=6, label="Averaged Test")
    ax.legend(handles=[handle], frameon=False, fontsize=12, loc="upper left")


def plot_parities(predictions, checks, output):
    checks = checks.set_index("model")
    fig = plt.figure(figsize=(24, 12))
    grid = fig.add_gridspec(2, 4, left=0.04, right=0.99, bottom=0.06, top=0.99,
                           wspace=0.3, hspace=0.3)
    for index, name in enumerate(MODELS):
        joint = sns.JointGrid(x="actual_lce", y="predicted_lce", data=predictions[name],
                             height=6, ratio=5, space=0)
        parity_panel(joint.ax_joint, joint.ax_marg_x, joint.ax_marg_y,
                     name, predictions[name], checks.loc[name])
        save(joint.figure, output, f"{name}_parity_plot")
        panel = grid[index // 4, index % 4].subgridspec(6, 6, wspace=0, hspace=0)
        ax = fig.add_subplot(panel[1:, :-1])
        marginal_x = fig.add_subplot(panel[0, :-1], sharex=ax)
        marginal_y = fig.add_subplot(panel[1:, -1], sharey=ax)
        parity_panel(ax, marginal_x, marginal_y, name, predictions[name], checks.loc[name])
    # The mosaic is a review overview; individual panels retain original 900 dpi.
    save(fig, output, "parity_mosaic", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fig3")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir; existing figures are never overwritten.")
    predictions, members, checks = load_comparisons()
    args.output_dir.mkdir(parents=True)
    plot_distributions(members, args.output_dir)
    plot_parities(predictions, checks, args.output_dir)
    print(checks.to_string(index=False))
    print(f"Saved Figure 3 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
