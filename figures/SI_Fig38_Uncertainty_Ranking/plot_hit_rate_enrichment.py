"""Plot Supplementary Figure 38 and summarize experimental hit-rate enrichment."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]


def main():
    df = pd.read_csv(REPO / "data/HELENA-CE-26.csv")
    n = len(df)
    base_99 = (df["LCE"] > 2.0).mean()
    base_995 = (df["LCE"] > 2.3).mean()
    exp_99 = 6 / 8
    exp_995 = 4 / 8
    enr_99 = exp_99 / base_99 if base_99 > 0 else float("inf")
    enr_995 = exp_995 / base_995 if base_995 > 0 else float("inf")
    txt = [f"Dataset n = {n}"]
    txt.append(f"Dataset base rate CE > 99%   (LCE > 2.0): {base_99 * 100:.1f}%  "
               f"({(df['LCE'] > 2.0).sum()}/{n})")
    txt.append(f"Dataset base rate CE > 99.5% (LCE > 2.3): {base_995 * 100:.1f}%  "
               f"({(df['LCE'] > 2.3).sum()}/{n})")
    txt.append("")
    txt.append(f"Experimental: 6/8 = {exp_99 * 100:.1f}% achieved CE > 99%")
    txt.append(f"Experimental: 4/8 = {exp_995 * 100:.1f}% achieved CE > 99.5%")
    txt.append("")
    txt.append(f"Enrichment over base rate (CE > 99%)  : {enr_99:.2f}x")
    txt.append(f"Enrichment over base rate (CE > 99.5%): {enr_995:.2f}x")
    body = "\n".join(txt)
    (FIGURE_DIR / "hit_rate_summary.txt").write_text(body + "\n")
    print(body)

    base_rates = np.array([base_99, base_995])
    hit_rates = np.array([exp_99, exp_995])
    enrichment = [enr_99, enr_995]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    baseline = ax.bar(
        x - width / 2, base_rates * 100, width, label="Dataset base rate",
        color="#f6c4cc", edgecolor="#555555", linewidth=1.0,
    )
    selected = ax.bar(
        x + width / 2, hit_rates * 100, width, label="HELENA hit rate",
        color="#a9d6b9", edgecolor="#555555", linewidth=1.0,
    )
    for bars in (baseline, selected):
        ax.bar_label(bars, labels=[f"{b.get_height():.0f}%" for b in bars],
                     padding=4, fontsize=12, color="#555555")

    purple = "#8c6bb1"
    for idx, factor in enumerate(enrichment):
        bottom = base_rates[idx] * 100 + 7
        top = max(base_rates[idx], hit_rates[idx]) * 100 + 12
        left, right = x[idx] - width / 2, x[idx] + width / 2
        ax.plot([left, left, right, right], [bottom, top, top, top - 3],
                color=purple, linewidth=1.8)
        ax.plot(right, top - 3, marker="v", color=purple, markersize=8)
        ax.text(x[idx], top + 2, f"{factor:.1f}×", ha="center", va="bottom",
                color=purple, fontsize=20, fontweight="bold")

    ax.set_xticks(x, ["CE > 99.0%", "CE > 99.5%"])
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.65, len(x) - 0.35)
    ax.set_ylabel("Hit rate (%)", fontsize=17)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=12, handlelength=1.6)
    fig.tight_layout()
    output = FIGURE_DIR / "hit_rate_enrichment.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
