"""Plot Supplementary Fig. 3 from the saved Hobold current-density sweep."""

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=HERE / "data")
parser.add_argument("--output", type=Path, default=HERE / "hobold_combined_ce_vs_current.png")
args = parser.parse_args()

full = pd.read_csv(args.data_dir / "hobold_predictions_full.csv")
res = pd.read_csv(args.data_dir / "hobold_validation.csv")
type_map = dict(zip(res["name"], res["hobold_type"]))

KEEP = {

    "1M LiClO4 / PC",
    "1M LiPF6 / EC:DEC (1:1)",
    "1M LiPF6 / EC:DMC (1:1)",
    "1M LiFSI / FEC",

    "2M LiFSI / FEC",
    "7M LiFSI / FEC",
    "1M LiTFSI 3wt% LiNO3 / DOL:DME (1:1)",
    "2M LiFSI 1M LiTFSI 3wt% LiNO3 / DOL:DME (1:1)",
}
full = full[full["Electrolyte"].isin(KEEP)].copy()
res = res[res["name"].isin(KEEP)].copy()
print("kept electrolytes & types:")
for n, t in zip(res["name"], res["hobold_type"]):
    print(f"  Type {t}: {n}")

C_T1 = "#d95f02"
C_T2 = "#1b9e77"
CURRENTS = sorted(full["Current"].unique())

fig, ax_l = plt.subplots(figsize=(5.5, 5.5))
ax_r = ax_l.twinx()

for name, sub in full.groupby("Electrolyte"):
    sub = sub.sort_values("Current")
    htype = type_map.get(name)
    ax = ax_l if htype == 1 else ax_r
    color = C_T1 if htype == 1 else C_T2
    ax.plot(sub["Current"], sub["pred_ce_mean"], marker="o",
            color=color, alpha=0.6, lw=1.5, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6)


for ax in (ax_l, ax_r):
    ax.spines["top"].set_visible(False)
ax_l.tick_params(axis="y", colors=C_T1, labelsize=10)
ax_r.tick_params(axis="y", colors=C_T2, labelsize=10)
ax_l.spines["left"].set_color(C_T1)
ax_r.spines["right"].set_color(C_T2)
ax_l.spines["right"].set_color(C_T2)

ax_l.set_xticks(CURRENTS)
ax_l.get_xaxis().set_major_formatter(ScalarFormatter())
ax_l.xaxis.set_minor_locator(NullLocator())
ax_l.set_xlim(0.0, 4.3)

import matplotlib.transforms as mtransforms
labels = ax_l.get_xticklabels()
dx_025 = mtransforms.ScaledTranslation(-2 / 72., 0, fig.dpi_scale_trans)
dx_050 = mtransforms.ScaledTranslation(+2 / 72., 0, fig.dpi_scale_trans)
labels[0].set_transform(labels[0].get_transform() + dx_025)
labels[1].set_transform(labels[1].get_transform() + dx_050)
ax_l.set_xlabel("Current density (mA cm$^{-2}$)", fontsize=11)

import numpy as np
TICK_STEP = 0.5



def snap(vals, step=TICK_STEP):
    lo = np.floor((vals.min() - step) / step) * step
    hi = np.ceil((vals.max() + step) / step) * step
    return lo, hi

t1v = full[full["Electrolyte"].map(type_map) == 1]["pred_ce_mean"].values
t2v = full[full["Electrolyte"].map(type_map) == 2]["pred_ce_mean"].values
lo1, hi1 = snap(t1v)
lo2, hi2 = snap(t2v)

span = max(hi1 - lo1, hi2 - lo2)
def grow(lo, hi, target):
    extra = target - (hi - lo)
    return lo - extra / 2, hi + extra / 2
lo1, hi1 = grow(lo1, hi1, span)
lo2, hi2 = grow(lo2, hi2, span)

lo1 = np.floor(lo1 / TICK_STEP) * TICK_STEP; hi1 = lo1 + span
lo2 = np.floor(lo2 / TICK_STEP) * TICK_STEP; hi2 = lo2 + span
ax_l.set_ylim(lo1, hi1)
ax_r.set_ylim(lo2, hi2)
ax_l.set_yticks(np.arange(np.ceil(lo1), np.floor(hi1) + 0.5, 1.0))

r_ticks = np.arange(np.ceil(lo2), np.floor(hi2) + 0.5, 1.0)
r_ticks = r_ticks[(r_ticks >= 95) & (r_ticks <= 100)]
ax_r.set_yticks(r_ticks)
ax_l.set_ylabel("Type 1 — CE (%)", fontsize=11, color=C_T1)
ax_r.set_ylabel("Type 2 — CE (%)", fontsize=11, color=C_T2)

legend_lines = [
    Line2D([0], [0], color=C_T1, lw=2, marker="o", markersize=6, label="Type 1"),
    Line2D([0], [0], color=C_T2, lw=2, marker="o", markersize=6, label="Type 2"),
]
ax_l.legend(handles=legend_lines, loc="upper center",
            bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, frameon=False)

fig.tight_layout()
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
