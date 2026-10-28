"""Reproduce Extended Data Figure 2 from the existing high-CE predictions CSV.

Download high_ce_predictions.csv from https://doi.org/10.5281/zenodo.23464260
and place it beside this script in figures/ExtFig2/.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOLVENT_COLUMNS = [f"Solvent_{i}" for i in range(1, 6)]
SALT_COLUMNS = [f"Salt_{i}" for i in range(1, 4)]
SALT_NAMES = {
    "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F": "FSI",
    "[Li+].[B-]1(OC(=O)C(=O)O1)(F)F": "DFOB",
    "[Li+].F[P-](F)(F)(F)(F)F": "PF6",
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F": "TFSI",
}


def positive_components(column):
    """Parse SMILES:molar_ratio entries, retaining only positive ratios."""
    values = column.dropna()
    if values.empty:
        return values
    parts = values.str.rsplit(":", n=1, expand=True)
    if parts.shape[1] != 2 or parts[0].eq("").any():
        raise ValueError(f"Invalid component in {column.name}; expected SMILES:molar_ratio.")
    ratios = pd.to_numeric(parts[1], errors="raise")
    if not np.isfinite(ratios).all() or ratios.isna().any() or ratios.lt(0).any():
        raise ValueError(f"Invalid molar ratio in {column.name}.")
    return parts.loc[ratios.gt(0), 0]


def count_salts(input_path, chunksize=100_000):
    counts = Counter({name: 0 for name in SALT_NAMES.values()})
    binary_rows = 0
    for chunk in pd.read_csv(
        input_path,
        usecols=SOLVENT_COLUMNS + SALT_COLUMNS,
        dtype="string",
        chunksize=chunksize,
    ):
        solvent_count = pd.Series(0, index=chunk.index)
        for column in SOLVENT_COLUMNS:
            solvent_count.loc[positive_components(chunk[column]).index] += 1
        binary = chunk.loc[solvent_count.eq(2)]
        binary_rows += len(binary)
        for column in SALT_COLUMNS:
            salts = positive_components(binary[column])
            unknown = set(salts) - SALT_NAMES.keys()
            if unknown:
                raise ValueError(f"Unrecognized salt SMILES: {sorted(unknown)}")
            counts.update(salts.map(SALT_NAMES).value_counts().to_dict())
    if not binary_rows or not sum(counts.values()):
        raise ValueError("No positive salt components found in binary-solvent formulations.")
    return counts, binary_rows


def plot_counts(counts, output_path):
    names = list(SALT_NAMES.values())
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.linewidth": 1.2,
        "mathtext.default": "regular",
    }):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            range(len(names)), [counts[name] for name in names],
            color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"],
        )
        ax.bar_label(bars, labels=[f"{counts[name]:,}" for name in names], padding=3)
        ax.set_xticks(range(len(names)), ["FSI", "DFOB", r"PF$_6$", "TFSI"])
        ax.set_xlabel("Anion Molecule")
        ax.set_ylabel("Number of Occurrences")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(length=0, pad=9)
        ax.set_ylim(0, max(counts.values()) * 1.06)
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent / "high_ce_predictions.csv",
        help="High-CE predictions CSV (default: high_ce_predictions.csv beside this script).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/extfig2/extended_data_Fig_2.png",
        help="New output PNG path; existing files are never overwritten.",
    )
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a PNG path.")
    if args.output.exists():
        parser.error("Choose a new --output path; existing figures are never overwritten.")
    counts, binary_rows = count_salts(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_counts(counts, args.output)
    print(f"Binary-solvent formulations: {binary_rows:,}")
    for name, count in counts.items():
        print(f"{name}: {count:,}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
