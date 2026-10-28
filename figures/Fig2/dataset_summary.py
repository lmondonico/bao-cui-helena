"""Plot the Figure 2 molecular histograms and moiety-frequency panels.

Uses the adjacent 2023 and 2026 datasets and the original HELENA tokenizer.
Moieties are counted once per molecule occurrence in a formulation. Panel b
shows the 30 moieties labelled in the paper; no dataset rows are filtered.
This script does not reconstruct the assembled figure or its central table.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from config.datasets import SOLVENT_COLUMNS, SALT_COLUMNS
from modules.tokenizer.tokenizer_helena import Tokenizer

COLORS = ("#3b71a1", "#cc4c35")
LABELS = ("2023 Dataset", "New Dataset")
PANEL_MOIETIES = (
    ("fluoro", "secondary_carbon", "lithium_cation", "ether", "methyl",
     "sulfonyl", "nitrogen_anion", "quaternary_carbon", "carbonate_ester",
     "tertiary_carbon", "phosphorus_anion", "carbonyl"),
    ("benzene", "nitrate_ion", "silyl", "boride_ion", "phosphoryl",
     "trivalent_nitrogen", "ring_5", "oxygen_anion", "amide", "chloro",
     "perchlorate_chlorine", "nitrile", "cyclotriphosphazene", "sulfinyl",
     "ammonium_ion", "trigonal_borate", "carboxylate", "arsenide_ion"),
)


def analyze_formulation_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    tokenizer = Tokenizer()
    molecule_cache = {}
    properties = []
    moieties = Counter()
    for column in SOLVENT_COLUMNS + SALT_COLUMNS:
        for value in dataset[column].dropna():
            smiles = str(value).split(":", 1)[0].strip()
            if smiles not in molecule_cache:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES in {csv_path}: {smiles}")
                canonical = Chem.MolToSmiles(mol)
                names = {name for name, _ in tokenizer.get_moieties(canonical)[3]}
                counts = (mol.GetNumHeavyAtoms(),
                          sum(atom.GetIsAromatic() for atom in mol.GetAtoms()),
                          mol.GetRingInfo().NumRings())
                molecule_cache[smiles] = (counts, names)
            counts, names = molecule_cache[smiles]
            properties.append(counts)
            moieties.update(names)
    return np.asarray(properties), moieties


def plot_molecule_properties(datasets, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ("Number of Heavy Atoms", "Number of Aromatic Atoms", "Number of Rings")
    for index, (ax, label) in enumerate(zip(axes, labels)):
        maximum = max(properties[:, index].max() for properties, _ in datasets)
        bins = np.arange(-0.5, maximum + 1.5)
        # Draw the smaller historical dataset on top, as in the paper.
        for source in (1, 0):
            ax.hist(datasets[source][0][:, index], bins=bins,
                    color=COLORS[source], alpha=0.6, label=LABELS[source])
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], legend_labels[::-1], fontsize=10)
        ax.set_xlabel(label, fontsize=14)
        ax.tick_params(labelsize=11)
        if index == 2:
            ax.set_xticks(np.arange(maximum + 1))
    axes[0].set_ylabel("Frequency", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "molecule_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_moiety_frequencies(datasets, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 3]})
    for ax, names in zip(axes, PANEL_MOIETIES):
        positions = np.arange(len(names))
        for index, (_, frequencies) in enumerate(datasets):
            ax.bar(positions + (index - 0.5) * 0.38,
                   [frequencies[name] for name in names], width=0.38,
                   color=COLORS[index], label=LABELS[index])
        ax.set_xticks(positions, names, rotation=90, fontsize=10)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Frequency", fontsize=14)
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "moiety_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    datasets = [analyze_formulation_dataset(HERE / name)
                for name in ("ce_dataset_2023.csv", "ce_dataset_2026.csv")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_molecule_properties(datasets, args.output_dir)
    plot_moiety_frequencies(datasets, args.output_dir)
    for label, (properties, _) in zip(LABELS, datasets):
        print(f"{label}: {len(properties)} molecule occurrences")
    print(f"Saved Figure 2 component plots to {args.output_dir}")


if __name__ == "__main__":
    main()
