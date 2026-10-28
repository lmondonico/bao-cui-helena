"""Plot Figure 3 held-out ensemble predictions from the bundled publication results."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figures.Fig3.plot_fig3 import load_comparisons, plot_parities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    predictions, _, checks = load_comparisons()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_parities(predictions, checks, args.output_dir)


if __name__ == "__main__":
    main()
