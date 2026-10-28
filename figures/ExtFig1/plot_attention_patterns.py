"""Regenerate both ExtFig1 heatmaps and a labelled, stacked 600-DPI PNG.

Inputs: data/HELENA-CE-26.csv and data/helena_ce_trained_weights.pth.
Uses the saved generator's selection and styling; no model training is performed.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def create_stacked_png(panel_a: Path, panel_b: Path, output_path: Path):
    """Stack a above b at equal widths, preserving each panel's aspect ratio."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    if output_path.exists():
        raise FileExistsError(f"Existing figures are never overwritten: {output_path}")

    paths = (panel_a, panel_b)
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)

    dpi = 600
    pixels_per_point = dpi / 72
    panel_width = min(width for width, _ in sizes)
    heights = [round(height * panel_width / width) for width, height in sizes]
    margin = round(14.4 * pixels_per_point)
    label_height = round(42 * pixels_per_point)
    gap = round(6 * pixels_per_point)
    label_baseline = round(31 * pixels_per_point)
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=["Arial", "DejaVu Sans"], weight="bold")
    )
    font = ImageFont.truetype(font_path, round(40 * pixels_per_point))

    with Image.new(
        "RGB",
        (panel_width + 2 * margin,
         sum(heights) + 2 * label_height + gap + 2 * margin),
        "white",
    ) as combined:
        draw = ImageDraw.Draw(combined)
        top = margin
        for label, path, height in zip(("a", "b"), paths, heights):
            draw.text(
                (margin, top + label_baseline), label,
                fill="black", font=font, anchor="ls",
            )
            with Image.open(path) as source:
                panel = source.convert("RGBA")
            try:
                if panel.size != (panel_width, height):
                    resized = panel.resize((panel_width, height), Image.Resampling.LANCZOS)
                    panel.close()
                    panel = resized
                combined.paste(panel, (margin, top + label_height), panel)
            finally:
                panel.close()
            top += label_height + height + gap
        combined.save(output_path, dpi=(dpi, dpi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/extfig1")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir; existing figures are never overwritten.")

    from tasks.postprocess.moiety_analysis import moiety_encod_att_heatmaps as plots

    args.output_dir.mkdir(parents=True)
    plots.MOIETY_HEATMAPS_SUMMARY_DIR = str(args.output_dir)
    analyzer = plots.MoietyEncoderAttentionHeatmaps()
    results = analyzer.analyze_extreme_lce_formulations(
        n_top=plots.summary_best_molecules, n_bottom=0,
        results_dir=str(args.output_dir),
    )
    if results.empty:
        raise ValueError("The original generator produced no attention data.")
    analyzer.create_summary_heatmaps(
        results, excluded_moieties=plots.EXCLUDED_MOIETIES, performances=("highest",)
    )
    analyzer.create_all_moieties_summary_heatmaps(
        results, excluded_moieties=plots.EXCLUDED_MOIETIES, performances=("highest",)
    )
    for original, published in [
        ("summary_heatmap_highest.png", "extended_data_Fig_1a.png"),
        ("summary_heatmap_all_moieties_highest_filtered.png", "extended_data_Fig_1b.png"),
    ]:
        (args.output_dir / original).rename(args.output_dir / published)
    create_stacked_png(
        args.output_dir / "extended_data_Fig_1a.png",
        args.output_dir / "extended_data_Fig_1b.png",
        args.output_dir / "extended_data_Fig_1.png",
    )


if __name__ == "__main__":
    main()
