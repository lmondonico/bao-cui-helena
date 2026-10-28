"""Plot viscosity predictions, moiety frequencies, and publication distributions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.viscosity_paths import (  # noqa: E402
    DATASET_PATH, FIGURES_OUTPUT_DIR, MOLECULE_INDEX_PATH, PUBLICATIONS_PATH,
    REFERENCE_RESULTS_DIR, TOKENIZER_MAPPING_PATH,
)

FONT_DIRS = (
    Path.home() / ".local" / "share" / "fonts" / "msttcorefonts",
    Path("/usr/share/fonts/truetype/msttcorefonts"),
    Path("/Library/Fonts"),
    Path("C:/Windows/Fonts"),
)
ARIAL_FILES = (
    "Arial.ttf", "Arial_Bold.ttf", "Arial_Italic.ttf", "Arial_Bold_Italic.ttf",
    "arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf",
)

MOLECULE_COLUMNS = [f"Solvent_{index}" for index in range(1, 6)] + [
    f"Salt_{index}" for index in range(1, 4)
]
PARITY_COLOR = sns.color_palette("bright", 10)[1]
MOIETY_MAGENTA = "#D72483"
JOURNAL_GREEN = "#7EC151"
INK = "#252525"
RASTER_DPI = 900


def canonical_doi(value: Any) -> str:
    doi = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


def configure_matplotlib() -> None:
    for folder in FONT_DIRS:
        for name in ARIAL_FILES:
            if (folder / name).is_file():
                font_manager.fontManager.addfont(str(folder / name))
    font_manager.findfont("Arial", fallback_to_default=False)  # raises if Arial is not installed
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "axes.labelsize": 21,
            "axes.labelweight": "normal",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "axes.edgecolor": INK,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def clean_axes(axis: mpl.axes.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", which="major", length=5, width=1.1, color=INK)


def save_figure(figure: mpl.figure.Figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    figure.savefig(png, dpi=RASTER_DPI, pad_inches=0.11)
    plt.close(figure)
    return [png]


def render_parity_plot(output: Path, predictions: pd.DataFrame) -> list[Path]:
    member_columns = sorted(
        (column for column in predictions.columns if column.startswith("Member_")),
        key=lambda column: int(column.split("_")[1]),
    )

    # "Averaged Test" means one point per sample: the mean of that sample's
    # member predictions made while it was held out in its outer test fold.
    averaged_test = predictions[member_columns].mean(axis=1).to_numpy(float)
    persisted = predictions["Predicted_Log10_Viscosity"].to_numpy(float)
    if not np.allclose(averaged_test, persisted, rtol=1e-12, atol=1e-12):
        raise AssertionError("Predicted_Log10_Viscosity is not the row-wise mean of the member predictions")
    plot_data = pd.DataFrame(
        {
            "Truth": predictions["Actual_Log10_Viscosity"].to_numpy(float),
            "Prediction": averaged_test,
        }
    )

    # Equal axes on half-log boundaries that contain every point.
    combined = plot_data[["Truth", "Prediction"]].to_numpy()
    lower = math.floor((float(combined.min()) - 0.1) * 2.0) / 2.0
    upper = math.ceil((float(combined.max()) + 0.1) * 2.0) / 2.0

    with mpl.rc_context(mpl.rcParamsDefault):
        grid = sns.JointGrid(x="Truth", y="Prediction", data=plot_data, space=0)
        grid.plot_marginals(
            sns.kdeplot,
            fill=True,
            color=PARITY_COLOR,
            alpha=0.3,
            bw_adjust=0.75,
        )

        axis = grid.ax_joint
        axis.scatter(
            plot_data["Truth"],
            plot_data["Prediction"],
            edgecolors="None",
            color=PARITY_COLOR,
            alpha=0.4,
            s=28,
            label="Averaged Test",
            zorder=3,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.plot(
            [lower, upper],
            [lower, upper],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=5,
        )
        axis.set_xlabel("Ground Truth Log(Viscosity) (mPa*s)", fontsize=16)
        axis.set_ylabel("Predicted Log(Viscosity) (mPa*s)", fontsize=16)
        for spine in axis.spines.values():
            spine.set_visible(True)

        r2 = r2_score(plot_data["Truth"], plot_data["Prediction"])
        rmse = np.sqrt(
            mean_squared_error(plot_data["Truth"], plot_data["Prediction"])
        )
        span = upper - lower
        annotation_x = lower + 0.80 * span
        for y_fraction, text, fontsize in (
            (0.20, "HELENA", 20),
            (0.14, f"R² = {r2:.3f}", 16),
            (0.08, f"RMSE = {rmse:.3f}", 16),
        ):
            axis.text(
                annotation_x,
                lower + y_fraction * span,
                text,
                color=PARITY_COLOR,
                fontsize=fontsize,
                ha="center",
                va="center",
            )

        legend_handles = [
            mlines.Line2D(
                [],
                [],
                color=PARITY_COLOR,
                marker="o",
                linestyle="None",
                markersize=6,
                label="Averaged Test",
            )
        ]
        axis.legend(handles=legend_handles, frameon=False, fontsize=12)
        axis.tick_params(axis="both", labelsize=14)

        png = output / "helena_parity.png"
        grid.figure.savefig(png, dpi=RASTER_DPI, bbox_inches="tight")
        plt.close(grid.figure)
    return [png]


def parse_component_smiles(cell: Any) -> str | None:
    if pd.isna(cell) or not str(cell).strip():
        return None
    value = str(cell).strip()
    try:
        smiles, coefficient = value.rsplit(":", 1)
        parsed = float(coefficient)
    except ValueError as exc:
        raise ValueError(f"Invalid component cell {value!r}") from exc
    if not smiles or not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Invalid component cell {value!r}")
    return smiles


def build_moiety_frequency(model_data: pd.DataFrame) -> pd.DataFrame:
    """Every HELENA moiety token of every populated component slot in every row (no deduplication or ratio weighting)."""
    molecule_index = pd.read_csv(MOLECULE_INDEX_PATH)
    token_lists: dict[str, list[int]] = {}
    for row in molecule_index.itertuples(index=False):
        tokens = [int(value) for value in json.loads(row.Moiety_Token_IDs_JSON)]
        if not tokens or any(token <= 0 for token in tokens):
            raise ValueError(f"Invalid saved tokens for {row.SMILES}")
        token_lists[str(row.SMILES)] = tokens
    mapping = json.loads(TOKENIZER_MAPPING_PATH.read_text(encoding="utf-8"))
    token_names = {
        int(key): str(value) for key, value in mapping["key_moiety_map"].items()
    }

    token_counts: Counter[int] = Counter()
    molecule_occurrences: Counter[str] = Counter()
    for row in model_data.itertuples(index=False):
        for column in MOLECULE_COLUMNS:
            smiles = parse_component_smiles(getattr(row, column))
            if smiles is None:
                continue
            if smiles not in token_lists:
                raise KeyError(f"Dataset molecule missing from viscosity_molecule_index.csv: {smiles}")
            molecule_occurrences[smiles] += 1
            token_counts.update(token_lists[smiles])

    expected_occurrences = int(molecule_index["Rows_Using_Molecule"].sum())
    actual_occurrences = sum(molecule_occurrences.values())
    if actual_occurrences != expected_occurrences:
        raise AssertionError(
            f"Molecule occurrence audit failed: {actual_occurrences} != {expected_occurrences}"
        )
    missing_names = sorted(set(token_counts) - set(token_names))
    if missing_names:
        raise KeyError(f"Token IDs lack semantic names: {missing_names}")

    rows = [
        {
            "Token_ID": token,
            "Moiety": token_names[token],
            "Display_Name": token_names[token].replace("_", " "),
            "Occurrence_Count": count,
        }
        for token, count in token_counts.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["Occurrence_Count", "Token_ID"], ascending=[False, True])
        .reset_index(drop=True)
    )


def render_moiety_plot(output: Path, frequency: pd.DataFrame) -> list[Path]:
    ordered = frequency.sort_values("Occurrence_Count", ascending=False)
    # 15.8 in wide; widen in 5% steps (fonts, bar width and height unchanged)
    # until no two bar labels overlap.
    width = 15.8
    while True:
        figure, value_labels = draw_moiety_plot(ordered, width)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        extents = [label.get_window_extent(renderer) for label in value_labels]
        plt.close(figure)
        if not any(left.overlaps(right) for left, right in zip(extents, extents[1:])):
            break
        width = round(width * 1.05, 2)
        if width > 60:
            raise AssertionError("Moiety bar labels still overlap at 60 in width")
    figure, _ = draw_moiety_plot(ordered, width)
    return save_figure(figure, output / "moiety_frequency")


def draw_moiety_plot(ordered: pd.DataFrame, width: float) -> tuple[plt.Figure, list[Any]]:
    figure, axis = plt.subplots(figsize=(width, 9.2), constrained_layout=True)
    bars = axis.bar(
        ordered["Display_Name"],
        ordered["Occurrence_Count"] - 0.8,
        bottom=0.8,
        color=MOIETY_MAGENTA,
        edgecolor="white",
        linewidth=0.55,
        width=0.76,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Frequency", labelpad=12)
    axis.set_yscale("log")
    axis.set_ylim(0.8, float(ordered["Occurrence_Count"].max()) * 2.0)
    axis.tick_params(axis="x", labelrotation=90)
    for label in axis.get_xticklabels():
        label.set_ha("center")
        label.set_va("top")
    value_labels = axis.bar_label(
        bars,
        labels=[f"{int(value):,}" for value in ordered["Occurrence_Count"]],
        padding=3,
        fontsize=11.5,
        color=INK,
    )
    clean_axes(axis)
    return figure, list(value_labels)


def load_publications(references: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(PUBLICATIONS_PATH)
    frame["DOI"] = frame["DOI"].map(canonical_doi)
    if set(frame["DOI"]) != set(references) or frame["DOI"].duplicated().any():
        raise ValueError("publications.csv must list every dataset DOI exactly once")
    frame["Published_Month"] = pd.to_numeric(frame["Published_Month"], errors="raise").astype(int)
    return frame


def build_publication_frequencies(
    model_data: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = model_data[["Row_ID", "Reference"]].copy()
    frame["DOI"] = frame["Reference"].map(canonical_doi)
    merged = frame.merge(metadata, on="DOI", how="left", validate="many_to_one")
    if len(merged) != len(model_data):
        raise AssertionError("Publication metadata join changed row count")
    required = ["Journal", "Published_Year", "Published_Month"]
    if merged[required].isna().any().any():
        raise AssertionError("Publication metadata join left missing values")

    journal = (
        merged.groupby("Journal", as_index=False)
        .agg(
            Measurement_Count=("Row_ID", "size"),
            Unique_Paper_Count=("DOI", "nunique"),
        )
        .sort_values(
            ["Unique_Paper_Count", "Measurement_Count", "Journal"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    paper_dates = metadata[["DOI", "Published_Year"]].copy()
    year_index = np.arange(
        int(paper_dates["Published_Year"].min()),
        int(paper_dates["Published_Year"].max()) + 1,
    )
    paper_counts = paper_dates.groupby("Published_Year")["DOI"].nunique()
    year = pd.DataFrame(
        {
            "Publication_Year": year_index,
            "Unique_Paper_Count": paper_counts.reindex(year_index, fill_value=0).to_numpy(int),
        }
    )
    return journal, year


def render_publication_plot(
    output: Path, journal: pd.DataFrame, year: pd.DataFrame
) -> list[Path]:
    top = journal.head(15).sort_values("Unique_Paper_Count", ascending=True)
    figure = plt.figure(figsize=(17.8, 11.4))
    journal_axis = figure.add_axes([0.30, 0.325, 0.675, 0.655])
    year_axis = figure.add_axes([0.015, 0.035, 0.96, 0.20])

    journal_axis.barh(
        top["Journal"],
        top["Unique_Paper_Count"],
        color=JOURNAL_GREEN,
        edgecolor="white",
        linewidth=0.65,
        height=0.72,
    )
    journal_axis.set_xlabel("Unique papers", labelpad=12, fontsize=27)
    journal_axis.set_ylabel("Journal", labelpad=12, fontsize=27)
    journal_axis.set_xlim(0, float(top["Unique_Paper_Count"].max()) * 1.18)
    clean_axes(journal_axis)

    years = year["Publication_Year"].to_numpy(int)
    counts = year["Unique_Paper_Count"].to_numpy(int)
    color_map = mpl.colormaps["magma_r"]
    normalization = mpl.colors.Normalize(vmin=0, vmax=max(int(counts.max()), 1))
    for index, (publication_year, count) in enumerate(zip(years, counts, strict=True)):
        color = color_map(normalization(count))
        red, green, blue = mpl.colors.to_rgb(color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text_color = INK if luminance > 0.58 else "white"
        year_axis.add_patch(
            FancyBboxPatch(
                (index - 0.41, 0.10),
                0.82,
                0.82,
                boxstyle="round,pad=0.015,rounding_size=0.08",
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
            )
        )
        year_axis.text(
            index,
            0.51,
            f"{count:,}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=text_color,
        )
        year_axis.text(
            index,
            -0.10,
            str(publication_year),
            ha="center",
            va="top",
            fontsize=17,
            color=INK,
        )
    year_axis.set_xlim(-0.5, len(years) - 0.5)
    year_axis.set_ylim(-0.28, 1.03)
    year_axis.set_aspect("equal", adjustable="box")
    year_axis.axis("off")
    return save_figure(figure, output / "publication_distribution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REFERENCE_RESULTS_DIR,
                        help="directory containing predictions.csv (input only)")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_OUTPUT_DIR,
                        help="new directory for the rendered figures")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir; existing figures are never overwritten.")
    configure_matplotlib()
    model_data = pd.read_csv(DATASET_PATH)
    predictions = pd.read_csv(args.results / "predictions.csv")
    references = sorted({canonical_doi(value) for value in model_data["Reference"]})
    args.output_dir.mkdir(parents=True)
    written = render_parity_plot(args.output_dir, predictions)
    written += render_moiety_plot(args.output_dir, build_moiety_frequency(model_data))
    written += render_publication_plot(args.output_dir, *build_publication_frequencies(model_data, load_publications(references)))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
