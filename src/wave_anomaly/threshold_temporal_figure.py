from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from wave_anomaly.config import load_config, resolve_path
from wave_anomaly.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot threshold scan and temporal summary figures.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, choices=["val", "test"], required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--summary-file", type=str, default=None)
    parser.add_argument("--threshold-table", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _setup_axis(ax) -> None:
    ax.set_facecolor("#f7f4ee")
    ax.grid(color="#ffffff", linewidth=0.9, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#c9c1b7")
        spine.set_linewidth(1.0)


def _date_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _plot_threshold_scan(ax, table: pd.DataFrame) -> None:
    table = table.sort_values("threshold")
    colors = {
        "f1": "#e45756",
        "precision": "#4c78a8",
        "recall": "#54a24b",
        "dice": "#f2a541",
    }
    labels = {
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "dice": "Dice",
    }
    for key in ["f1", "precision", "recall", "dice"]:
        ax.plot(table["threshold"], table[key], color=colors[key], linewidth=2.1, label=labels[key])

    best_row = table.loc[table["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])
    best_f1 = float(best_row["f1"])
    ax.axvline(best_threshold, color="#333333", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.scatter([best_threshold], [best_f1], color=colors["f1"], s=36, zorder=5)
    ax.annotate(
        f"best F1={best_f1:.3f}\nthreshold={best_threshold:.3f}",
        xy=(best_threshold, best_f1),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffaf2", "edgecolor": "#d8cdbd"},
        arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.8},
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Scan", weight="semibold")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    _setup_axis(ax)


def _rolling(series: pd.Series, window: int = 8) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _plot_probability_timeseries(ax, summary: pd.DataFrame) -> None:
    times = pd.to_datetime(summary["target_time"])
    max_prob = summary["max_prob"].astype(float)
    mean_prob = summary["mean_prob"].astype(float)

    ax_right = ax.twinx()

    ax.plot(times, max_prob, color="#dd8452", linewidth=0.8, alpha=0.20)
    ax.plot(times, _rolling(max_prob), color="#dd8452", linewidth=2.3, label="max_prob")
    ax_right.plot(times, mean_prob, color="#4c78a8", linewidth=0.8, alpha=0.25)
    ax_right.plot(times, _rolling(mean_prob), color="#4c78a8", linewidth=2.3, label="mean_prob")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Max Probability", color="#dd8452")
    ax.tick_params(axis="y", colors="#dd8452")
    mean_upper = max(float(mean_prob.max()) * 1.15, 0.02)
    ax_right.set_ylim(0.0, mean_upper)
    ax_right.set_ylabel("Mean Probability", color="#4c78a8")
    ax_right.tick_params(axis="y", colors="#4c78a8")
    ax.set_title("Probability Over Time", weight="semibold")
    _setup_axis(ax)
    ax_right.grid(False)
    ax_right.spines["right"].set_color("#c9c1b7")
    ax_right.spines["right"].set_linewidth(1.0)
    _date_axis(ax)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", frameon=False)


def _plot_area_timeseries(ax, summary: pd.DataFrame) -> None:
    times = pd.to_datetime(summary["target_time"])
    predicted_area = summary["predicted_area"].astype(float)
    label_area = summary["label_area"].astype(float)

    ax.fill_between(times, label_area, color="#4c78a8", alpha=0.12)
    ax.fill_between(times, predicted_area, color="#e45756", alpha=0.10)
    ax.plot(times, label_area, color="#4c78a8", linewidth=0.9, alpha=0.25)
    ax.plot(times, predicted_area, color="#e45756", linewidth=0.9, alpha=0.25)
    ax.plot(times, _rolling(label_area, window=12), color="#4c78a8", linewidth=2.4, label="label_area")
    ax.plot(times, _rolling(predicted_area, window=12), color="#e45756", linewidth=2.4, label="predicted_area")
    ax.set_ylabel("Area (pixels)")
    ax.set_title("Affected Area Over Time", weight="semibold")
    ax.legend(loc="upper right", frameon=False)
    _setup_axis(ax)
    _date_axis(ax)


def plot_dashboard(output_path: Path, summary: pd.DataFrame, threshold_table: pd.DataFrame, split: str, year: int) -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#fbfaf7",
        }
    )

    fig = plt.figure(figsize=(13, 9), facecolor="#fbfaf7")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.18)

    ax_threshold = fig.add_subplot(grid[0, 0])
    ax_probability = fig.add_subplot(grid[0, 1])
    ax_area = fig.add_subplot(grid[1, :])

    _plot_threshold_scan(ax_threshold, threshold_table)
    _plot_probability_timeseries(ax_probability, summary)
    _plot_area_timeseries(ax_area, summary)

    fig.suptitle(f"Threshold And Temporal Results ({split.upper()} {year})", fontsize=16, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Data source: threshold table from evaluation and prediction summary exported during inference.",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base_dir = config_path.parent.parent

    prediction_dir = resolve_path(config["data"]["prediction_dir"], base_dir=base_dir)
    report_dir = ensure_dir(resolve_path(config["eval"]["report_dir"], base_dir=base_dir))
    figure_dir = ensure_dir(report_dir / "figures")

    summary_path = resolve_path(
        args.summary_file if args.summary_file else prediction_dir / f"prediction_{args.year}_summary.csv",
        base_dir=base_dir,
    )
    threshold_path = resolve_path(
        args.threshold_table if args.threshold_table else report_dir / f"{args.split}_{args.year}_threshold_table.csv",
        base_dir=base_dir,
    )
    output_path = resolve_path(
        args.output if args.output else figure_dir / f"threshold_temporal_dashboard_{args.year}.png",
        base_dir=base_dir,
    )
    ensure_dir(output_path.parent)

    summary = pd.read_csv(summary_path)
    threshold_table = pd.read_csv(threshold_path)
    plot_dashboard(output_path, summary, threshold_table, args.split, args.year)
    print(f"[dashboard] saved {output_path}")


if __name__ == "__main__":
    main()
