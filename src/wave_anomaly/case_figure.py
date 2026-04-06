from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from wave_anomaly.config import load_config, resolve_path
from wave_anomaly.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a polished multi-timestep case figure.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--window", type=int, default=4, help="Number of consecutive timesteps to display.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional explicit start index in the prediction file.")
    parser.add_argument("--prediction-file", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "case_field",
        ["#3f2b96", "#315fd4", "#1f9ed7", "#31c7a0", "#a8db34", "#fde725"],
        N=256,
    )


def _diff_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "case_diff",
        ["#3459c7", "#61b7d9", "#f5f6f8", "#f7d26a", "#d85f43"],
        N=256,
    )


def _format_lon(value: float) -> str:
    return f"{value:.0f}E"


def _format_lat(value: float) -> str:
    return f"{value:.0f}N"


def _window_bbox(mask_window: np.ndarray) -> tuple[int, int, int, int] | None:
    union = mask_window.any(axis=0)
    if not np.any(union):
        return None
    ys, xs = np.where(union)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _select_window(labels: np.ndarray, probabilities: np.ndarray, window: int) -> int:
    label_area = (labels > 0.5).reshape(labels.shape[0], -1).sum(axis=1)
    best_score = None
    best_start = 0
    for start in range(0, labels.shape[0] - window + 1):
        label_window = labels[start : start + window] > 0.5
        area_sum = float(label_area[start : start + window].sum())
        if area_sum <= 0:
            continue
        bbox = _window_bbox(label_window)
        if bbox is None:
            continue
        y0, y1, x0, x1 = bbox
        bbox_area = float((y1 - y0 + 1) * (x1 - x0 + 1))
        peak_prob = float(np.nanmax(probabilities[start : start + window]))
        score = area_sum / max(bbox_area, 1.0) + 0.05 * peak_prob
        if best_score is None or score > best_score:
            best_score = score
            best_start = start
    if best_score is not None:
        return best_start

    peak_index = int(np.nanargmax(probabilities))
    return max(0, min(peak_index - window // 2, labels.shape[0] - window))


def _crop_slices(labels: np.ndarray, probabilities: np.ndarray) -> tuple[slice, slice]:
    base_mask = labels > 0.5
    bbox = _window_bbox(base_mask)
    if bbox is None:
        prob_threshold = max(float(np.nanmax(probabilities)) * 0.6, 0.3)
        bbox = _window_bbox(probabilities >= prob_threshold)
    if bbox is None:
        _, cy, cx = np.unravel_index(int(np.nanargmax(probabilities)), probabilities.shape)
        bbox = (cy, cy, cx, cx)

    y0, y1, x0, x1 = bbox
    span_y = y1 - y0 + 1
    span_x = x1 - x0 + 1
    pad_y = max(8, int(round(span_y * 0.8)))
    pad_x = max(10, int(round(span_x * 0.8)))
    target_h = max(span_y + 2 * pad_y, 28)
    target_w = max(span_x + 2 * pad_x, 34)
    center_y = (y0 + y1) // 2
    center_x = (x0 + x1) // 2

    full_h = labels.shape[1]
    full_w = labels.shape[2]

    y_start = max(0, center_y - target_h // 2)
    y_stop = min(full_h, y_start + target_h)
    y_start = max(0, y_stop - target_h)

    x_start = max(0, center_x - target_w // 2)
    x_stop = min(full_w, x_start + target_w)
    x_start = max(0, x_stop - target_w)

    return slice(y_start, y_stop), slice(x_start, x_stop)


def _plot_case(
    output_path: Path,
    times: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "#faf8f4",
            "axes.facecolor": "#f4f1eb",
            "axes.edgecolor": "#888888",
        }
    )

    difference = probabilities - labels
    window = labels.shape[0]
    fig, axes = plt.subplots(window, 3, figsize=(11, 3.1 * window), sharex=True, sharey=True)
    if window == 1:
        axes = np.asarray([axes])

    field_cmap = _field_cmap()
    diff_cmap = _diff_cmap()
    diff_limit = float(np.nanpercentile(np.abs(difference), 99))
    diff_limit = min(1.0, max(0.25, diff_limit))
    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    column_titles = ["Ground Truth", "Predicted Probability", "Difference (Pred - Truth)"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    last_field = None
    last_diff = None
    for row in range(window):
        label = labels[row]
        prob = probabilities[row]
        diff = difference[row]
        timestamp = np.datetime_as_string(times[row], unit="h").replace("T", " ")

        panels = [label, prob, diff]
        cmaps = [field_cmap, field_cmap, diff_cmap]
        norms = [None, None, diff_norm]
        vmins = [0.0, 0.0, None]
        vmaxs = [1.0, 1.0, None]

        for col in range(3):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                lon_grid,
                lat_grid,
                panels[col],
                shading="auto",
                cmap=cmaps[col],
                norm=norms[col],
                vmin=vmins[col],
                vmax=vmaxs[col],
            )
            ax.set_aspect("equal")
            ax.contour(longitudes, latitudes, label, levels=[0.5], colors="#ffffff", linewidths=0.8, alpha=0.9)
            ax.grid(color="#ffffff", linewidth=0.45, alpha=0.28)
            ax.text(
                0.03,
                0.96,
                letters[row * 3 + col],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                weight="bold",
                color="#222222",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.86},
            )
            if row == 0:
                ax.set_title(column_titles[col], pad=8, weight="semibold")
            if col == 0:
                ax.text(
                    -0.23,
                    0.5,
                    timestamp,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#333333",
                    weight="semibold",
                )

            if col < 2:
                last_field = mesh
            else:
                last_diff = mesh

    for row in range(window):
        for col in range(3):
            ax = axes[row, col]
            if row == window - 1:
                x_ticks = np.linspace(longitudes[0], longitudes[-1], 4)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([_format_lon(x) for x in x_ticks])
            else:
                ax.set_xticklabels([])
            if col == 0:
                y_ticks = np.linspace(latitudes[0], latitudes[-1], 4)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([_format_lat(y) for y in y_ticks])
            else:
                ax.set_yticklabels([])

    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.12, hspace=0.08, wspace=0.12)
    fig.suptitle("Sequential Typhoon Case: Truth, Prediction and Difference", y=0.975, fontsize=14, weight="bold")

    left_box = axes[-1, 0].get_position()
    middle_box = axes[-1, 1].get_position()
    right_box = axes[-1, 2].get_position()

    cax_field = fig.add_axes([left_box.x0 + 0.015, 0.065, middle_box.x1 - left_box.x0 - 0.03, 0.014])
    cbar_field = fig.colorbar(
        last_field,
        cax=cax_field,
        orientation="horizontal",
    )
    cbar_field.set_label("Truth / Prediction Scale")

    cax_diff = fig.add_axes([right_box.x0 + 0.02, 0.065, right_box.x1 - right_box.x0 - 0.04, 0.014])
    cbar_diff = fig.colorbar(
        last_diff,
        cax=cax_diff,
        orientation="horizontal",
    )
    cbar_diff.set_label("Prediction - Truth")

    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    prediction_dir = resolve_path(config["data"]["prediction_dir"], base_dir=config_path.parent.parent)
    report_dir = ensure_dir(resolve_path(config["eval"]["report_dir"], base_dir=config_path.parent.parent))
    figure_dir = ensure_dir(report_dir / "figures")
    prediction_path = resolve_path(
        args.prediction_file if args.prediction_file else prediction_dir / f"prediction_{args.year}.nc",
        base_dir=config_path.parent.parent,
    )
    output_path = resolve_path(
        args.output if args.output else figure_dir / f"case_sequence_{args.year}.png",
        base_dir=config_path.parent.parent,
    )
    ensure_dir(output_path.parent)

    with xr.open_dataset(prediction_path) as ds:
        labels = np.asarray(ds["label"].values, dtype=np.float32)
        probabilities = np.asarray(ds["probability"].values, dtype=np.float32)
        times = np.asarray(ds["time"].values)
        latitudes = np.asarray(ds["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(ds["longitude"].values, dtype=np.float32)

    start_index = args.start_index
    if start_index is None:
        start_index = _select_window(labels, probabilities, args.window)
    start_index = max(0, min(int(start_index), labels.shape[0] - args.window))
    end_index = start_index + args.window

    label_window = labels[start_index:end_index]
    prob_window = probabilities[start_index:end_index]
    time_window = times[start_index:end_index]
    y_slice, x_slice = _crop_slices(label_window, prob_window)

    _plot_case(
        output_path=output_path,
        times=time_window,
        latitudes=latitudes[y_slice],
        longitudes=longitudes[x_slice],
        labels=label_window[:, y_slice, x_slice],
        probabilities=prob_window[:, y_slice, x_slice],
    )

    print(
        "[case] saved figure="
        f"{output_path} start_index={start_index} "
        f"time_start={np.datetime_as_string(time_window[0], unit='h')} "
        f"time_end={np.datetime_as_string(time_window[-1], unit='h')}"
    )


if __name__ == "__main__":
    main()
