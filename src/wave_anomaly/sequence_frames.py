from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from wave_anomaly.cache import cache_path_for_year, open_cache
from wave_anomaly.config import load_config, resolve_path
from wave_anomaly.utils import ensure_dir, save_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export history wind fields and prediction frames for a case.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--target-time", type=str, default=None, help="Optional explicit target time, e.g. 2025-11-10T12:00:00.")
    parser.add_argument("--target-index", type=int, default=None, help="Optional explicit target index in prediction_YYYY.nc.")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "history_field",
        ["#3f2b96", "#315fd4", "#1f9ed7", "#31c7a0", "#a8db34", "#fde725"],
        N=256,
    )


def _prediction_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "prediction_field",
        ["#32237b", "#2e59d9", "#22a6d5", "#6fd36e", "#fde725"],
        N=256,
    )


def _wave_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "wave_field",
        ["#14213d", "#1f6f8b", "#22a6b3", "#7ed957", "#fde725"],
        N=256,
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _select_target_index(labels: np.ndarray, probabilities: np.ndarray, history_len: int) -> int:
    best_score = None
    best_index = history_len - 1
    for idx in range(history_len - 1, labels.shape[0]):
        label = labels[idx]
        bbox = _bbox_from_mask(label > 0.5)
        area = float((label > 0.5).sum())
        if bbox is None or area <= 0:
            continue
        y0, y1, x0, x1 = bbox
        bbox_area = float((y1 - y0 + 1) * (x1 - x0 + 1))
        score = (area / max(bbox_area, 1.0)) * np.sqrt(area)
        score += 0.02 * float(np.nanmax(probabilities[idx]))
        if best_score is None or score > best_score:
            best_score = score
            best_index = idx
    if best_score is not None:
        return best_index

    return int(np.nanargmax(probabilities.reshape(probabilities.shape[0], -1).max(axis=1)))


def _crop_slices(label: np.ndarray, probability: np.ndarray, full_shape: tuple[int, int]) -> tuple[slice, slice]:
    bbox = _bbox_from_mask((label > 0.5) | (probability > 0.5))
    if bbox is None:
        y, x = np.unravel_index(int(np.nanargmax(probability)), probability.shape)
        bbox = (y, y, x, x)

    y0, y1, x0, x1 = bbox
    span_y = y1 - y0 + 1
    span_x = x1 - x0 + 1
    pad_y = max(10, int(round(span_y * 0.9)))
    pad_x = max(12, int(round(span_x * 0.9)))
    target_h = max(36, span_y + 2 * pad_y)
    target_w = max(42, span_x + 2 * pad_x)
    center_y = (y0 + y1) // 2
    center_x = (x0 + x1) // 2
    full_h, full_w = full_shape

    y_start = max(0, center_y - target_h // 2)
    y_stop = min(full_h, y_start + target_h)
    y_start = max(0, y_stop - target_h)
    x_start = max(0, center_x - target_w // 2)
    x_stop = min(full_w, x_start + target_w)
    x_start = max(0, x_stop - target_w)
    return slice(y_start, y_stop), slice(x_start, x_stop)


def _find_wind_speed(cache) -> np.ndarray:
    channels = list(cache.meta.get("wind_channels", []))
    if "ws" in channels:
        idx = channels.index("ws")
        return np.asarray(cache.wind[:, idx], dtype=np.float32)

    u = np.asarray(cache.wind[:, 0], dtype=np.float32)
    v = np.asarray(cache.wind[:, 1], dtype=np.float32)
    return np.sqrt(np.square(u) + np.square(v))


def _find_wave_field(cache) -> tuple[np.ndarray, str]:
    channels = list(cache.meta.get("wave_channels", []))
    if "swh" in channels:
        idx = channels.index("swh")
        return np.asarray(cache.wave[:, idx], dtype=np.float32), "swh"

    return np.asarray(cache.wave[:, 0], dtype=np.float32), channels[0] if channels else "wave_channel_0"


def _save_clean_image(
    data: np.ndarray,
    output_path: Path,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(3.0, 3.6), facecolor="white")
    ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.01, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base_dir = config_path.parent.parent

    history_len = int(config["data"]["history_len"])
    prediction_dir = resolve_path(config["data"]["prediction_dir"], base_dir=base_dir)
    prediction_path = prediction_dir / f"prediction_{args.year}.nc"
    cache_path = cache_path_for_year(args.year, config)
    default_output_dir = Path("outputs/reports/sequence_frames") / str(args.year)
    output_dir = ensure_dir(resolve_path(args.output_dir or default_output_dir, base_dir=base_dir))

    with xr.open_dataset(prediction_path) as ds:
        probabilities = np.asarray(ds["probability"].values, dtype=np.float32)
        labels = np.asarray(ds["label"].values, dtype=np.float32)
        times = np.asarray(ds["time"].values)

    if args.target_index is not None:
        target_index = int(args.target_index)
    elif args.target_time is not None:
        requested = np.datetime64(pd.to_datetime(args.target_time))
        matches = np.where(times == requested)[0]
        if len(matches) == 0:
            raise ValueError(f"target_time {args.target_time} not found in prediction file.")
        target_index = int(matches[0])
    else:
        target_index = _select_target_index(labels, probabilities, history_len)

    if target_index < history_len - 1:
        raise ValueError(f"target_index {target_index} is too early for history_len={history_len}.")

    cache = open_cache(cache_path)
    try:
        wind_speed = _find_wind_speed(cache)
        wave_field, wave_channel = _find_wave_field(cache)
        history_start = target_index - history_len + 1
        history_end = target_index
        label_frame = labels[target_index]
        prob_frame = probabilities[target_index]
        y_slice, x_slice = _crop_slices(label_frame, prob_frame, label_frame.shape)

        wind_window = wind_speed[history_start : history_end + 1, y_slice, x_slice]
        wave_window = wave_field[history_start : history_end + 1, y_slice, x_slice]
        prob_crop = prob_frame[y_slice, x_slice]
        history_times = times[history_start : history_end + 1]
        target_time = times[target_index]

        wind_vmin = float(np.nanpercentile(wind_window, 2))
        wind_vmax = float(np.nanpercentile(wind_window, 98))
        wind_vmax = max(wind_vmax, wind_vmin + 1.0e-6)
        wave_vmin = float(np.nanpercentile(wave_window, 2))
        wave_vmax = float(np.nanpercentile(wave_window, 98))
        wave_vmax = max(wave_vmax, wave_vmin + 1.0e-6)

        rows: list[dict[str, Any]] = []
        field_cmap = _field_cmap()
        wave_cmap = _wave_cmap()
        prediction_cmap = _prediction_cmap()
        for step, (frame, frame_time) in enumerate(zip(wind_window, history_times), start=1):
            time_tag = np.datetime_as_string(frame_time, unit="h").replace(":", "").replace("-", "")
            path = output_dir / f"history_{step:02d}_{time_tag}.png"
            _save_clean_image(frame, path, field_cmap, wind_vmin, wind_vmax)
            rows.append(
                {
                    "frame_type": "history_wind",
                    "step": step,
                    "time": np.datetime_as_string(frame_time, unit="s"),
                    "path": str(path),
                }
            )

        for step, (frame, frame_time) in enumerate(zip(wave_window, history_times), start=1):
            time_tag = np.datetime_as_string(frame_time, unit="h").replace(":", "").replace("-", "")
            path = output_dir / f"wave_{step:02d}_{time_tag}.png"
            _save_clean_image(frame, path, wave_cmap, wave_vmin, wave_vmax)
            rows.append(
                {
                    "frame_type": "history_wave",
                    "step": step,
                    "time": np.datetime_as_string(frame_time, unit="s"),
                    "path": str(path),
                }
            )

        target_tag = np.datetime_as_string(target_time, unit="h").replace(":", "").replace("-", "")
        prediction_path_out = output_dir / f"prediction_{target_tag}.png"
        _save_clean_image(prob_crop, prediction_path_out, prediction_cmap, 0.0, 1.0)
        rows.append(
            {
                "frame_type": "prediction_probability",
                "step": 0,
                "time": np.datetime_as_string(target_time, unit="s"),
                "path": str(prediction_path_out),
            }
        )

        write_csv(
            output_dir / "frame_manifest.csv",
            rows,
            fieldnames=["frame_type", "step", "time", "path"],
        )
        save_json(
            output_dir / "metadata.json",
            {
                "year": args.year,
                "target_index": target_index,
                "target_time": np.datetime_as_string(target_time, unit="s"),
                "history_start_index": history_start,
                "history_end_index": history_end,
                "history_start_time": np.datetime_as_string(history_times[0], unit="s"),
                "history_end_time": np.datetime_as_string(history_times[-1], unit="s"),
                "crop": {
                    "y_start": y_slice.start,
                    "y_stop": y_slice.stop,
                    "x_start": x_slice.start,
                    "x_stop": x_slice.stop,
                },
                "wind_color_scale": {"vmin": wind_vmin, "vmax": wind_vmax},
                "wave_channel": wave_channel,
                "wave_color_scale": {"vmin": wave_vmin, "vmax": wave_vmax},
            },
        )
    finally:
        cache.close()

    print(
        "[sequence] saved "
        f"{history_len} history wind frames + {history_len} history wave frames + 1 prediction frame -> {output_dir} "
        f"(target_time={np.datetime_as_string(target_time, unit='s')})"
    )


if __name__ == "__main__":
    main()
