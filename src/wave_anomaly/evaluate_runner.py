from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from .cache import cache_path_for_year, open_cache
from .config import load_config, resolve_path
from .metrics import StreamingPixelMetrics, merge_metric_dicts, object_metrics
from .utils import ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction NetCDF files.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["test", "val"], required=True)
    return parser.parse_args()


def years_for_split(config: dict[str, Any], split: str) -> list[int]:
    return list(config["data"]["test_years"] if split == "test" else config["data"]["val_years"])


def evaluate_prediction_file(prediction_path, config: dict[str, Any], use_tqdm: bool = True) -> dict[str, Any]:
    thresholds = np.linspace(1.0, 0.0, int(config["eval"].get("pr_auc_thresholds", 201)))
    metric_acc = StreamingPixelMetrics(thresholds)
    object_rows: list[dict[str, float]] = []

    print(f"[evaluate] loading prediction file: {prediction_path}", flush=True)
    with xr.open_dataset(prediction_path) as ds:
        threshold = float(ds.attrs.get("threshold", config["eval"]["threshold"]))
        probabilities = ds["probability"].values
        labels = ds["label"].values
        valid_mask = (labels >= 0).astype(np.float32)
        clipped_labels = np.where(labels < 0, 0, labels)

        print(
            f"[evaluate] threshold scan on {probabilities.shape[0]} timesteps with {len(thresholds)} thresholds",
            flush=True,
        )
        metric_acc.update(clipped_labels, np.nan_to_num(probabilities, nan=0.0), valid_mask)

        progress = tqdm(
            range(probabilities.shape[0]),
            desc="object-metrics",
            dynamic_ncols=True,
            leave=False,
            disable=not use_tqdm,
        )
        for idx in progress:
            if not np.any(valid_mask[idx] > 0.5):
                continue
            if np.isnan(probabilities[idx]).all():
                continue
            object_rows.append(
                object_metrics(
                    y_true=clipped_labels[idx],
                    y_prob=np.nan_to_num(probabilities[idx], nan=0.0),
                    threshold=threshold,
                    connectivity=int(config["eval"]["connectivity"]),
                    min_area=int(config["eval"]["min_area"]),
                    valid_mask=valid_mask[idx],
                )
            )
        progress.close()

    summary = metric_acc.summary_at(threshold)
    summary["pr_auc"] = metric_acc.pr_auc()
    if object_rows:
        summary.update(merge_metric_dicts(object_rows))
    return {"summary": summary, "table": metric_acc.table()}


def save_threshold_table(output_dir, split: str, year: int, table: list[dict[str, float]]) -> None:
    csv_path = output_dir / f"{split}_{year}_threshold_table.csv"
    fieldnames = [
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "positive_hit_rate",
        "f1",
        "iou",
        "dice",
        "csi",
        "pod",
        "far",
        "accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            row_to_write = dict(row)
            row_to_write["positive_hit_rate"] = row["recall"]
            writer.writerow({key: row_to_write.get(key, "") for key in fieldnames})


def save_curves(output_dir, year: int, table: list[dict[str, float]]) -> None:
    thresholds = [row["threshold"] for row in table]
    precision = [row["precision"] for row in table]
    recall = [row["recall"] for row in table]
    f1 = [row["f1"] for row in table]

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve {year}")
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_curve_{year}.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(f"Threshold Scan {year}")
    plt.tight_layout()
    plt.savefig(output_dir / f"threshold_curve_{year}.png")
    plt.close()


def save_core_metrics_bar(output_dir: Path, year: int, summary: dict[str, Any]) -> None:
    metrics = {
        "Accuracy": float(summary.get("accuracy", 0.0)),
        "F1": float(summary.get("f1", 0.0)),
        "Dice": float(summary.get("dice", 0.0)),
        "IoU": float(summary.get("iou", 0.0)),
        "PR-AUC": float(summary.get("pr_auc", 0.0)),
        "object_pod": float(summary.get("object_pod", 0.0)),
    }
    plt.figure(figsize=(7, 4.5))
    plt.bar(list(metrics.keys()), list(metrics.values()), color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Core Metrics {year}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"core_metrics_bar_{year}.png")
    plt.close()


def _select_background_field(cache, time_index: int) -> tuple[np.ndarray, str]:
    wind_channels = list(cache.meta.get("wind_channels", []))
    wave_channels = list(cache.meta.get("wave_channels", []))
    if "ws" in wind_channels:
        idx = wind_channels.index("ws")
        return np.asarray(cache.wind[time_index, idx], dtype=np.float32), "Wind Speed"
    if "swh" in wave_channels:
        idx = wave_channels.index("swh")
        return np.asarray(cache.wave[time_index, idx], dtype=np.float32), "Wave Height"
    return np.asarray(cache.wind[time_index, 0], dtype=np.float32), "Wind Channel 0"


def _representative_index(labels: np.ndarray, binaries: np.ndarray) -> int:
    label_area = (labels > 0.5).reshape(labels.shape[0], -1).sum(axis=1)
    pred_area = (binaries > 0.5).reshape(binaries.shape[0], -1).sum(axis=1)
    if np.any(label_area > 0):
        return int(np.argmax(label_area))
    return int(np.argmax(pred_area))


def save_sample_panels(output_dir: Path, year: int, prediction_path: Path, config: dict[str, Any]) -> None:
    cache = open_cache(cache_path_for_year(year, config))
    try:
        with xr.open_dataset(prediction_path) as ds:
            probabilities = np.asarray(ds["probability"].values, dtype=np.float32)
            binaries = np.asarray(ds["binary_prediction"].values, dtype=np.int8)
            labels = np.asarray(ds["label"].values, dtype=np.int8)
            times = np.asarray(ds["time"].values)

        sample_idx = _representative_index(labels, binaries)
        background, field_name = _select_background_field(cache, sample_idx)
        label = labels[sample_idx]
        probability = probabilities[sample_idx]
        binary = binaries[sample_idx]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(background, origin="lower", cmap="viridis")
        axes[0, 0].set_title(field_name)
        axes[0, 1].imshow(label, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("Ground Truth")
        axes[1, 0].imshow(probability, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("Prediction Probability")
        axes[1, 1].imshow(binary, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title("Binary Prediction")
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Representative Sample {year} @ {np.datetime_as_string(times[sample_idx], unit='h')}")
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_four_panel_{year}.png")
        plt.close(fig)

        tp = (binary == 1) & (label == 1)
        fp = (binary == 1) & (label == 0)
        fn = (binary == 0) & (label == 1)
        error_rgb = np.ones((*label.shape, 3), dtype=np.float32)
        error_rgb[tp] = np.array([0.2, 0.7, 0.2], dtype=np.float32)
        error_rgb[fp] = np.array([0.9, 0.2, 0.2], dtype=np.float32)
        error_rgb[fn] = np.array([0.2, 0.4, 0.95], dtype=np.float32)
        plt.figure(figsize=(6, 5))
        plt.imshow(error_rgb, origin="lower")
        plt.title("Error Distribution (TP green / FP red / FN blue)")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(output_dir / f"error_distribution_{year}.png")
        plt.close()

        predicted_area = (binaries > 0.5).reshape(binaries.shape[0], -1).sum(axis=1)
        label_area = (labels > 0.5).reshape(labels.shape[0], -1).sum(axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(times, predicted_area, label="predicted_area", color="#E45756")
        plt.plot(times, label_area, label="label_area", color="#4C78A8")
        plt.xlabel("Time")
        plt.ylabel("Area (pixels)")
        plt.title(f"Area Time Series {year}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"area_timeseries_{year}.png")
        plt.close()
    finally:
        cache.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    prediction_dir = resolve_path(args.pred_dir or config["data"]["prediction_dir"])
    report_dir = ensure_dir(resolve_path(config["eval"]["report_dir"]))
    figure_dir = ensure_dir(report_dir / "figures")
    split_years = years_for_split(config, args.split)
    use_tqdm = bool(config["eval"].get("use_tqdm", True))
    aggregate_rows: list[dict[str, float]] = []

    csv_path = report_dir / f"{args.split}_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "year",
            "threshold",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "iou",
            "dice",
            "csi",
            "pod",
            "far",
            "object_csi",
            "object_pod",
            "object_far",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for year in split_years:
            prediction_path = prediction_dir / f"prediction_{year}.nc"
            if not prediction_path.exists():
                print(f"[evaluate] missing {prediction_path}, skip")
                continue
            print(f"[evaluate] start split={args.split} year={year}", flush=True)
            result = evaluate_prediction_file(prediction_path, config, use_tqdm=use_tqdm)
            summary = {"year": year, **result["summary"]}
            writer.writerow({key: summary.get(key, "") for key in fieldnames})
            aggregate_rows.append(result["summary"])
            save_json(report_dir / f"{args.split}_{year}_metrics.json", result)
            save_threshold_table(report_dir, args.split, year, result["table"])
            print(f"[evaluate] saving curves for year={year}", flush=True)
            save_curves(report_dir, year, result["table"])
            save_core_metrics_bar(figure_dir, year, result["summary"])
            save_sample_panels(figure_dir, year, prediction_path, config)
            print(f"[evaluate] year={year} file={prediction_path}")

    save_json(report_dir / f"{args.split}_aggregate.json", merge_metric_dicts(aggregate_rows))


if __name__ == "__main__":
    main()
