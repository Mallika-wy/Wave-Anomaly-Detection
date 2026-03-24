from __future__ import annotations

import argparse
import csv
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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


def evaluate_prediction_file(prediction_path, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = np.linspace(1.0, 0.0, int(config["eval"].get("pr_auc_thresholds", 201)))
    metric_acc = StreamingPixelMetrics(thresholds)
    object_rows: list[dict[str, float]] = []

    with xr.open_dataset(prediction_path) as ds:
        threshold = float(ds.attrs.get("threshold", config["eval"]["threshold"]))
        probabilities = ds["probability"].values
        labels = ds["label"].values
        valid_mask = (labels >= 0).astype(np.float32)
        clipped_labels = np.where(labels < 0, 0, labels)
        metric_acc.update(clipped_labels, np.nan_to_num(probabilities, nan=0.0), valid_mask)

        for idx in range(probabilities.shape[0]):
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

    summary = metric_acc.summary_at(threshold)
    summary["pr_auc"] = metric_acc.pr_auc()
    if object_rows:
        summary.update(merge_metric_dicts(object_rows))
    return {"summary": summary, "table": metric_acc.table()}


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    prediction_dir = resolve_path(args.pred_dir or config["data"]["prediction_dir"])
    report_dir = ensure_dir(resolve_path(config["eval"]["report_dir"]))
    split_years = years_for_split(config, args.split)
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
            result = evaluate_prediction_file(prediction_path, config)
            summary = {"year": year, **result["summary"]}
            writer.writerow({key: summary.get(key, "") for key in fieldnames})
            aggregate_rows.append(result["summary"])
            save_json(report_dir / f"{args.split}_{year}_metrics.json", result)
            save_curves(report_dir, year, result["table"])
            print(f"[evaluate] year={year} file={prediction_path}")

    save_json(report_dir / f"{args.split}_aggregate.json", merge_metric_dicts(aggregate_rows))


if __name__ == "__main__":
    main()
