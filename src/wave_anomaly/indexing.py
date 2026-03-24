from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .config import load_config, resolve_path
from .utils import save_json, seed_everything, write_csv


INDEX_FIELDS = [
    "sample_id",
    "split",
    "year",
    "cache_path",
    "label_path",
    "target_time",
    "history_start",
    "history_end",
    "history_start_index",
    "history_end_index",
    "target_index",
    "has_positive",
    "valid_ratio",
]


def cache_path_for_year(year: int, config: dict[str, Any]) -> Path:
    cache_dir = resolve_path(config["data"]["cache_dir"])
    return cache_dir / config["data"]["cache_filename_template"].format(year=year)


def label_path_for_year(year: int, config: dict[str, Any]) -> Path:
    label_dir = resolve_path(config["data"]["label_dir"])
    return label_dir / config["data"]["label_filename_template"].format(year=year)


def build_rows_for_cache(
    cache_path: Path,
    year: int,
    split: str,
    history_len: int,
    pred_offset: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    drop_unlabeled = bool(config["data"].get("drop_unlabeled_split_samples", True))
    with xr.open_dataset(cache_path) as ds:
        times = ds["time"].values
        labels = ds["label"].values
        quality = ds["quality_mask"].values.astype(np.float32)
        label_source = ds["label"].attrs.get("source_path", str(label_path_for_year(year, config)))

        for target_index in range(history_len - 1 + pred_offset, len(times)):
            history_end_index = target_index - pred_offset
            history_start_index = history_end_index - history_len + 1
            label_frame = labels[target_index]
            valid_label_mask = label_frame >= 0
            if drop_unlabeled and not np.any(valid_label_mask):
                continue
            has_positive = bool(np.any(label_frame == 1))
            valid_ratio = float(quality[history_start_index : history_end_index + 1].mean())
            row = {
                "sample_id": f"{split}-{year}-{target_index:04d}",
                "split": split,
                "year": year,
                "cache_path": str(cache_path),
                "label_path": label_source,
                "target_time": np.datetime_as_string(times[target_index], unit="s"),
                "history_start": np.datetime_as_string(times[history_start_index], unit="s"),
                "history_end": np.datetime_as_string(times[history_end_index], unit="s"),
                "history_start_index": history_start_index,
                "history_end_index": history_end_index,
                "target_index": target_index,
                "has_positive": int(has_positive),
                "valid_ratio": f"{valid_ratio:.6f}",
            }
            rows.append(row)
    return rows


def split_year_mapping(config: dict[str, Any]) -> dict[str, list[int]]:
    return {
        "train": list(config["data"]["train_years"]),
        "test": list(config["data"]["test_years"]),
        "val": list(config["data"]["val_years"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sliding-window indices from yearly caches.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    history_len = int(config["data"]["history_len"])
    pred_offset = int(config["data"]["pred_offset"])
    index_dir = resolve_path(config["data"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "test", "val")}
    manifest: dict[str, Any] = {"splits": {}, "history_len": history_len, "pred_offset": pred_offset}

    for split, years in split_year_mapping(config).items():
        missing_caches: list[str] = []
        for year in years:
            cache_path = cache_path_for_year(year, config)
            if not cache_path.exists():
                missing_caches.append(str(cache_path))
                continue
            rows = build_rows_for_cache(cache_path, year, split, history_len, pred_offset, config)
            split_rows[split].extend(rows)
        out_path = index_dir / f"{split}_index.csv"
        write_csv(out_path, split_rows[split], INDEX_FIELDS)
        manifest["splits"][split] = {
            "count": len(split_rows[split]),
            "path": str(out_path),
            "missing_caches": missing_caches,
        }
        print(f"[build_index] {split}: {len(split_rows[split])} rows -> {out_path}")

    all_rows = split_rows["train"] + split_rows["test"] + split_rows["val"]
    write_csv(index_dir / "all_index.csv", all_rows, INDEX_FIELDS)
    save_json(index_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
