from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from .indexing import build_rows_for_cache, cache_path_for_year
from .model import build_model
from .utils import ensure_dir


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    threshold: float | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
        "threshold": threshold,
    }
    out_path = Path(path)
    ensure_dir(out_path.parent)
    torch.save(payload, out_path)


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def build_prediction_output_path(output_dir: str | Path, year: int) -> Path:
    return ensure_dir(output_dir) / f"prediction_{year}.nc"


def inference_rows_for_year(year: int, config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    cache_path = cache_path_for_year(year, config)
    rows = build_rows_for_cache(
        cache_path=cache_path,
        year=year,
        split="predict",
        history_len=int(config["data"]["history_len"]),
        pred_offset=int(config["data"]["pred_offset"]),
        config={**config, "data": {**config["data"], "drop_unlabeled_split_samples": False}},
    )
    return cache_path, rows


def predict_year(
    year: int,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    cache_path: Path,
    output_dir: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    with xr.open_dataset(cache_path) as cache_ds:
        times = cache_ds["time"].values
        latitude = cache_ds["latitude"].values
        longitude = cache_ds["longitude"].values
        labels = cache_ds["label"].values.astype(np.int8)
        probabilities = np.full(labels.shape, np.nan, dtype=np.float32)
        binaries = np.full(labels.shape, -1, dtype=np.int8)

    summary_rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        x_wind, x_wave, y, meta = dataset[idx]
        target_index = int(dataset.rows[idx]["target_index"])
        with torch.no_grad():
            prob = model(
                x_wind.unsqueeze(0).to(device),
                x_wave.unsqueeze(0).to(device),
                return_logits=False,
            ).squeeze(0).squeeze(0).cpu().numpy()
        binary = (prob >= threshold).astype(np.int8)
        probabilities[target_index] = prob
        binaries[target_index] = binary
        summary_rows.append(
            {
                "sample_id": meta["sample_id"],
                "year": meta["year"],
                "target_time": meta["target_time"],
                "max_prob": float(np.nanmax(prob)),
                "mean_prob": float(np.nanmean(prob)),
                "predicted_area": int(binary.sum()),
                "label_area": int(y.numpy().sum()),
            }
        )

    ds = xr.Dataset(
        {
            "probability": (("time", "latitude", "longitude"), probabilities),
            "binary_prediction": (("time", "latitude", "longitude"), binaries),
            "label": (("time", "latitude", "longitude"), labels),
        },
        coords={"time": times, "latitude": latitude, "longitude": longitude},
        attrs={"year": year, "threshold": threshold},
    )
    out_path = build_prediction_output_path(output_dir, year)
    encoding = {
        "probability": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "binary_prediction": {"zlib": True, "complevel": 4, "dtype": "int8"},
        "label": {"zlib": True, "complevel": 4, "dtype": "int8"},
    }
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path, summary_rows
