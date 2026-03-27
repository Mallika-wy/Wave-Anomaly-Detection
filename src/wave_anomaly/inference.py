from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .cache import cache_path_for_year, open_cache
from .indexing import build_rows_for_cache
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
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    cache_path: Path,
    output_dir: str | Path,
    use_tqdm: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    cache = open_cache(cache_path)
    try:
        times = np.asarray(cache.time)
        latitude = np.asarray(cache.latitude)
        longitude = np.asarray(cache.longitude)
        labels = np.asarray(cache.label, dtype=np.int8)
        probabilities = np.full(labels.shape, np.nan, dtype=np.float32)
        binaries = np.full(labels.shape, -1, dtype=np.int8)
    finally:
        cache.close()

    summary_rows: list[dict[str, Any]] = []
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"predict {year}",
        dynamic_ncols=True,
        leave=False,
        disable=not use_tqdm,
    )
    for x_wind, x_wave, y, meta in progress:
        with torch.no_grad():
            prob = model(
                x_wind.to(device, non_blocking=True),
                x_wave.to(device, non_blocking=True),
                return_logits=False,
            ).squeeze(1).cpu().numpy()
        binary = (prob >= threshold).astype(np.int8)
        target_indices = meta["target_index"].cpu().numpy().astype(np.int64)
        probabilities[target_indices] = prob
        binaries[target_indices] = binary
        for sample_idx, target_index in enumerate(target_indices):
            metric_label = meta["metric_label"][sample_idx].numpy()
            summary_rows.append(
                {
                    "sample_id": meta["sample_id"][sample_idx],
                    "year": int(meta["year"][sample_idx]),
                    "target_time": meta["target_time"][sample_idx],
                    "max_prob": float(np.nanmax(prob[sample_idx])),
                    "mean_prob": float(np.nanmean(prob[sample_idx])),
                    "predicted_area": int(binary[sample_idx].sum()),
                    "label_area": int(metric_label.sum()),
                }
            )
    progress.close()

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
    print(f"[predict] writing NetCDF -> {out_path}", flush=True)
    encoding = {
        "probability": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "binary_prediction": {"zlib": True, "complevel": 4, "dtype": "int8"},
        "label": {"zlib": True, "complevel": 4, "dtype": "int8"},
    }
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path, summary_rows
