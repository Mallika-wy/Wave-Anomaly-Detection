from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .config import resolve_path
from .utils import ensure_dir, load_json, save_json


META_FILENAME = "meta.json"
ARRAY_FILENAMES = {
    "wind": "wind.npy",
    "wave": "wave.npy",
    "label": "label.npy",
    "label_soft": "label_soft.npy",
    "quality_mask": "quality_mask.npy",
    "time": "time.npy",
    "latitude": "latitude.npy",
    "longitude": "longitude.npy",
}


def cache_path_for_year(year: int, config: dict[str, Any]) -> Path:
    cache_dir = resolve_path(config["data"]["cache_dir"])
    return cache_dir / config["data"]["cache_filename_template"].format(year=year)


@dataclass
class TrainReadyCache:
    root: Path
    mmap_mode: str | None = "r"
    meta: dict[str, Any] = field(init=False)
    _arrays: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.meta = load_json(self.root / META_FILENAME)

    def _array_path(self, key: str) -> Path:
        return self.root / ARRAY_FILENAMES[key]

    def _load_array(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = np.load(self._array_path(key), mmap_mode=self.mmap_mode, allow_pickle=False)
        return self._arrays[key]

    @property
    def wind(self) -> np.ndarray:
        return self._load_array("wind")

    @property
    def wave(self) -> np.ndarray:
        return self._load_array("wave")

    @property
    def label(self) -> np.ndarray:
        return self._load_array("label")

    @property
    def quality_mask(self) -> np.ndarray:
        return self._load_array("quality_mask")

    @property
    def label_soft(self) -> np.ndarray:
        path = self._array_path("label_soft")
        if not path.exists():
            return self.label
        return self._load_array("label_soft")

    @property
    def time(self) -> np.ndarray:
        return self._load_array("time")

    @property
    def latitude(self) -> np.ndarray:
        return self._load_array("latitude")

    @property
    def longitude(self) -> np.ndarray:
        return self._load_array("longitude")

    def close(self) -> None:
        self._arrays.clear()


def open_cache(cache_path: str | Path, mmap_mode: str | None = "r") -> TrainReadyCache:
    return TrainReadyCache(Path(cache_path), mmap_mode=mmap_mode)


def _write_numpy_array(path: Path, arr: xr.DataArray | np.ndarray, time_chunk_size: int) -> None:
    ensure_dir(path.parent)
    if isinstance(arr, xr.DataArray):
        shape = tuple(int(size) for size in arr.shape)
        target = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=shape)
        if arr.dims and arr.dims[0] == "time":
            for start in range(0, shape[0], time_chunk_size):
                end = min(start + time_chunk_size, shape[0])
                target[start:end] = np.asarray(arr.isel(time=slice(start, end)).values)
        else:
            target[...] = np.asarray(arr.values)
        del target
        return

    np.save(path, np.asarray(arr), allow_pickle=False)


def write_train_ready_cache(
    cache_path: str | Path,
    ds: xr.Dataset,
    year: int,
    config: dict[str, Any],
) -> Path:
    out_path = ensure_dir(cache_path)
    time_chunk_size = int(config["data"].get("cache_write_time_chunk", 64))

    _write_numpy_array(out_path / ARRAY_FILENAMES["wind"], ds["wind"], time_chunk_size)
    _write_numpy_array(out_path / ARRAY_FILENAMES["wave"], ds["wave"], time_chunk_size)
    _write_numpy_array(out_path / ARRAY_FILENAMES["label"], ds["label"], time_chunk_size)
    if "label_soft" in ds:
        _write_numpy_array(out_path / ARRAY_FILENAMES["label_soft"], ds["label_soft"], time_chunk_size)
    _write_numpy_array(out_path / ARRAY_FILENAMES["quality_mask"], ds["quality_mask"], time_chunk_size)
    np.save(out_path / ARRAY_FILENAMES["time"], np.asarray(ds["time"].values), allow_pickle=False)
    np.save(out_path / ARRAY_FILENAMES["latitude"], np.asarray(ds["latitude"].values), allow_pickle=False)
    np.save(out_path / ARRAY_FILENAMES["longitude"], np.asarray(ds["longitude"].values), allow_pickle=False)

    meta = {
        "format": "train_ready_v1",
        "year": int(year),
        "processed_grid": config["data"]["processed_grid"],
        "history_len": int(config["data"]["history_len"]),
        "wind_channels": ds["wind_channel"].values.tolist(),
        "wave_channels": ds["wave_channel"].values.tolist(),
        "label_source": ds["label"].attrs.get("source_path"),
        "shapes": {
            "wind": [int(size) for size in ds["wind"].shape],
            "wave": [int(size) for size in ds["wave"].shape],
            "label": [int(size) for size in ds["label"].shape],
            "label_soft": [int(size) for size in ds["label_soft"].shape] if "label_soft" in ds else None,
            "quality_mask": [int(size) for size in ds["quality_mask"].shape],
        },
    }
    save_json(out_path / META_FILENAME, meta)
    return out_path


def compute_stats(cache_paths: list[Path], chunk_size: int = 64) -> dict[str, Any]:
    accumulators = {"wind": None, "wave": None}
    channel_names = {"wind": [], "wave": []}

    for cache_path in cache_paths:
        cache = open_cache(cache_path)
        try:
            for key, names_key in (("wind", "wind_channels"), ("wave", "wave_channels")):
                arr = getattr(cache, key)
                if accumulators[key] is None:
                    channel_count = int(arr.shape[1])
                    accumulators[key] = {
                        "sum": np.zeros(channel_count, dtype=np.float64),
                        "sumsq": np.zeros(channel_count, dtype=np.float64),
                        "count": np.zeros(channel_count, dtype=np.float64),
                    }
                    channel_names[key] = list(cache.meta[names_key])

                for start in range(0, int(arr.shape[0]), chunk_size):
                    end = min(start + chunk_size, int(arr.shape[0]))
                    chunk = np.asarray(arr[start:end], dtype=np.float64)
                    accumulators[key]["sum"] += chunk.sum(axis=(0, 2, 3))
                    accumulators[key]["sumsq"] += np.square(chunk).sum(axis=(0, 2, 3))
                    accumulators[key]["count"] += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        finally:
            cache.close()

    stats: dict[str, Any] = {}
    for key in ("wind", "wave"):
        if accumulators[key] is None:
            raise RuntimeError(f"No cache files available to compute {key} statistics.")
        sums = accumulators[key]["sum"]
        sumsq = accumulators[key]["sumsq"]
        counts = np.maximum(accumulators[key]["count"], 1.0)
        means = sums / counts
        variances = np.maximum((sumsq / counts) - np.square(means), 1.0e-8)
        stats[key] = {
            "channel_names": channel_names[key],
            "mean": means.tolist(),
            "std": np.sqrt(variances).tolist(),
        }
    return stats
