from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .cache import TrainReadyCache
from .utils import load_json, read_csv


class WaveAnomalyDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path | None = None,
        rows: list[dict[str, Any]] | None = None,
        stats_path: str | Path | None = None,
    ) -> None:
        if rows is None and index_path is None:
            raise ValueError("Either index_path or rows must be provided.")
        self.rows = rows if rows is not None else read_csv(index_path)  # type: ignore[arg-type]
        self.stats = load_json(stats_path) if stats_path is not None else None
        self._caches: dict[str, TrainReadyCache] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_cache(self, cache_path: str) -> TrainReadyCache:
        if cache_path not in self._caches:
            self._caches[cache_path] = TrainReadyCache(Path(cache_path))
        return self._caches[cache_path]

    def _normalize(self, arr: np.ndarray, branch: str) -> np.ndarray:
        if self.stats is None:
            return arr
        mean = np.asarray(self.stats[branch]["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(self.stats[branch]["std"], dtype=np.float32)[:, None, None]
        return (arr - mean) / np.maximum(std, 1.0e-6)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        cache = self._get_cache(row["cache_path"])

        start_idx = int(row["history_start_index"])
        end_idx = int(row["history_end_index"])
        target_idx = int(row["target_index"])

        wind = np.array(cache.wind[start_idx : end_idx + 1], dtype=np.float32, copy=True)
        wave = np.array(cache.wave[start_idx : end_idx + 1], dtype=np.float32, copy=True)
        label = np.array(cache.label[target_idx], dtype=np.float32, copy=True)
        loss_mask = (label >= 0).astype(np.float32)
        label = np.where(label < 0, 0.0, label)

        wind = self._normalize(wind, "wind")
        wave = self._normalize(wave, "wave")

        meta = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "year": int(row["year"]),
            "target_time": row["target_time"],
            "loss_mask": torch.from_numpy(loss_mask[None, ...]),
        }

        return (
            torch.from_numpy(wind),
            torch.from_numpy(wave),
            torch.from_numpy(label[None, ...]),
            meta,
        )

    def close(self) -> None:
        for cache in self._caches.values():
            cache.close()
        self._caches.clear()


def build_weighted_sampler(
    rows: list[dict[str, Any]],
    positive_weight: float,
    negative_weight: float,
) -> WeightedRandomSampler:
    weights = [
        positive_weight if int(row["has_positive"]) == 1 else negative_weight
        for row in rows
    ]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
