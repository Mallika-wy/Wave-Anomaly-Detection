from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_thresholds(start: float, end: float, step: float) -> np.ndarray:
    count = int(round((end - start) / step)) + 1
    return np.round(np.linspace(start, end, count), 6)


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


def counts_to_metrics(counts: MetricCounts) -> dict[str, float]:
    tp = float(counts.tp)
    fp = float(counts.fp)
    fn = float(counts.fn)
    tn = float(counts.tn)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-8)
    iou = tp / max(tp + fp + fn, 1.0)
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    csi = iou
    pod = recall
    far = fp / max(tp + fp, 1.0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "csi": csi,
        "pod": pod,
        "far": far,
        "accuracy": accuracy,
    }


class StreamingPixelMetrics:
    def __init__(self, thresholds: np.ndarray) -> None:
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.counts = [MetricCounts() for _ in self.thresholds]

    def update(self, y_true: np.ndarray, y_prob: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        true_flat = y_true.reshape(-1).astype(np.float32)
        prob_flat = y_prob.reshape(-1).astype(np.float32)
        if valid_mask is not None:
            keep = valid_mask.reshape(-1) > 0.5
            true_flat = true_flat[keep]
            prob_flat = prob_flat[keep]
        if true_flat.size == 0:
            return
        true_bool = true_flat >= 0.5
        for idx, threshold in enumerate(self.thresholds):
            pred_bool = prob_flat >= threshold
            counts = self.counts[idx]
            counts.tp += int(np.sum(pred_bool & true_bool))
            counts.fp += int(np.sum(pred_bool & ~true_bool))
            counts.fn += int(np.sum(~pred_bool & true_bool))
            counts.tn += int(np.sum(~pred_bool & ~true_bool))

    def summary_at(self, threshold: float) -> dict[str, float]:
        idx = int(np.argmin(np.abs(self.thresholds - threshold)))
        return {"threshold": float(self.thresholds[idx]), **counts_to_metrics(self.counts[idx])}

    def table(self) -> list[dict[str, float]]:
        return [
            {"threshold": float(threshold), **counts_to_metrics(counts)}
            for threshold, counts in zip(self.thresholds, self.counts)
        ]

    def pr_auc(self) -> float:
        table = self.table()
        recall = np.asarray([row["recall"] for row in table], dtype=np.float64)
        precision = np.asarray([row["precision"] for row in table], dtype=np.float64)
        order = np.argsort(recall)
        return float(np.trapz(precision[order], recall[order]))

    def best_threshold(self, key: str = "f1") -> dict[str, float]:
        table = self.table()
        best = max(table, key=lambda row: (row[key], row["csi"], -abs(row["threshold"] - 0.5)))
        best["pr_auc"] = self.pr_auc()
        return best


def connected_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = mask.astype(bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    components: list[np.ndarray] = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cur_row, cur_col = stack.pop()
                pixels.append((cur_row, cur_col))
                for d_row, d_col in offsets:
                    nxt_row = cur_row + d_row
                    nxt_col = cur_col + d_col
                    if nxt_row < 0 or nxt_row >= height or nxt_col < 0 or nxt_col >= width:
                        continue
                    if visited[nxt_row, nxt_col] or not mask[nxt_row, nxt_col]:
                        continue
                    visited[nxt_row, nxt_col] = True
                    stack.append((nxt_row, nxt_col))
            component = np.zeros_like(mask, dtype=bool)
            rows, cols = zip(*pixels)
            component[list(rows), list(cols)] = True
            components.append(component)
    return components


def _filter_components(components: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    return [component for component in components if int(component.sum()) >= min_area]


def object_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    connectivity: int = 8,
    min_area: int = 4,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if valid_mask is not None:
        y_true = np.where(valid_mask > 0.5, y_true, 0.0)
        y_prob = np.where(valid_mask > 0.5, y_prob, 0.0)

    pred_components = _filter_components(connected_components(y_prob >= threshold, connectivity), min_area)
    true_components = _filter_components(connected_components(y_true >= 0.5, connectivity), min_area)

    hits = 0
    for true_component in true_components:
        if any(np.any(true_component & pred_component) for pred_component in pred_components):
            hits += 1
    misses = max(len(true_components) - hits, 0)
    false_alarms = 0
    for pred_component in pred_components:
        if not any(np.any(pred_component & true_component) for true_component in true_components):
            false_alarms += 1

    return {
        "hits": float(hits),
        "misses": float(misses),
        "false_alarms": float(false_alarms),
        "object_csi": float(hits / max(hits + misses + false_alarms, 1)),
        "object_pod": float(hits / max(hits + misses, 1)),
        "object_far": float(false_alarms / max(hits + false_alarms, 1)),
    }


def merge_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts]))
        for key in metric_dicts[0].keys()
    }
