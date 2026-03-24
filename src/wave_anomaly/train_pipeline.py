from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .dataset import WaveAnomalyDataset, build_weighted_sampler
from .inference import load_model_from_checkpoint, save_checkpoint, select_device
from .losses import build_loss
from .metrics import StreamingPixelMetrics, make_thresholds, object_metrics
from .model import build_model
from .utils import ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the wave anomaly detection baseline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def build_loader(
    index_path: Path,
    stats_path: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    positive_sample_weight: float | None = None,
    negative_sample_weight: float | None = None,
) -> tuple[WaveAnomalyDataset, DataLoader]:
    dataset = WaveAnomalyDataset(index_path=index_path, stats_path=stats_path)
    sampler = None
    if positive_sample_weight is not None and negative_sample_weight is not None and shuffle:
        sampler = build_weighted_sampler(dataset.rows, positive_sample_weight, negative_sample_weight)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_grad_norm: float,
) -> float:
    model.train()
    losses: list[float] = []
    for x_wind, x_wave, y, meta in loader:
        x_wind = x_wind.to(device)
        x_wave = x_wave.to(device)
        y = y.to(device)
        loss_mask = meta["loss_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled and device.type == "cuda"):
            logits = model(x_wind, x_wave, return_logits=True)
            loss = criterion(logits, y, loss_mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    pr_threshold_count: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    metrics = StreamingPixelMetrics(np.linspace(1.0, 0.0, pr_threshold_count))
    with torch.no_grad():
        for x_wind, x_wave, y, meta in loader:
            x_wind = x_wind.to(device)
            x_wave = x_wave.to(device)
            y = y.to(device)
            loss_mask = meta["loss_mask"].to(device)
            logits = model(x_wind, x_wave, return_logits=True)
            loss = criterion(logits, y, loss_mask)
            prob = torch.sigmoid(logits)
            metrics.update(y.cpu().numpy(), prob.cpu().numpy(), loss_mask.cpu().numpy())
            losses.append(float(loss.detach().cpu().item()))
    summary = metrics.best_threshold(key="f1")
    summary["loss"] = float(np.mean(losses)) if losses else 0.0
    summary["pr_auc"] = metrics.pr_auc()
    return summary


def scan_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    thresholds = make_thresholds(
        float(config["eval"]["threshold_start"]),
        float(config["eval"]["threshold_end"]),
        float(config["eval"]["threshold_step"]),
    )
    metric_acc = StreamingPixelMetrics(thresholds)
    object_rows: list[dict[str, float]] = []

    model.eval()
    with torch.no_grad():
        for x_wind, x_wave, y, meta in loader:
            x_wind = x_wind.to(device)
            x_wave = x_wave.to(device)
            prob = model(x_wind, x_wave, return_logits=False).cpu().numpy()
            y_np = y.numpy()
            mask_np = meta["loss_mask"].numpy()
            metric_acc.update(y_np, prob, mask_np)
    best = metric_acc.best_threshold(key="f1")

    with torch.no_grad():
        for x_wind, x_wave, y, meta in loader:
            x_wind = x_wind.to(device)
            x_wave = x_wave.to(device)
            prob = model(x_wind, x_wave, return_logits=False).cpu().numpy()
            for sample_idx in range(prob.shape[0]):
                object_rows.append(
                    object_metrics(
                        y_true=y[sample_idx, 0].numpy(),
                        y_prob=prob[sample_idx, 0],
                        threshold=float(best["threshold"]),
                        connectivity=int(config["eval"]["connectivity"]),
                        min_area=int(config["eval"]["min_area"]),
                        valid_mask=meta["loss_mask"][sample_idx, 0].numpy(),
                    )
                )
    if object_rows:
        best.update(
            {
                "object_csi": float(np.mean([row["object_csi"] for row in object_rows])),
                "object_pod": float(np.mean([row["object_pod"] for row in object_rows])),
                "object_far": float(np.mean([row["object_far"] for row in object_rows])),
            }
        )
    return best, metric_acc.table()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    output_dir = ensure_dir(resolve_path(config["train"]["output_dir"]))
    stats_path = resolve_path(config["data"]["stats_path"])
    index_dir = resolve_path(config["data"]["index_dir"])
    train_index = index_dir / "train_index.csv"
    test_index = index_dir / "test_index.csv"
    val_index = index_dir / "val_index.csv"

    device = select_device(str(config["train"].get("device", "auto")))
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])

    train_dataset, train_loader = build_loader(
        train_index,
        stats_path,
        batch_size,
        num_workers,
        shuffle=True,
        positive_sample_weight=float(config["train"].get("positive_sample_weight", 4.0)),
        negative_sample_weight=float(config["train"].get("negative_sample_weight", 1.0)),
    )
    test_dataset, test_loader = build_loader(
        test_index,
        stats_path,
        batch_size,
        num_workers,
        shuffle=False,
    )
    val_dataset = None
    val_loader = None
    if val_index.exists():
        val_dataset, val_loader = build_loader(
            val_index,
            stats_path,
            batch_size,
            num_workers,
            shuffle=False,
        )

    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["train"]["epochs"])),
    )
    scaler = GradScaler(enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best.ckpt"
    last_path = output_dir / "last.ckpt"

    best_pr_auc = -float("inf")
    best_epoch = 0
    threshold_csv_path = output_dir / "test_threshold_scan.csv"

    with history_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["epoch", "train_loss", "test_loss", "test_pr_auc", "test_f1", "test_threshold"],
        )
        writer.writeheader()

        for epoch in range(1, int(config["train"]["epochs"]) + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_enabled=bool(config["train"].get("amp", True)),
                max_grad_norm=float(config["train"].get("max_grad_norm", 1.0)),
            )
            test_summary = evaluate_loader(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                pr_threshold_count=int(config["eval"].get("pr_auc_thresholds", 201)),
            )
            scheduler.step()

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "test_loss": f"{test_summary['loss']:.6f}",
                    "test_pr_auc": f"{test_summary['pr_auc']:.6f}",
                    "test_f1": f"{test_summary['f1']:.6f}",
                    "test_threshold": f"{test_summary['threshold']:.6f}",
                }
            )
            fp.flush()

            save_checkpoint(last_path, model, optimizer, scheduler, epoch, test_summary, config)
            if test_summary["pr_auc"] > best_pr_auc:
                best_pr_auc = float(test_summary["pr_auc"])
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, test_summary, config)
            print(
                f"[train] epoch={epoch} train_loss={train_loss:.4f} "
                f"test_pr_auc={test_summary['pr_auc']:.4f} test_f1={test_summary['f1']:.4f}"
            )

    model, checkpoint = load_model_from_checkpoint(best_path, config, device)
    best_threshold, threshold_table = scan_thresholds(model, test_loader, device, config)
    save_json(output_dir / "test_threshold_scan.json", {"best": best_threshold, "table": threshold_table})
    with threshold_csv_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["threshold", "precision", "recall", "f1", "iou", "dice", "csi", "pod", "far", "accuracy"]
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in threshold_table)
    save_checkpoint(
        best_path,
        model,
        optimizer,
        scheduler,
        int(checkpoint["epoch"]),
        checkpoint["metrics"],
        config,
        threshold=float(best_threshold["threshold"]),
    )

    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_test_pr_auc": best_pr_auc,
        "best_threshold": best_threshold,
    }
    if val_loader is not None:
        summary["val_summary"] = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            pr_threshold_count=int(config["eval"].get("pr_auc_thresholds", 201)),
        )
    save_json(output_dir / "summary.json", summary)

    train_dataset.close()
    test_dataset.close()
    if val_dataset is not None:
        val_dataset.close()


if __name__ == "__main__":
    main()
