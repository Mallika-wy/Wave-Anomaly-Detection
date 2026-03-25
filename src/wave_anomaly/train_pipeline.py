from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

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
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[WaveAnomalyDataset, DataLoader]:
    dataset = WaveAnomalyDataset(index_path=index_path, stats_path=stats_path)
    sampler = None
    if positive_sample_weight is not None and negative_sample_weight is not None and shuffle:
        sampler = build_weighted_sampler(dataset.rows, positive_sample_weight, negative_sample_weight)
        shuffle = False
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)
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
    log_interval: int,
    epoch: int,
    total_epochs: int,
    use_tqdm: bool,
) -> float:
    model.train()
    losses: list[float] = []
    total_steps = len(loader)
    start_time = time.time()
    progress = tqdm(
        loader,
        total=total_steps,
        desc=f"train {epoch}/{total_epochs}",
        dynamic_ncols=True,
        leave=False,
        disable=not use_tqdm,
    )
    for step, (x_wind, x_wave, y, meta) in enumerate(progress, start=1):
        x_wind = x_wind.to(device, non_blocking=True)
        x_wave = x_wave.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_mask = meta["loss_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(x_wind, x_wave, return_logits=True)
            loss = criterion(logits, y, loss_mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        if step == 1 or step % log_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            avg_loss = float(np.mean(losses))
            speed = step / max(elapsed, 1.0e-6)
            eta_seconds = max(total_steps - step, 0) / max(speed, 1.0e-6)
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{avg_loss:.4f}",
                ips=f"{speed:.2f}",
                eta=f"{eta_seconds / 60.0:.1f}m",
            )
    progress.close()
    return float(np.mean(losses)) if losses else 0.0


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    pr_threshold_count: int,
    log_interval: int = 0,
    split_name: str = "eval",
    use_tqdm: bool = True,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    metrics = StreamingPixelMetrics(np.linspace(1.0, 0.0, pr_threshold_count))
    total_steps = len(loader)
    start_time = time.time()
    progress = tqdm(
        loader,
        total=total_steps,
        desc=split_name,
        dynamic_ncols=True,
        leave=False,
        disable=not use_tqdm,
    )
    with torch.no_grad():
        for step, (x_wind, x_wave, y, meta) in enumerate(progress, start=1):
            x_wind = x_wind.to(device, non_blocking=True)
            x_wave = x_wave.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_mask = meta["loss_mask"].to(device, non_blocking=True)
            logits = model(x_wind, x_wave, return_logits=True)
            loss = criterion(logits, y, loss_mask)
            prob = torch.sigmoid(logits)
            metrics.update(y.cpu().numpy(), prob.cpu().numpy(), loss_mask.cpu().numpy())
            losses.append(float(loss.detach().cpu().item()))
            if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == total_steps):
                elapsed = time.time() - start_time
                speed = step / max(elapsed, 1.0e-6)
                eta_seconds = max(total_steps - step, 0) / max(speed, 1.0e-6)
                progress.set_postfix(
                    avg_loss=f"{float(np.mean(losses)):.4f}",
                    ips=f"{speed:.2f}",
                    eta=f"{eta_seconds / 60.0:.1f}m",
                )
    progress.close()
    summary = metrics.best_threshold(key="f1")
    summary["loss"] = float(np.mean(losses)) if losses else 0.0
    summary["pr_auc"] = metrics.pr_auc()
    return summary


def init_wandb_run(
    config: dict[str, Any],
    output_dir: Path,
    model: torch.nn.Module,
) -> Any | None:
    wandb_config = config.get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in the config, but the package is not installed.")

    run = wandb.init(
        project=str(wandb_config.get("project", config.get("project_name", "wave-anomaly-detection"))),
        entity=wandb_config.get("entity") or None,
        name=wandb_config.get("run_name") or None,
        notes=wandb_config.get("notes") or None,
        tags=list(wandb_config.get("tags", [])),
        mode=str(wandb_config.get("mode", "online")),
        dir=str(output_dir),
        config=config,
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    if bool(wandb_config.get("watch_model", False)):
        run.watch(
            model,
            log=str(wandb_config.get("watch_log", "gradients")),
            log_freq=int(wandb_config.get("watch_log_freq", 100)),
        )
    return run


def log_wandb_metrics(
    run: Any | None,
    epoch: int,
    train_loss: float,
    test_summary: dict[str, Any],
    lr: float,
    best_pr_auc: float,
    best_epoch: int,
) -> None:
    if run is None:
        return

    run.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "test/loss": float(test_summary["loss"]),
            "test/pr_auc": float(test_summary["pr_auc"]),
            "test/f1": float(test_summary["f1"]),
            "test/precision": float(test_summary["precision"]),
            "test/recall": float(test_summary["recall"]),
            "test/threshold": float(test_summary["threshold"]),
            "train/lr": lr,
            "best/test_pr_auc": best_pr_auc,
            "best/epoch": best_epoch,
        },
        step=epoch,
    )


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
            x_wind = x_wind.to(device, non_blocking=True)
            x_wave = x_wave.to(device, non_blocking=True)
            prob = model(x_wind, x_wave, return_logits=False).cpu().numpy()
            y_np = y.numpy()
            mask_np = meta["loss_mask"].numpy()
            metric_acc.update(y_np, prob, mask_np)
    best = metric_acc.best_threshold(key="f1")

    with torch.no_grad():
        for x_wind, x_wave, y, meta in loader:
            x_wind = x_wind.to(device, non_blocking=True)
            x_wave = x_wave.to(device, non_blocking=True)
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
    persistent_workers = bool(config["train"].get("persistent_workers", True))
    prefetch_factor = int(config["train"].get("prefetch_factor", 2))
    log_interval = max(1, int(config["train"].get("log_interval", 100)))
    eval_log_interval = max(1, int(config["train"].get("eval_log_interval", 100)))
    use_tqdm = bool(config["train"].get("use_tqdm", True))

    train_dataset, train_loader = build_loader(
        train_index,
        stats_path,
        batch_size,
        num_workers,
        shuffle=True,
        positive_sample_weight=float(config["train"].get("positive_sample_weight", 4.0)),
        negative_sample_weight=float(config["train"].get("negative_sample_weight", 1.0)),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_dataset, test_loader = build_loader(
        test_index,
        stats_path,
        batch_size,
        num_workers,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
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
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
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
    scaler = GradScaler(device=device.type, enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best.ckpt"
    last_path = output_dir / "last.ckpt"
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    checkpoint_interval = max(1, int(config["train"].get("checkpoint_interval", 5)))

    best_pr_auc = -float("inf")
    best_epoch = 0
    threshold_csv_path = output_dir / "test_threshold_scan.csv"
    wandb_run = init_wandb_run(config, output_dir, model)

    try:
        with history_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["epoch", "train_loss", "test_loss", "test_pr_auc", "test_f1", "test_threshold"],
            )
            writer.writeheader()

            total_epochs = int(config["train"]["epochs"])
            for epoch in range(1, total_epochs + 1):
                print(f"[train] starting epoch {epoch}/{total_epochs}", flush=True)
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    amp_enabled=bool(config["train"].get("amp", True)),
                    max_grad_norm=float(config["train"].get("max_grad_norm", 1.0)),
                    log_interval=log_interval,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    use_tqdm=use_tqdm,
                )
                print(f"[test] starting evaluation for epoch {epoch}/{total_epochs}", flush=True)
                test_summary = evaluate_loader(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    pr_threshold_count=int(config["eval"].get("pr_auc_thresholds", 201)),
                    log_interval=eval_log_interval,
                    split_name="test",
                    use_tqdm=use_tqdm,
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
                if epoch % checkpoint_interval == 0:
                    epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.ckpt"
                    save_checkpoint(epoch_checkpoint_path, model, optimizer, scheduler, epoch, test_summary, config)
                if test_summary["pr_auc"] > best_pr_auc:
                    best_pr_auc = float(test_summary["pr_auc"])
                    best_epoch = epoch
                    save_checkpoint(best_path, model, optimizer, scheduler, epoch, test_summary, config)

                log_wandb_metrics(
                    run=wandb_run,
                    epoch=epoch,
                    train_loss=train_loss,
                    test_summary=test_summary,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    best_pr_auc=best_pr_auc,
                    best_epoch=best_epoch,
                )
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
            print("[val] starting final evaluation", flush=True)
            summary["val_summary"] = evaluate_loader(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                pr_threshold_count=int(config["eval"].get("pr_auc_thresholds", 201)),
                log_interval=eval_log_interval,
                split_name="val",
                use_tqdm=use_tqdm,
            )
        save_json(output_dir / "summary.json", summary)

        if wandb_run is not None:
            wandb_run.summary["best_epoch"] = int(best_epoch)
            wandb_run.summary["best_test_pr_auc"] = float(best_pr_auc)
            wandb_run.summary["best_threshold"] = float(best_threshold["threshold"])
            if "val_summary" in summary:
                for key, value in summary["val_summary"].items():
                    if isinstance(value, (int, float, np.floating)):
                        wandb_run.summary[f"val/{key}"] = float(value)
    finally:
        train_dataset.close()
        test_dataset.close()
        if val_dataset is not None:
            val_dataset.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
