from __future__ import annotations

import argparse
import csv

from .config import load_config, resolve_path
from .dataset import WaveAnomalyDataset
from .inference import inference_rows_for_year, load_model_from_checkpoint, predict_year, select_device
from .utils import ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run yearly prediction for the wave anomaly baseline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    stats_path = resolve_path(config["data"]["stats_path"])
    device = select_device(str(config["train"].get("device", "auto")))
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, config, device)
    threshold = checkpoint.get("threshold")
    if threshold is None:
        threshold = float(config["eval"]["threshold"])

    cache_path, rows = inference_rows_for_year(args.year, config)
    dataset = WaveAnomalyDataset(rows=rows, stats_path=stats_path)
    pred_dir = ensure_dir(resolve_path(config["data"]["prediction_dir"]))
    out_path, summary_rows = predict_year(
        year=args.year,
        dataset=dataset,
        model=model,
        device=device,
        threshold=float(threshold),
        cache_path=cache_path,
        output_dir=pred_dir,
    )

    summary_path = pred_dir / f"prediction_{args.year}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["sample_id", "year", "target_time", "max_prob", "mean_prob", "predicted_area", "label_area"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    dataset.close()
    print(f"[predict] saved {out_path}")
    print(f"[predict] saved {summary_path}")


if __name__ == "__main__":
    main()
