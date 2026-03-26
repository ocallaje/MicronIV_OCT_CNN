"""
04_evaluate.py
──────────────
Evaluate a trained model on the held-out test set.

Computes:
  - Mean Dice, IoU across all test images
  - Mean Absolute Boundary Error (ILM and RPE separately)
  - Thickness RMSE
  - Per-image results CSV

Usage:
    python scripts/04_evaluate.py \
        --checkpoint checkpoints/best_model.ckpt \
        --config configs/default.yaml \
        --output_dir results/test_evaluation
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import OCTDataset
from src.data.augmentation import get_val_transforms
from src.models.unet import load_model_from_checkpoint
from src.utils.metrics import compute_all_metrics
from src.utils.visualise import plot_segmentation_comparison, save_overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--output_dir",  default="results/test_evaluation")
    p.add_argument("--save_visuals", action="store_true", default=True,
                   help="Save overlay images for every test sample")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint(args.checkpoint, cfg)
    model.eval().to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    image_size = (cfg["image"]["height"], cfg["image"]["width"])
    test_csv   = Path(cfg["data"]["splits_dir"]) / "test.csv"
    test_ds    = OCTDataset(
        test_csv,
        image_size=image_size,
        transform=get_val_transforms(image_size),
    )
    print(f"Test set: {len(test_ds)} images")

    threshold    = cfg["inference"]["threshold"]
    um_per_pixel = cfg["image"]["um_per_pixel"]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    per_image_results = []

    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Evaluating"):
            sample = test_ds[i]
            image_tensor = sample["image"].unsqueeze(0).to(device)  # (1, 1, H, W)
            mask_gt      = sample["mask"][0].numpy()                 # (H, W) {0, 1}

            logits   = model(image_tensor)
            probs    = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask_pred = (probs > threshold).astype(np.uint8)

            metrics = compute_all_metrics(mask_pred, mask_gt, um_per_pixel)
            metrics["image_path"] = sample["image_path"]
            per_image_results.append(metrics)

            # Save overlay images
            if args.save_visuals:
                image_np = (sample["image"][0].numpy() * 255).astype(np.uint8)
                vis_path = output_dir / f"sample_{i:04d}_overlay.png"
                save_overlay(image_np, mask_pred, vis_path)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(per_image_results)
    results_df.to_csv(output_dir / "per_image_metrics.csv", index=False)

    numeric_cols = [c for c in results_df.columns if c != "image_path"]
    summary = results_df[numeric_cols].agg(["mean", "std", "min", "max"])

    print("\n── Test Set Results ───────────────────────────────────────────")
    print(f"  N images       : {len(results_df)}")
    print(f"  Dice (mean±std): {summary.loc['mean','dice']:.4f} ± {summary.loc['std','dice']:.4f}")
    print(f"  IoU  (mean±std): {summary.loc['mean','iou']:.4f} ± {summary.loc['std','iou']:.4f}")
    print(f"  ILM error      : {summary.loc['mean','ilm_error_um']:.2f} ± {summary.loc['std','ilm_error_um']:.2f} µm")
    print(f"  RPE error      : {summary.loc['mean','rpe_error_um']:.2f} ± {summary.loc['std','rpe_error_um']:.2f} µm")
    print(f"  Thickness RMSE : {summary.loc['mean','rmse_um']:.2f} ± {summary.loc['std','rmse_um']:.2f} µm")
    print("───────────────────────────────────────────────────────────────")

    # Save summary JSON
    summary_dict = {
        col: {"mean": summary.loc["mean", col], "std": summary.loc["std", col]}
        for col in numeric_cols
    }
    with open(output_dir / "summary_metrics.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()