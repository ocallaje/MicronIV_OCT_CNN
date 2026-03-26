"""
predict.py
──────────
Run inference on new OCT images using a trained checkpoint.

Single image or batch. Outputs:
  - Binary mask PNG
  - Thickness profile CSV
  - Overlay visualisation (optional)
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

from src.inference.thickness import (
    mask_to_thickness_profile,
    plot_thickness_profile,
    save_thickness_csv,
    summarise_thickness,
)
from src.models.unet import load_model_from_checkpoint
from src.utils.visualise import save_overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Core predictor class
# ─────────────────────────────────────────────────────────────────────────────

class OCTPredictor:
    """
    Wraps a trained model for inference on new OCT images.

    Usage
    -----
        predictor = OCTPredictor("checkpoints/best_model.ckpt", "configs/default.yaml")
        result = predictor.predict("path/to/new_oct.tif")
        print(result["summary"])
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: Optional[str] = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = load_model_from_checkpoint(checkpoint_path, self.cfg)
        self.model.eval().to(self.device)

        self.image_size = (self.cfg["image"]["height"], self.cfg["image"]["width"])
        self.threshold  = self.cfg["inference"]["threshold"]
        self.um_per_pixel = self.cfg["image"]["um_per_pixel"]

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_outputs: bool = True,
    ) -> dict:
        """
        Run prediction on a single OCT image.

        Parameters
        ----------
        image_path  : path to input OCT image
        output_dir  : where to save outputs (defaults to same dir as image)
        save_outputs: whether to save mask, CSV, overlay to disk

        Returns
        -------
        dict with keys:
            'mask'          : (H_orig, W_orig) binary numpy array
            'thickness_profile' : from mask_to_thickness_profile()
            'summary'       : from summarise_thickness()
            'output_paths'  : dict of saved file paths
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # Load and preprocess
        image_orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_orig is None:
            raise IOError(f"Could not read: {image_path}")
        H_orig, W_orig = image_orig.shape

        image_resized = cv2.resize(
            image_orig,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Tensor: (1, 1, H, W), float [0, 1]
        tensor = (
            torch.from_numpy(image_resized)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # Inference
        logits = self.model(tensor)           # (1, 1, H, W)
        probs  = torch.sigmoid(logits)
        mask_resized = (probs[0, 0].cpu().numpy() > self.threshold).astype(np.uint8)

        # Resize mask back to original image dimensions
        mask = cv2.resize(
            mask_resized,
            (W_orig, H_orig),
            interpolation=cv2.INTER_NEAREST,
        )

        # Thickness analysis
        profile = mask_to_thickness_profile(mask, um_per_pixel=self.um_per_pixel)
        summary = summarise_thickness(profile)

        output_paths = {}
        if save_outputs:
            out_dir = Path(output_dir) if output_dir else image_path.parent / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem

            # Save binary mask
            mask_path = out_dir / f"{stem}_mask.png"
            cv2.imwrite(str(mask_path), mask * 255)
            output_paths["mask"] = str(mask_path)

            # Save thickness CSV
            csv_path = out_dir / f"{stem}_thickness.csv"
            save_thickness_csv(profile, csv_path)
            output_paths["thickness_csv"] = str(csv_path)

            # Save overlay visualisation
            if self.cfg["inference"]["output_overlays"]:
                overlay_path = out_dir / f"{stem}_overlay.png"
                save_overlay(image_orig, mask, overlay_path)
                output_paths["overlay"] = str(overlay_path)

                plot_path = out_dir / f"{stem}_thickness_plot.png"
                plot_thickness_profile(image_orig, profile, plot_path, title=stem)
                output_paths["thickness_plot"] = str(plot_path)

        return {
            "mask":               mask,
            "thickness_profile":  profile,
            "summary":            summary,
            "output_paths":       output_paths,
        }

    def predict_batch(
        self,
        image_dir: str,
        output_dir: str,
        image_ext: str = ".tif",
    ) -> list[dict]:
        """Run prediction on all images in a directory."""
        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.rglob(f"*{image_ext}"))
        print(f"Found {len(image_paths)} images in {image_dir}")

        results = []
        for img_path in image_paths:
            print(f"  Processing: {img_path.name}")
            result = self.predict(str(img_path), output_dir=output_dir)
            result["image_path"] = str(img_path)
            results.append(result)

        print(f"\nDone. Results saved to {output_dir}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run OCT segmentation inference.")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config",     required=True, help="Path to config YAML")
    p.add_argument("--image",      help="Single image path")
    p.add_argument("--image_dir",  help="Directory of images (batch mode)")
    p.add_argument("--output_dir", default="predictions")
    p.add_argument("--image_ext",  default=".tif")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = OCTPredictor(args.checkpoint, args.config)

    if args.image:
        result = predictor.predict(args.image, output_dir=args.output_dir)
        print("\n── Thickness Summary ──")
        s = result["summary"]
        print(f"  Mean thickness : {s['microns']['mean']:.1f} µm")
        print(f"  Std            : {s['microns']['std']:.1f} µm")
        print(f"  Min / Max      : {s['microns']['min']:.1f} / {s['microns']['max']:.1f} µm")
        print(f"\nOutputs saved to: {result['output_paths']}")

    elif args.image_dir:
        predictor.predict_batch(args.image_dir, args.output_dir, args.image_ext)
    else:
        print("Provide --image or --image_dir")