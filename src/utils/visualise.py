"""
visualise.py
────────────
Visualisation utilities for OCT segmentation results.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    alpha: float = 0.35,
    color: tuple = (0, 255, 100),   # green overlay
) -> None:
    """
    Save OCT image with mask overlay.

    Parameters
    ----------
    image       : (H, W) grayscale uint8
    mask        : (H, W) binary {0, 1}
    output_path : save path
    alpha       : mask transparency (0=invisible, 1=opaque)
    color       : BGR overlay colour
    """
    # Convert grayscale → BGR for colour overlay
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay   = image_bgr.copy()

    overlay[mask == 1] = color

    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(str(output_path), blended)


def plot_segmentation_comparison(
    image:      np.ndarray,
    pred_mask:  np.ndarray,
    gt_mask:    Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "Segmentation Result",
) -> plt.Figure:
    """
    Side-by-side comparison: [OCT] [Ground Truth] [Prediction]
    If no GT provided, shows [OCT] [Prediction] only.
    """
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("OCT Image")
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(gt_mask, alpha=0.4, cmap="Greens")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(pred_mask, alpha=0.4, cmap="Reds")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
    else:
        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(pred_mask, alpha=0.4, cmap="Greens")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig


def plot_boundary_overlay(
    image: np.ndarray,
    ilm_y: np.ndarray,
    rpe_y: np.ndarray,
    gt_ilm_y: Optional[np.ndarray] = None,
    gt_rpe_y: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "Boundary Overlay",
) -> plt.Figure:
    """
    Plot OCT image with predicted (and optionally ground truth) boundaries.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image, cmap="gray", aspect="auto")

    x = np.arange(len(ilm_y))
    valid = ~np.isnan(ilm_y) & ~np.isnan(rpe_y)

    if gt_ilm_y is not None:
        ax.plot(x[valid], gt_ilm_y[valid], "g-",  linewidth=1.5, label="GT ILM",  alpha=0.8)
        ax.plot(x[valid], gt_rpe_y[valid], "g--", linewidth=1.5, label="GT RPE",  alpha=0.8)

    ax.plot(x[valid], ilm_y[valid], "c-",  linewidth=1.5, label="Pred ILM")
    ax.plot(x[valid], rpe_y[valid], "y--", linewidth=1.5, label="Pred RPE")

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig