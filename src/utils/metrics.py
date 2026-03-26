"""
metrics.py
──────────
Evaluation metrics for OCT segmentation.

  - Dice coefficient
  - IoU (Jaccard index)
  - Mean Absolute Boundary Error (MABE) in pixels and µm
  - Thickness RMSE
"""

import numpy as np


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Binary Dice coefficient.

    Parameters
    ----------
    pred, target : binary arrays (0/1), any shape
    """
    pred   = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Intersection over Union (Jaccard index)."""
    pred   = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union        = np.logical_or(pred, target).sum()
    return (intersection + smooth) / (union + smooth)


def mask_to_boundaries(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract top (ILM) and bottom (RPE) boundary y-coordinates from a binary mask.

    Parameters
    ----------
    mask : (H, W) binary array

    Returns
    -------
    top_boundary : (W,) y-coordinate of topmost mask pixel per column
    bot_boundary : (W,) y-coordinate of bottommost mask pixel per column
    """
    H, W = mask.shape
    top_boundary = np.full(W, np.nan)
    bot_boundary = np.full(W, np.nan)

    for x in range(W):
        col = mask[:, x]
        ys  = np.where(col)[0]
        if len(ys) > 0:
            top_boundary[x] = ys[0]
            bot_boundary[x] = ys[-1]

    return top_boundary, bot_boundary


def mean_absolute_boundary_error(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    um_per_pixel: float = 1.0,
) -> dict:
    """
    Compute mean absolute error between predicted and ground truth boundaries.

    Returns dict with:
        'ilm_error_px'  : MAE of ILM boundary in pixels
        'rpe_error_px'  : MAE of RPE boundary in pixels
        'ilm_error_um'  : MAE in µm
        'rpe_error_um'  : MAE in µm
    """
    pred_ilm, pred_rpe     = mask_to_boundaries(pred_mask)
    target_ilm, target_rpe = mask_to_boundaries(target_mask)

    # Only evaluate columns where both masks have valid predictions
    valid_ilm = ~np.isnan(pred_ilm) & ~np.isnan(target_ilm)
    valid_rpe = ~np.isnan(pred_rpe) & ~np.isnan(target_rpe)

    ilm_err = np.abs(pred_ilm[valid_ilm] - target_ilm[valid_ilm]).mean()
    rpe_err = np.abs(pred_rpe[valid_rpe] - target_rpe[valid_rpe]).mean()

    return {
        "ilm_error_px": float(ilm_err),
        "rpe_error_px": float(rpe_err),
        "ilm_error_um": float(ilm_err * um_per_pixel),
        "rpe_error_um": float(rpe_err * um_per_pixel),
    }


def thickness_rmse(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    um_per_pixel: float = 1.0,
) -> dict:
    """
    Root mean square error of per-column thickness.

    Returns dict with 'rmse_px' and 'rmse_um'.
    """
    pred_ilm, pred_rpe     = mask_to_boundaries(pred_mask)
    target_ilm, target_rpe = mask_to_boundaries(target_mask)

    valid = (
        ~np.isnan(pred_ilm) & ~np.isnan(pred_rpe)
        & ~np.isnan(target_ilm) & ~np.isnan(target_rpe)
    )

    pred_thick   = pred_rpe[valid]   - pred_ilm[valid]
    target_thick = target_rpe[valid] - target_ilm[valid]
    rmse = np.sqrt(np.mean((pred_thick - target_thick) ** 2))

    return {
        "rmse_px": float(rmse),
        "rmse_um": float(rmse * um_per_pixel),
    }


def compute_all_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    um_per_pixel: float = 1.0,
) -> dict:
    """Convenience: compute all metrics at once."""
    return {
        "dice":  dice_coefficient(pred_mask, target_mask),
        "iou":   iou_score(pred_mask, target_mask),
        **mean_absolute_boundary_error(pred_mask, target_mask, um_per_pixel),
        **thickness_rmse(pred_mask, target_mask, um_per_pixel),
    }