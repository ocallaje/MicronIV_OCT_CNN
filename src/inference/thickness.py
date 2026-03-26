"""
thickness.py
────────────
Extract retinal thickness profiles from segmentation masks.

Converts binary masks → per-column thickness → summary statistics → µm.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_thickness_profile(
    mask: np.ndarray,
    um_per_pixel: float = 1.0,
) -> dict:
    """
    Compute per-column retinal thickness from a binary mask.

    Parameters
    ----------
    mask         : (H, W) binary array {0, 1}
    um_per_pixel : axial scale factor (µm per pixel)

    Returns
    -------
    dict with:
        'ilm_y'        : (W,) ILM boundary y-coordinates (pixels)
        'rpe_y'        : (W,) RPE boundary y-coordinates (pixels)
        'thickness_px' : (W,) per-column thickness in pixels
        'thickness_um' : (W,) per-column thickness in µm
        'valid_cols'   : (W,) boolean — columns where mask is present
    """
    H, W = mask.shape
    ilm_y    = np.full(W, np.nan)
    rpe_y    = np.full(W, np.nan)
    valid    = np.zeros(W, dtype=bool)

    for x in range(W):
        col = mask[:, x]
        ys  = np.where(col)[0]
        if len(ys) > 0:
            ilm_y[x] = ys[0]
            rpe_y[x] = ys[-1]
            valid[x] = True

    thickness_px = np.where(valid, rpe_y - ilm_y, np.nan)
    thickness_um = thickness_px * um_per_pixel

    return {
        "ilm_y":        ilm_y,
        "rpe_y":        rpe_y,
        "thickness_px": thickness_px,
        "thickness_um": thickness_um,
        "valid_cols":   valid,
    }


def summarise_thickness(profile: dict) -> dict:
    """
    Compute summary statistics from a thickness profile.

    Returns mean, median, std, min, max — in both px and µm.
    """
    t_px = profile["thickness_px"][profile["valid_cols"]]
    t_um = profile["thickness_um"][profile["valid_cols"]]

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
        }

    return {
        "n_valid_columns": int(profile["valid_cols"].sum()),
        "pixels": stats(t_px),
        "microns": stats(t_um),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Zonal analysis (e.g. ETDRS-style regions)
# ─────────────────────────────────────────────────────────────────────────────

def zonal_thickness(
    thickness_um: np.ndarray,
    valid_cols: np.ndarray,
    n_zones: int = 3,
) -> dict:
    """
    Divide the B-scan into N equal horizontal zones and compute mean thickness.

    For a B-scan this gives nasal / central / temporal regions (or finer).

    Parameters
    ----------
    thickness_um : (W,) per-column thickness in µm
    valid_cols   : (W,) boolean mask of valid columns
    n_zones      : number of equal-width zones (default 3)

    Returns
    -------
    dict: {'zone_0': mean_um, 'zone_1': mean_um, ..., 'zone_N-1': mean_um}
    """
    W = len(thickness_um)
    zone_width = W // n_zones
    result = {}

    for i in range(n_zones):
        start = i * zone_width
        end   = (i + 1) * zone_width if i < n_zones - 1 else W
        zone_t = thickness_um[start:end]
        zone_v = valid_cols[start:end]
        valid_t = zone_t[zone_v]
        result[f"zone_{i}_mean_um"] = float(np.mean(valid_t)) if len(valid_t) > 0 else np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_thickness_profile(
    image: Optional[np.ndarray],
    profile: dict,
    output_path: Optional[Path] = None,
    title: str = "Retinal Thickness Profile",
) -> plt.Figure:
    """
    Plot the thickness profile with optional OCT image overlay.
    """
    fig, axes = plt.subplots(
        1 if image is None else 2,
        1,
        figsize=(12, 8 if image is not None else 4),
    )

    if image is not None:
        ax_img, ax_thick = axes
        ax_img.imshow(image, cmap="gray")
        x_vals = np.arange(len(profile["ilm_y"]))
        valid = profile["valid_cols"]
        ax_img.plot(x_vals[valid], profile["ilm_y"][valid],
                    color="cyan",   linewidth=1.5, label="ILM")
        ax_img.plot(x_vals[valid], profile["rpe_y"][valid],
                    color="yellow", linewidth=1.5, label="RPE")
        ax_img.set_title("OCT with Segmented Boundaries")
        ax_img.legend(loc="upper right", fontsize=8)
        ax_img.axis("off")
    else:
        ax_thick = axes

    x_vals = np.arange(len(profile["thickness_um"]))
    valid  = profile["valid_cols"]
    ax_thick.plot(x_vals[valid], profile["thickness_um"][valid],
                  color="#2196F3", linewidth=1.5)
    ax_thick.fill_between(x_vals[valid], profile["thickness_um"][valid],
                          alpha=0.2, color="#2196F3")
    ax_thick.set_xlabel("A-scan column (pixels)")
    ax_thick.set_ylabel("Thickness (µm)")
    ax_thick.set_title(title)
    ax_thick.grid(True, alpha=0.3)

    # Annotate mean
    mean_um = np.nanmean(profile["thickness_um"])
    ax_thick.axhline(mean_um, color="red", linestyle="--", linewidth=1,
                     label=f"Mean: {mean_um:.1f} µm")
    ax_thick.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Saved thickness plot → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_thickness_csv(profile: dict, output_path: Path) -> None:
    """Save per-column thickness data to CSV."""
    W = len(profile["thickness_px"])
    df = pd.DataFrame({
        "column":       np.arange(W),
        "ilm_y":        profile["ilm_y"],
        "rpe_y":        profile["rpe_y"],
        "thickness_px": profile["thickness_px"],
        "thickness_um": profile["thickness_um"],
        "valid":        profile["valid_cols"],
    })
    df.to_csv(output_path, index=False)
    print(f"Saved thickness CSV → {output_path}")