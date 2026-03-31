"""
prepare_masks.py
────────────────
Convert per-column boundary coordinates (ILM, RPE) into binary PNG masks.

Expected boundary CSV format (one row per A-scan column):
    x,ilm_y,rpe_y
    0,45,210
    1,44,211
    ...

For multi-layer extension, add columns: x,ilm_y,gcl_y,inl_y,...,rpe_y
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Core mask generation
# ─────────────────────────────────────────────────────────────────────────────

def boundaries_to_binary_mask(
    image_shape: tuple[int, int],
    top_coords: np.ndarray,
    bot_coords: np.ndarray,
) -> np.ndarray:
    """
    Fill a binary mask between two boundary curves.

    Parameters
    ----------
    image_shape : (H, W)
    top_coords  : y-values for upper boundary (ILM), shape (W,)
    bot_coords  : y-values for lower boundary (RPE), shape (W,)

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype uint8, values {0, 1}
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for x in range(W):
        y_top = int(np.clip(round(top_coords[x]), 0, H - 1))
        y_bot = int(np.clip(round(bot_coords[x]), 0, H - 1))

        if y_top > y_bot:
            log.warning(f"Column {x}: ILM ({y_top}) below RPE ({y_bot}) — swapping.")
            y_top, y_bot = y_bot, y_top

        mask[y_top : y_bot + 1, x] = 1

    return mask


def boundaries_to_multilabel_mask(
    image_shape: tuple[int, int],
    boundaries: dict[str, np.ndarray],
    layer_order: list[str],
) -> np.ndarray:
    """
    Generate a multi-class mask where each layer has a unique integer label.

    Parameters
    ----------
    image_shape  : (H, W)
    boundaries   : {'ilm': array, 'gcl': array, ..., 'rpe': array}
    layer_order  : ordered list of boundary names from top to bottom
                   e.g. ['ilm', 'gcl', 'ipl', 'inl', 'opl', 'onl', 'rpe']
                   Layers are the spaces BETWEEN consecutive boundaries.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype uint8
           0 = background, 1 = layer1, 2 = layer2, ...
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for layer_idx in range(len(layer_order) - 1):
        top_name = layer_order[layer_idx]
        bot_name = layer_order[layer_idx + 1]
        top_coords = boundaries[top_name]
        bot_coords = boundaries[bot_name]

        for x in range(W):
            y_top = int(np.clip(round(top_coords[x]), 0, H - 1))
            y_bot = int(np.clip(round(bot_coords[x]), 0, H - 1))
            if y_top > y_bot:
                y_top, y_bot = y_bot, y_top
            mask[y_top : y_bot + 1, x] = layer_idx + 1

    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(col: str) -> str:
    """Lowercase + remove spaces/underscores for flexible matching."""
    return col.lower().replace("_", "").replace(" ", "")

def load_boundary_csv(csv_path: Path) -> pd.DataFrame:
    """Load boundary coordinates. Expects columns: x, ilm_y, rpe_y (+ more)."""
    df = pd.read_csv(csv_path, sep=None, engine="python")  # auto-detect delimiter
    
    # Build normalized lookup
    norm_cols = {normalize(c): c for c in df.columns}
    aliases = {
        "x": ["x"],
        "ilm": ["ilmy", "ilm", "infl", "nfl"],
        "rpe": ["rpey", "rpe"],
    }

    resolved = {}

    for target, options in aliases.items():
        found = None
        for opt in options:
            if opt in norm_cols:
                found = norm_cols[opt]
                break
        if found is None:
            raise ValueError(
                f"{csv_path.name}: missing column for '{target}' (tried {options})"
            )
        resolved[target] = found

    # Rename to standard names
    df = df.rename(columns={
        resolved["x"]: "x",
        resolved["ilm"]: "ilm",
        resolved["rpe"]: "rpe",
    })

    return df.sort_values("x").reset_index(drop=True)

def get_image_shape(image_path: Path) -> tuple[int, int]:
    """Return (H, W) of an image without fully loading it."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img.shape[:2]


# ─────────────────────────────────────────────────────────────────────────────
#  Process a single image
# ─────────────────────────────────────────────────────────────────────────────

def process_one(
    image_path: Path,
    boundary_path: Path,
    output_dir: Path,
    boundary_suffix: str = "_boundaries.csv",
) -> Path:
    """Generate and save a binary mask for one image."""
    H, W = get_image_shape(image_path)
    df = load_boundary_csv(boundary_path)

    # Validate column count matches image width
    if len(df) != W:
        log.warning(
            f"{image_path.name}: boundary has {len(df)} columns, image has {W}. "
            "Interpolating to match."
        )
        old_x = df["x"].values
        new_x = np.arange(W)
        ilm_y = np.interp(new_x, old_x, df["ilm"].values)
        rpe_y = np.interp(new_x, old_x, df["rpe"].values)
    else:
        ilm_y = df["ilm"].values
        rpe_y = df["rpe"].values

    mask = boundaries_to_binary_mask((H, W), ilm_y, rpe_y)

    # Save as PNG (lossless), pixel values 0 / 255 for easy viewing
    output_path = output_dir / (image_path.stem + "_mask.png")
    cv2.imwrite(str(output_path), mask * 255)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    image_ext: str = ".tif",
    boundary_suffix: str = "_layers.csv",
) -> None:
    """
    Walk raw_dir, find all images, generate corresponding masks.

    Expects pairs:
        <animal_id>/image_001.tif
        <animal_id>/image_001_layers.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(raw_dir.rglob(f"*{image_ext}"))

    if not image_paths:
        raise FileNotFoundError(f"No {image_ext} images found in {raw_dir}")

    log.info(f"Found {len(image_paths)} images in {raw_dir}")

    success, skipped = 0, 0

    for img_path in tqdm(image_paths, desc="Generating masks"):
        # ignore fundus images
        if is_fundus(img_path):
            continue

        boundary_path = img_path.with_name(img_path.stem + boundary_suffix)
        if not boundary_path.exists():
            log.warning(f"No boundary file for {img_path.name} — skipping.")
            skipped += 1
            continue

        

        # Mirror the subdirectory structure under output_dir
        rel_dir = img_path.parent.relative_to(raw_dir)
        out_subdir = output_dir / rel_dir
        out_subdir.mkdir(parents=True, exist_ok=True)

        try:
            out_path = process_one(img_path, boundary_path, out_subdir, boundary_suffix)
            log.debug(f"Saved mask → {out_path}")
            success += 1
        except Exception as e:
            log.error(f"Failed on {img_path.parent.name}/{img_path.name}: {e}")
            skipped += 1

    log.info(f"Done — {success} masks saved, {skipped} skipped.")


def is_fundus(path: Path) -> bool:
    return path.stem.lower().endswith("fundus")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate binary masks from boundary CSVs.")
    p.add_argument("--data_dir",  type=Path, required=True,  help="Root of raw data")
    p.add_argument("--output_dir", type=Path, required=True, help="Where to save masks")
    p.add_argument("--image_ext", default=".tif",            help="Image extension")
    p.add_argument(
        "--boundary_suffix",
        default="_layers.csv",
        help="Suffix appended to image stem to find boundary CSV",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        raw_dir=args.data_dir,
        output_dir=args.output_dir,
        image_ext=args.image_ext,
        boundary_suffix=args.boundary_suffix,
    )