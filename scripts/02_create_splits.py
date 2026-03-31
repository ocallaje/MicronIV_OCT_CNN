"""
02_create_splits.py
───────────────────
Create train/val/test splits, stratified by animal ID.

WHY split by animal?
  Each eye/animal contributes multiple B-scans. If you split randomly by image,
  the same eye appears in both train and test — the model memorises that retina
  rather than generalising. Always split at the animal level.

Expects:
    data/raw/<animal_id>/<image_name>.tif
    data/masks/<animal_id>/<image_name>_mask.png

Output CSVs (data/splits/):
    train.csv  |  val.csv  |  test.csv
    Columns: image_path, mask_path, animal_id

Usage:
    python scripts/02_create_splits.py \
        --raw_dir data/raw \
        --mask_dir data/masks \
        --output_dir data/splits \
        --val_fraction 0.15 \
        --test_fraction 0.15 \
        --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",       type=Path, required=True)
    p.add_argument("--mask_dir",      type=Path, required=True)
    p.add_argument("--output_dir",    type=Path, default=Path("data/splits"))
    p.add_argument("--config",        type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--val_fraction",  type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    image_ext = cfg["data"]["image_ext"]

    # ── Discover all (image, mask) pairs ─────────────────────────────────────
    records = []
    for img_path in sorted(args.raw_dir.rglob(f"*{image_ext}")):
        # Skip fundus images
        if is_fundus(img_path):
            continue

        animal_id = img_path.parent.name  # assumes <raw_dir>/<animal_id>/<image>
        mask_path = args.mask_dir / img_path.parent.relative_to(args.raw_dir) \
                    / (img_path.stem + "_mask.png")

        if not mask_path.exists():
            print(f"  [skip] No mask for {img_path.name}")
            continue

        records.append({
            "image_path": str(img_path),
            "mask_path":  str(mask_path),
            "animal_id":  animal_id,
        })

    df = pd.DataFrame(records)
    print(f"\nFound {len(df)} image-mask pairs across "
          f"{df['animal_id'].nunique()} animals.\n")

    # ── Split by animal ID ───────────────────────────────────────────────────
    animals = sorted(df["animal_id"].unique())
    random.seed(args.seed)
    random.shuffle(animals)

    n_test = max(1, round(len(animals) * args.test_fraction))
    n_val  = max(1, round(len(animals) * args.val_fraction))
    n_train = len(animals) - n_test - n_val

    if n_train < 1:
        raise ValueError(
            f"Not enough animals ({len(animals)}) for a train split. "
            "Reduce val/test fractions or add more data."
        )

    test_animals  = animals[:n_test]
    val_animals   = animals[n_test : n_test + n_val]
    train_animals = animals[n_test + n_val :]

    train_df = df[df["animal_id"].isin(train_animals)]
    val_df   = df[df["animal_id"].isin(val_animals)]
    test_df  = df[df["animal_id"].isin(test_animals)]

    # ── Save ─────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(  args.output_dir / "val.csv",   index=False)
    test_df.to_csv( args.output_dir / "test.csv",  index=False)

    print("── Split Summary ──────────────────────────────")
    print(f"  Train : {len(train_df):4d} images | {len(train_animals)} animals: {train_animals}")
    print(f"  Val   : {len(val_df):4d} images | {len(val_animals)}  animals: {val_animals}")
    print(f"  Test  : {len(test_df):4d} images | {len(test_animals)}  animals: {test_animals}")
    print(f"\nSaved splits → {args.output_dir}/")


def is_fundus(path: Path) -> bool:
    return "fundus" in path.stem.lower()


if __name__ == "__main__":
    main()