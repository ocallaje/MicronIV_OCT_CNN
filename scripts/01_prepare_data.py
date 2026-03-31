"""
01_prepare_data.py
──────────────────
Generate binary masks from boundary coordinate CSVs.

Usage:
    python scripts/01_prepare_data.py \
        --data_dir data/raw \
        --output_dir data/masks \
        --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.prepare_masks import process_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=Path, default="/mnt/AI/OCT_seg_micron/data/raw")
    p.add_argument("--output_dir", type=Path, default="/mnt/AI/OCT_seg_micron/data/masks")
    p.add_argument("--config",     type=Path, default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    process_dataset(
        raw_dir=args.data_dir,
        output_dir=args.output_dir,
        image_ext=cfg["data"]["image_ext"],
        boundary_suffix=cfg["data"]["boundary_suffix"],
    )


if __name__ == "__main__":
    main()