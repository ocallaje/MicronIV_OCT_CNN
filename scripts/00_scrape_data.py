#!/usr/bin/env python3
"""
collect_oct.py

Recursively scan a root server directory for OCT data sets (identified by
an accompanying .prl file) and copy the complete 5‑file bundle into a
dedicated folder under a configurable output directory for later PyTorch
processing.

Author: OpenAI ChatGPT


run with 
python collect_oct.py     -s /mnt/share/studies/Biocryst     -d /mnt/AI/OCT_seg_micron/     --log-level INFO --dry-run
"""

import argparse
import sys
import logging
import re
from pathlib import Path
import shutil
import os

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory tree for OCT data sets and copy the "
            "necessary files into an output directory ready for PyTorch."
        )
    )
    parser.add_argument(
        "-s", "--source",
        required=True,
        type=Path,
        help="Root directory to scan for OCT data."
    )
    parser.add_argument(
        "-d", "--destination",
        required=True,
        type=Path,
        help="Base directory where processed data will be written."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files, just print what would happen."
    )
    return parser.parse_args()


def safe_suffix_match(filename: str, suffix: str) -> bool:
    """Return True if *filename* ends with *suffix* irrespective of case."""
    return filename.lower().endswith(suffix.lower())


def find_matching_files(
    dir_path: Path,
    base_name: str,
    suffixes: dict
) -> dict | None:
    """
    Return a dictionary of suffix -> Path object for all 5 required files
    if they all exist (case‑insensitive). Otherwise, return None.
    """
    # Build a mapping of lower‑case filenames to Path objects.
    files_in_dir = {p.name.lower(): p for p in dir_path.iterdir() if p.is_file()}

    matches = {}
    for key, suffix in suffixes.items():
        expected_name = f"{base_name}{suffix}".lower()
        #print(expected_name)
        if expected_name not in files_in_dir:
            return None
        matches[key] = files_in_dir[expected_name]
        #print("found")
    return matches


def detect_eye(base_name: str) -> str | None:
    """Detect 'RE' or 'LE' in the base_name (case‑insensitive)."""
    m = re.search(r"(?i)(RE|LE)", base_name)
    if m:
        return m.group(0).upper()
    return None


def make_unique_dest(
    dest_root: Path,
    animal_id: str,
    eye: str
) -> Path:
    """
    Construct a unique destination folder of the form
    <dest_root>/<animal_id>_<eye> (or <animal_id>_<eye>_1, _2, ... if needed).
    """
    base_name = f"{animal_id}_{eye}"
    dest_path = dest_root / base_name
    counter = 1
    while dest_path.exists():
        dest_path = dest_root / f"{base_name}_{counter}"
        counter += 1
    return dest_path


def copy_files(
    files_mapping: dict,
    dest_dir: Path,
    dry_run: bool
) -> None:
    """Copy the 5 files into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, src_path in files_mapping.items():
        dst_path = dest_dir / src_path.name
        if dry_run:
            logging.debug(f"[DRY‑RUN] Would copy {src_path} -> {dst_path}")
        else:
            shutil.copy2(src_path, dst_path)
            logging.debug(f"Copied {src_path} -> {dst_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=getattr(logging, args.log_level.upper()),
    )

    source_root = args.source.resolve()
    dest_root = args.destination.resolve()

    if not source_root.is_dir():
        logging.error(f"Source directory {source_root} does not exist or is not a directory.")
        sys.exit(1)

    dest_root.mkdir(parents=True, exist_ok=True)

    # Suffix map (key used for debugging)
    suffix_map = {
        "prl": "_raw.prl",
        "tif": ".tif",
        "fundus": "_fundus.tif",
        "layers": "_layers.csv",
        "thickness": "_thicknesses.csv",
    }

    total_sets = 0
    processed_sets = 0
    skipped_sets = 0

    # Walk recursively
    for dirpath, _, _ in os.walk(source_root):
        dir_path = Path(dirpath)

        # Find all .prl files in this directory (case‑insensitive)
        prl_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and safe_suffix_match(f.name, ".prl")
        ]

        for prl_file in prl_files:
            total_sets += 1
            base_name = prl_file.stem.rsplit('_raw', 1)[0]

            # Check eye label
            eye = detect_eye(base_name)
            if not eye:
                logging.warning(
                    f"Could not find eye label (RE/LE) in basename "
                    f"'{base_name}'. Skipping set in {dir_path}"
                )
                skipped_sets += 1
                continue

            # Find the other 4 files
            matches = find_matching_files(dir_path, base_name, suffix_map)
            if not matches:
                logging.warning(
                    f"Missing required files for base '{base_name}' in {dir_path}. "
                    f"Skipping."
                )
                skipped_sets += 1
                continue

            # Destination folder
            animal_id = dir_path.name  # the parent folder name
            dest_dir = make_unique_dest(dest_root, animal_id, eye)

            copy_files(matches, dest_dir, args.dry_run)
            processed_sets += 1
            logging.info(f"Processed set '{base_name}' (eye: {eye}) to {dest_dir}")

    # Summary
    logging.info("===== Summary =====")
    logging.info(f"Total .prl sets found          : {total_sets}")
    logging.info(f"Sets processed (copied)        : {processed_sets}")
    logging.info(f"Sets skipped (missing data)    : {skipped_sets}")


if __name__ == "__main__":
    main()