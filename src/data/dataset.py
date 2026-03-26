"""
dataset.py
──────────
PyTorch Dataset for OCT segmentation.

Loads paired (image, mask) from disk, applies augmentation, and returns
tensors ready for a U-Net.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OCTDataset(Dataset):
    """
    Parameters
    ----------
    split_csv   : path to CSV with columns [image_path, mask_path]
    image_size  : (H, W) to resize to
    transform   : albumentations Compose pipeline (or None)
    """

    def __init__(
        self,
        split_csv: Path,
        image_size: tuple[int, int] = (512, 512),
        transform=None,
    ):
        self.df = pd.read_csv(split_csv)
        self.image_size = image_size  # (H, W)
        self.transform = transform

        # Validate all files exist upfront — fail early
        missing = []
        for _, row in self.df.iterrows():
            if not Path(row["image_path"]).exists():
                missing.append(row["image_path"])
            if not Path(row["mask_path"]).exists():
                missing.append(row["mask_path"])
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files not found:\n" + "\n".join(missing[:5])
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = self._load_image(row["image_path"])
        mask  = self._load_mask(row["mask_path"])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # Convert to tensors
        # image: (H, W) → (1, H, W), float32 [0, 1]
        # mask:  (H, W) → (1, H, W), float32 {0, 1}
        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask_tensor  = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return {
            "image":      image_tensor,
            "mask":       mask_tensor,
            "image_path": str(row["image_path"]),
        }

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not load image: {path}")
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]),
                         interpolation=cv2.INTER_LINEAR)
        return img  # uint8, shape (H, W)

    def _load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not load mask: {path}")
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]),
                          interpolation=cv2.INTER_NEAREST)  # no interpolation for masks!
        return mask  # uint8 {0, 255}, shape (H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  DataModule (PyTorch Lightning)
# ─────────────────────────────────────────────────────────────────────────────

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.augmentation import get_train_transforms, get_val_transforms


class OCTDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        splits_dir = Path(cfg["data"]["splits_dir"])
        self.train_csv = splits_dir / "train.csv"
        self.val_csv   = splits_dir / "val.csv"
        self.test_csv  = splits_dir / "test.csv"
        self.image_size = (cfg["image"]["height"], cfg["image"]["width"])
        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = cfg["training"]["num_workers"]

    def setup(self, stage: Optional[str] = None):
        aug_cfg = self.cfg["augmentation"]
        self.train_ds = OCTDataset(
            self.train_csv,
            image_size=self.image_size,
            transform=get_train_transforms(self.image_size, aug_cfg),
        )
        self.val_ds = OCTDataset(
            self.val_csv,
            image_size=self.image_size,
            transform=get_val_transforms(self.image_size),
        )
        self.test_ds = OCTDataset(
            self.test_csv,
            image_size=self.image_size,
            transform=get_val_transforms(self.image_size),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.cfg["training"]["pin_memory"],
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.cfg["training"]["pin_memory"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,  # one at a time for evaluation
            shuffle=False,
            num_workers=self.num_workers,
        )