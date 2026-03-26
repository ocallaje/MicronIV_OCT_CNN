"""
03_train.py
───────────
Launch U-Net training with PyTorch Lightning.

Usage:
    python scripts/03_train.py --config configs/default.yaml

    # Resume from checkpoint:
    python scripts/03_train.py --config configs/default.yaml \
        --resume checkpoints/last.ckpt

    # Override config values on CLI:
    python scripts/03_train.py --config configs/default.yaml \
        --batch_size 4 --epochs 50 --lr 5e-5
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import OCTDataModule
from src.training.trainer import OCTSegmentationModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=Path, default="configs/default.yaml")
    p.add_argument("--resume",     type=str,  default=None, help="Resume from checkpoint")
    # CLI overrides
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--gpus",       type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.batch_size: cfg["training"]["batch_size"]    = args.batch_size
    if args.epochs:     cfg["training"]["epochs"]        = args.epochs
    if args.lr:         cfg["training"]["learning_rate"] = args.lr

    pl.seed_everything(42, workers=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    datamodule = OCTDataModule(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = OCTSegmentationModule.load_from_checkpoint(args.resume, cfg=cfg)
    else:
        model = OCTSegmentationModule(cfg)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cfg = cfg["checkpoint"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_cfg["dir"],
        filename="epoch{epoch:03d}_val_dice{val_dice:.4f}",
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_top_k=ckpt_cfg["save_top_k"],
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_dice",
        patience=cfg["training"]["early_stopping_patience"],
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    try:
        callbacks.append(RichProgressBar())
    except ImportError:
        pass

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger("logs", name=cfg["logging"]["project_name"])

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices=args.gpus or "auto",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        deterministic=False,  # True is slower but fully reproducible
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    print(f"\n✓ Training complete.")
    print(f"  Best checkpoint : {checkpoint_callback.best_model_path}")
    print(f"  Best val Dice   : {checkpoint_callback.best_model_score:.4f}")
    print(f"\nTo view training logs:\n  tensorboard --logdir logs/")


if __name__ == "__main__":
    main()