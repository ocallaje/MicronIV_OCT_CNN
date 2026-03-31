"""
trainer.py
──────────
PyTorch Lightning module for OCT segmentation training.

Handles:
  - Forward pass + loss computation
  - Optimiser + LR scheduler setup
  - Metric logging (Dice, IoU) per epoch
  - Optional image logging (overlay predictions on OCT)
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex

from src.models.unet import build_model
from src.training.losses import CombinedLoss


class OCTSegmentationModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Model
        self.model = build_model(cfg)

        # Loss
        loss_cfg = cfg["loss"]
        self.criterion = CombinedLoss(
            dice_weight=loss_cfg["dice_weight"],
            bce_weight=loss_cfg["bce_weight"],
        )

        # Metrics (torchmetrics handles batching correctly)
        self.train_dice = DiceScore(num_classes=2, include_background=False, average="macro")
        self.val_dice   = DiceScore(num_classes=2, include_background=False, average="macro")
        self.val_iou    = JaccardIndex(task="binary", threshold=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ──────────────────────────────────────────────────────────────────────────
    #  Steps
    # ──────────────────────────────────────────────────────────────────────────

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]   # (B, 1, H, W)
        masks  = batch["mask"]    # (B, 1, H, W)  values {0, 1}
        logits = self(images)     # (B, 1, H, W)  raw logits
        loss   = self.criterion(logits, masks)
        probs  = torch.sigmoid(logits)
        return loss, probs, masks

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, probs, masks = self._shared_step(batch)

        # DiceScore needs: preds (B, num_classes, H, W) one-hot, target (B, H, W) long
        preds_onehot = self._to_onehot(probs)          # (B, 2, H, W)
        target_onehot = self._mask_to_onehot(masks)

        self.train_dice(preds_onehot, target_onehot)

        self.log("train_loss", loss,            on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _to_onehot(self, probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert (B, 1, H, W) sigmoid probs → (B, 2, H, W) one-hot."""
        fg = (probs[:, 0] > threshold).long()  # (B, H, W)
        bg = 1 - fg
        return torch.stack([bg, fg], dim=1).float()
    
    def _mask_to_onehot(self, masks: torch.Tensor) -> torch.Tensor:
        """Convert (B, 1, H, W) float mask → (B, 2, H, W) one-hot."""
        fg = masks[:, 0].long()   # (B, H, W)
        bg = 1 - fg
        return torch.stack([bg, fg], dim=1).float()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, probs, masks = self._shared_step(batch)

        #self.val_dice(probs, masks.int())
        preds_onehot = self._to_onehot(probs)          # (B, 2, H, W)
        target_onehot = self._mask_to_onehot(masks)            # (B, H, W)

        self.val_dice(preds_onehot, target_onehot)
        self.val_iou(probs,  masks.int())

        self.log("val_loss", loss,          on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou",  self.val_iou,  on_step=False, on_epoch=True)

        # Log overlay images every 5 epochs (first batch only)
        if (
            batch_idx == 0
            and self.cfg["logging"]["log_images"]
            and self.current_epoch % 5 == 0
        ):
            self._log_overlay_images(batch["image"], probs, masks)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        loss, probs, masks = self._shared_step(batch)
        preds = (probs > self.cfg["inference"]["threshold"]).float()

        # Per-image Dice for test reporting
        dice = self._per_image_dice(preds, masks)

        return {
            "test_loss":  loss.item(),
            "test_dice":  dice,
            "image_path": batch["image_path"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Optimiser
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        train_cfg = self.cfg["training"]
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        scheduler_name = train_cfg.get("lr_scheduler", "cosine").lower()

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=train_cfg["epochs"],
                eta_min=train_cfg["learning_rate"] * 0.01,
            )
            return {"optimizer": optimiser,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, mode="max", patience=5, factor=0.5
            )
            return {"optimizer": optimiser,
                    "lr_scheduler": {"scheduler": scheduler,
                                     "monitor": "val_dice",
                                     "interval": "epoch"}}

        return optimiser

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _per_image_dice(
        self, preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
    ) -> float:
        preds   = preds.view(preds.size(0), -1).float()
        targets = targets.view(targets.size(0), -1).float()
        intersection = (preds * targets).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + smooth
        )
        return dice.mean().item()

    def _log_overlay_images(
        self,
        images: torch.Tensor,
        probs:  torch.Tensor,
        masks:  torch.Tensor,
        n: int = 4,
    ) -> None:
        """Log up to n overlay images to the Lightning logger."""
        try:
            import torchvision.utils as vutils

            imgs   = images[:n].repeat(1, 3, 1, 1)   # (N, 3, H, W) — grayscale → RGB
            pred_overlay = probs[:n].repeat(1, 3, 1, 1)
            gt_overlay   = masks[:n].repeat(1, 3, 1, 1)

            grid = vutils.make_grid(
                torch.cat([imgs, gt_overlay, pred_overlay], dim=0),
                nrow=n,
                normalize=True,
            )
            self.logger.experiment.add_image(
                "val_overlay (top: image | mid: GT | bot: pred)",
                grid,
                global_step=self.current_epoch,
            )
        except Exception:
            pass  # Logger may not support images — silently skip