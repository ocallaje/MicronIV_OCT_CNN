"""
losses.py
─────────
Combined Dice + BCE loss for binary segmentation.

Why both?
- BCE:  pixel-wise, ensures the model learns fine boundaries
- Dice: region-overlap based, handles class imbalance (retina vs background)
        and directly optimises the metric we care about

For multi-class extension, swap to CrossEntropyLoss + multi-class Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Works on raw logits (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()

class AsymmetricBCE(nn.Module):
    """
    Penalise false positives (predicting mask where there is none) 
    more heavily than false negatives. Corrects upward ILM bias.

    fp_weight > 1.0 = penalise over-prediction (too wide/high)
    fn_weight > 1.0 = penalise under-prediction (too narrow)
    """
    def __init__(self, fp_weight=2.0, fn_weight=1.0):
        super().__init__()
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # False positive: predicted 1, truth is 0 (over-prediction)
        fp_loss = -targets * torch.log(probs + 1e-6)
        # False negative: predicted 0, truth is 1 (under-prediction)  
        fn_loss = -(1 - targets) * torch.log(1 - probs + 1e-6)

        loss = self.fn_weight * fp_loss + self.fp_weight * fn_loss
        return loss.mean()
    
class CombinedLoss(nn.Module):
    """
    Weighted sum of Dice loss and Binary Cross-Entropy loss.

    Parameters
    ----------
    dice_weight : float  (default 0.5)
    bce_weight  : float  (default 0.5)
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, fp_weight=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss()
        #self.bce_loss    = nn.BCEWithLogitsLoss()
        self.bce_loss    = AsymmetricBCE(fp_weight=fp_weight, fn_weight=1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce  = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-class extension (for individual layer segmentation)
# ─────────────────────────────────────────────────────────────────────────────

class MultiClassDiceLoss(nn.Module):
    """
    Mean Dice loss across all classes (excluding background class 0).
    Use when num_classes > 1.
    """

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.start_class = 1 if ignore_background else 0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W) raw logits
        targets : (B, H, W)    integer class labels
        """
        probs = F.softmax(logits, dim=1)
        dice_scores = []

        for c in range(self.start_class, self.num_classes):
            prob_c   = probs[:, c].reshape(probs.size(0), -1)
            target_c = (targets == c).float().reshape(targets.size(0), -1)
            intersection = (prob_c * target_c).sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (
                prob_c.sum(dim=1) + target_c.sum(dim=1) + self.smooth
            )
            dice_scores.append(1.0 - dice.mean())

        return torch.stack(dice_scores).mean()