"""
unet.py
───────
U-Net model wrapper using segmentation-models-pytorch.

Supports binary (ILM→RPE) and multi-class (individual layers) segmentation.
Easily swap encoder backbones without changing any other code.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """
    Build a segmentation model from config.

    Supported architectures:
        'unet'         → U-Net (recommended starting point)
        'unetplusplus' → U-Net++ (marginally better, more memory)

    Supported encoders (ImageNet pretrained):
        'resnet34'          — fast, great for small datasets
        'resnet50'          — more capacity
        'efficientnet-b3'   — efficient, strong performance
        'mit_b2'            — SegFormer encoder (if you have lots of data)

    Parameters
    ----------
    cfg : full config dict (loaded from YAML)

    Returns
    -------
    model : nn.Module
    """
    model_cfg = cfg["model"]
    arch      = model_cfg["architecture"].lower()
    num_classes = model_cfg["num_classes"]

    # smp uses 'activation=None' for raw logits — we handle activation in loss/inference
    common_kwargs = dict(
        encoder_name    = model_cfg["encoder"],
        encoder_weights = model_cfg["encoder_weights"],
        in_channels     = model_cfg["in_channels"],
        classes         = num_classes,
        activation      = None,  # raw logits → use BCEWithLogitsLoss or CrossEntropyLoss
    )

    if arch == "unet":
        model = smp.Unet(**common_kwargs)
    elif arch == "unetplusplus":
        model = smp.UnetPlusPlus(**common_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'unet' or 'unetplusplus'.")

    # Log parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch} | Encoder: {model_cfg['encoder']} | "
          f"Params: {n_params / 1e6:.1f}M | Classes: {num_classes}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: load a trained checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, cfg: dict) -> nn.Module:
    """
    Load model weights from a PyTorch Lightning checkpoint.

    Usage:
        model = load_model_from_checkpoint("checkpoints/best_model.ckpt", cfg)
        model.eval()
    """
    from src.training.trainer import OCTSegmentationModule

    module = OCTSegmentationModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
    )
    return module.model