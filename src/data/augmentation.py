"""
augmentation.py
───────────────
Albumentations augmentation pipelines for OCT images.

Key OCT-specific rules:
  ✓  Horizontal flip   — left/right reversal is valid
  ✗  Vertical flip     — would invert layer ordering (ILM becomes bottom)
  ✓  Brightness/contrast — intensity varies between sessions
  ✓  Elastic transform — simulates mild tissue/fixation deformation
  ✗  Large crops       — avoid cutting off layers at image edges
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: tuple[int, int], aug_cfg: dict) -> A.Compose:
    """
    Full augmentation pipeline for training.

    Parameters
    ----------
    image_size : (H, W) — output size
    aug_cfg    : augmentation section from config YAML
    """
    H, W = image_size
    return A.Compose(
        [
            # ── Spatial ────────────────────────────────────────────────────
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
            # NOTE: No VerticalFlip — would break anatomical layer ordering.

            # Mild elastic deformation — simulates retinal curvature variation
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                p=aug_cfg.get("elastic_transform_p", 0.3),
            ),

            # Small shifts / scales — no rotation (layers should stay horizontal)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=3,      # very small rotation only
                border_mode=0,
                p=0.4,
            ),

            # ── Intensity ──────────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                contrast_limit=aug_cfg.get("contrast_limit", 0.2),
                p=0.6,
            ),

            # Simulate speckle noise common in OCT
            A.GaussNoise(
                var_limit=(5.0, 30.0),
                p=aug_cfg.get("gaussian_noise_p", 0.3),
            ),

            # Simulate signal dropout / shadowing artifacts
            A.CoarseDropout(
                max_holes=4,
                max_height=H // 16,
                max_width=W // 8,
                fill_value=0,
                p=0.15,
            ),

            # Slight blur — simulates focus variation
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),

            # Ensure correct size after augmentation
            A.Resize(H, W),
        ],
        # Ensure the same spatial transforms apply to both image and mask
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(image_size: tuple[int, int]) -> A.Compose:
    """
    Minimal pipeline for validation / test — resize only.
    No augmentation to ensure reproducible evaluation.
    """
    H, W = image_size
    return A.Compose(
        [A.Resize(H, W)],
        additional_targets={"mask": "mask"},
    )


def get_tta_transforms(image_size: tuple[int, int]) -> list[A.Compose]:
    """
    Test-time augmentation (TTA) ensemble.
    Returns a list of transforms; run inference on each and average predictions.

    Only horizontal flip is safe for OCT TTA.
    """
    H, W = image_size
    base = A.Resize(H, W)
    return [
        A.Compose([base]),                          # original
        A.Compose([A.HorizontalFlip(p=1.0), base]), # flipped
    ]