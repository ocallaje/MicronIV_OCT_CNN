# OCT Retinal Layer Segmentation

A deep learning pipeline for automatic segmentation of mouse retinal OCT images using U-Net. Starting with ILM→RPE total retinal segmentation, extensible to individual layer boundaries.

## Overview

```
Input: OCT grayscale image
Output: Binary mask (ILM to RPE) → thickness profile (µm/pixel)
```

**Pipeline:**
1. Convert boundary coordinates → binary masks
2. Train U-Net (ResNet34 encoder, ImageNet pretrained)
3. Predict masks on new images
4. Extract per-column thickness profiles

---

## Repo Structure

```
oct_segmentation/
├── configs/
│   └── default.yaml          # All hyperparameters and paths
├── data/
│   ├── raw/                  # Original OCT images (.tif / .png)
│   ├── masks/                # Generated binary masks
│   └── splits/               # Train/val/test CSVs (split by animal)
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   ├── augmentation.py   # Albumentations pipeline
│   │   └── prepare_masks.py  # Boundary coords → masks
│   ├── models/
│   │   └── unet.py           # U-Net wrapper (smp)
│   ├── training/
│   │   ├── trainer.py        # PyTorch Lightning module
│   │   └── losses.py         # Dice + BCE combined loss
│   ├── inference/
│   │   ├── predict.py        # Single image / batch prediction
│   │   └── thickness.py      # Mask → thickness profile
│   └── utils/
│       ├── visualise.py      # Overlay masks on OCT images
│       └── metrics.py        # Dice, IoU, boundary error
├── scripts/
│   ├── 01_prepare_data.py    # Run mask generation
│   ├── 02_create_splits.py   # Create train/val/test splits
│   ├── 03_train.py           # Launch training
│   └── 04_evaluate.py        # Evaluate on test set
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── setup.py
```

---

## Quickstart

### 1. Install

```bash
git clone <your-repo>
cd oct_segmentation
pip install -e .
```

### 2. Prepare your data

Your boundary coordinate files should be structured as:

```
data/raw/
├── animal_01/
│   ├── image_001.tif
│   ├── image_001_boundaries.csv   # columns: x, ilm_y, rpe_y
│   ├── image_002.tif
│   └── ...
└── animal_02/
    └── ...
```

Boundary CSV format:
```csv
x,ilm_y,rpe_y
0,45,210
1,44,211
...
```

### 3. Generate masks

```bash
python scripts/01_prepare_data.py --data_dir data/raw --output_dir data/masks
```

### 4. Create splits

```bash
python scripts/02_create_splits.py --data_dir data/raw --output_dir data/splits
```

### 5. Train

```bash
python scripts/03_train.py --config configs/default.yaml
```

### 6. Predict on new images

```bash
python scripts/04_evaluate.py \
    --checkpoint checkpoints/best_model.ckpt \
    --image path/to/new_oct.tif \
    --output results/
```

---

## Expected Input Format

- **Images**: Grayscale OCT B-scans, any resolution (resized internally)
- **Boundaries**: CSV with per-column ILM and RPE y-coordinates
- **Scale**: Set `um_per_pixel` in config for physical thickness units

---

## Extending to More Layers

To add segmentation of additional layers (e.g. GCL, INL, ONL):

1. Add extra boundary columns to your CSVs
2. Set `num_classes: N` in `configs/default.yaml`
3. Update `src/data/prepare_masks.py` to generate multi-class masks
4. Retrain — the U-Net architecture scales automatically

---

## Dependencies

- PyTorch >= 2.0
- segmentation-models-pytorch
- albumentations
- PyTorch Lightning
- OpenCV, NumPy, pandas, matplotlib