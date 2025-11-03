# GeoSeg-Kathe: Semantic Segmentation for Biodiversity Land Cover Classification

A comprehensive deep learning framework for semantic segmentation of remote sensing imagery, specifically designed for biodiversity and land cover classification tasks. This repository implements multiple state-of-the-art segmentation architectures including UNetFormer, FTUNetFormer, and DCSwin. [BASED ON LIBO WANG'S RESEARCH.](https://github.com/WangLibo1995/GeoSeg). None of this would have been possible without his research! Please see his for original repository. 

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Key Files and Directories](#key-files-and-directories)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluation](#evaluation)
- [Model Architectures](#model-architectures)
- [Configuration](#configuration)
- [Class Definitions](#class-definitions)

## Overview

This repository provides a PyTorch Lightning-based framework for training and evaluating semantic segmentation models on remote sensing data. The primary focus is on biodiversity land cover classification with support for both standard PNG images and geospatial TIFF formats.

**Supported Classes:**
1. Forest land
2. Grassland
3. Cropland
4. Settlement
5. Seminatural Grassland

## Features

- **Multiple Architectures**: UNetFormer, FTUNetFormer (Feature-Transform UNetFormer), and DCSwin (Dual-Channel Swin Transformer)
- **Geospatial Data Support**: Native support for GeoTIFF files with proper handling of multi-band imagery
- **Automated Hyperparameter Tuning**: Systematic hyperparameter search with checkpoint management
- **Test Time Augmentation**: Support for both flip (lr) and multi-scale (d4) TTA
- **Comprehensive Evaluation**: Detailed metrics including mIoU, F1-score, OA, and per-class performance
- **Large Image Inference**: Patch-based processing for huge images with seamless reconstruction

## Installation
```bash
# Create conda environment
conda create -n geoseg python=3.11
conda activate geoseg

# Install dependencies
pip install -r requirements.txt

# Additional requirements
pip install pytorch-lightning==2.3.0
pip install gdal rasterio  # For GeoTIFF support
```

## Dataset Structure
```
data/
├── Biodiversity/
│   ├── Train/
│   │   └── Rural/
│   │       ├── images_png/
│   │       └── masks_png/
│   ├── Val/
│   │   └── Rural/
│   │       ├── images_png/
│   │       └── masks_png/
│   └── Test/
│       └── Rural/
│           └── images_png/
│
└── Biodiversity_tiff/
    ├── Train/
    │   ├── images/  # .tif files
    │   └── masks/   # .png files
    ├── Val/
    └── Test/
```

## Key Files and Directories

### Core Training Files
- **`train_supervision.py`**: Main training script using PyTorch Lightning
- **`config/biodiversity/*.py`**: Configuration files for different models (unetformer.py, ftunetformer.py, dcswin.py)
- **`config/biodiversity_tiff/*.py`**: Configuration files for GeoTIFF datasets

### Model Architectures
- **`geoseg/models/UNetFormer.py`**: UNetFormer architecture with ResNet18 backbone
- **`geoseg/models/FTUNetFormer.py`**: Feature-Transform UNetFormer with Swin Transformer backbone
- **`geoseg/models/DCSwin.py`**: Dual-Channel Swin Transformer

### Dataset Handlers
- **`geoseg/datasets/biodiversity_dataset.py`**: Dataset loader for PNG images
- **`geoseg/datasets/biodiversity_tiff_dataset.py`**: Dataset loader for GeoTIFF files with proper geospatial handling
- **`geoseg/datasets/transform.py`**: Custom augmentation transforms

### Loss Functions
- **`geoseg/losses/`**: Comprehensive loss function library including:
  - `joint_loss.py`: Combined CE + Dice loss
  - `useful_loss.py`: UNetFormer-specific loss with auxiliary outputs
  - `dice.py`, `focal.py`, `lovasz.py`: Various segmentation losses

### Inference Scripts
- **`outputs/biodiversity_test_direct.py`**: Inference with custom checkpoint paths
- **`outputs/biodiversity_tiff_test_direct.py`**: Inference for GeoTIFF data
- **`inference_huge_image.py`**: Patch-based inference for large images

### Hyperparameter Tuning
- **`hyperparameter_unetformer.py`**: Grid search for UNetFormer
- **`hyperparameter_ftunetformer_tiff.py`**: Grid search for FTUNetFormer on GeoTIFF data
- **`hyperparameter_dcswin.py`**: Grid search for DCSwin

### Utilities
- **`tools/metric.py`**: Evaluation metrics (IoU, F1, OA, Precision, Recall)
- **`tools/biodiversity_mask_convert.py`**: Convert masks to proper class indices
- **`data_diagnostics.py`**: Dataset validation and diagnostics
- **`evaluation/model_metrics.py`**: Generate comprehensive evaluation reports with visualizations
- **`evaluation/process_metrics.py`**: Process and rank model performance across experiments

## Usage

### Training

#### Basic Training
```bash
# Train UNetFormer on PNG dataset
python train_supervision.py -c config/biodiversity/unetformer.py

# Train FTUNetFormer on GeoTIFF dataset
python train_supervision.py -c config/biodiversity_tiff/ftunetformer.py

# Train DCSwin
python train_supervision.py -c config/biodiversity/dcswin.py
```

#### Resume Training
```bash
python train_supervision.py -c config/biodiversity/unetformer.py \
    --resume_ckpt_path model_weights/biodiversity/checkpoint.ckpt
```

### Inference

#### Standard Inference
```bash
# Inference with config default weights
python outputs/biodiversity_test.py \
    -c config/biodiversity/unetformer.py \
    -o predictions/output \
    --rgb

# Inference with specific checkpoint
python outputs/biodiversity_test_direct.py \
    -c config/biodiversity/unetformer.py \
    -w model_weights/biodiversity/best_model.ckpt \
    -i data/Biodiversity/Test/Rural/images_png \
    -o predictions/output \
    --rgb
```

#### GeoTIFF Inference
```bash
python outputs/biodiversity_tiff_test_direct.py \
    -c config/biodiversity_tiff/unetformer.py \
    -w model_weights/biodiversity_tiff/best_model.ckpt \
    -i data/Biodiversity_tiff/Test/images \
    -o predictions/output \
    --rgb
```

#### Test Time Augmentation Options
- `None`: No augmentation
- `lr`: Horizontal and vertical flips
- `d4`: Multi-scale augmentation (0.75x, 1.0x, 1.25x, 1.5x)

### Hyperparameter Tuning
```bash
# UNetFormer hyperparameter search
python hyperparameter_unetformer.py

# FTUNetFormer on GeoTIFF data
python hyperparameter_ftunetformer_tiff.py

# DCSwin hyperparameter search
python hyperparameter_dcswin.py
```

Hyperparameter ranges are defined in each script:
- Learning rates: 5e-4 to 3e-5
- Batch sizes: 4, 8, 16
- Weight decays: 1e-2 to 1e-1
- Scales: 0.75, 1.0, 1.25

Results are saved in structured directories:
```
model_weights/biodiversity/
└── biodiversityL5e-04BL1e-04W1e-02BW1e-02B16E30S1.00/
    ├── config.txt
    ├── output.txt
    ├── last.ckpt
    └── best_epoch_*.ckpt
```

### Evaluation

#### Generate Evaluation Reports
```bash
# Evaluate all models in a directory
python evaluation/model_metrics.py

# This generates for each model:
# - confusion_matrix.png
# - class_iou_scores.png
# - class_f1_scores.png
# - evaluation_report.txt
```

#### Process and Rank Results
```bash
# Extract and rank metrics from all experiments
python evaluation/process_metrics.py

# Generates:
# - miou_scores.json
# - f1_scores.json
# - oa_scores.json
# - validation_metrics_sorted.txt
```

#### Dataset Diagnostics
```bash
python data_diagnostics.py \
    --data-root data/Biodiversity/Train \
    --img-dir images_png \
    --mask-dir masks_png \
    --img-suffix .png \
    --mask-suffix .png
```

## Model Architectures

### UNetFormer
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Decoder**: Global-Local Attention with window size 8
- **Features**: Auxiliary loss support, lightweight, fast training
- **Best for**: Balanced performance and speed

### FTUNetFormer
- **Backbone**: Swin Transformer (pretrained)
- **Decoder**: Enhanced decoder with 256 channels
- **Features**: Feature transformation, better for complex patterns
- **Best for**: High accuracy requirements

### DCSwin
- **Backbone**: Swin Transformer Base/Small
- **Features**: Dual-channel spatial attention, channel attention
- **Best for**: Large-scale features, complex spatial relationships

## Configuration

### Key Configuration Parameters
```python
# Training hyperparameters
max_epoch = 30
train_batch_size = 16
val_batch_size = 16
lr = 5e-4
backbone_lr = 5e-5
weight_decay = 0.01

# Model
num_classes = 6
classes = ('Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland')

# Loss
loss = UnetFormerLoss(ignore_index=0)  # or JointLoss for FTUNetFormer/DCSwin

# Optimizer
optimizer = Lookahead(torch.optim.AdamW(...))
lr_scheduler = CosineAnnealingLR(...)

# Checkpointing
monitor = 'val_mIoU'  # or 'val_F1'
save_top_k = 1
save_last = True
```

## Class Definitions

### Color Palette (RGB)
```python
PALETTE = [
    [250, 62, 119],   # Forest land
    [168, 232, 84],   # Grassland
    [242, 180, 92],   # Cropland
    [116, 116, 116],  # Settlement
    [255, 214, 33]    # Seminatural Grassland
]
```

### Mask Values
- Background/Ignore: 0
- Forest land: 1
- Grassland: 2
- Cropland: 3
- Settlement: 4
- Seminatural Grassland: 5

## Metrics

The framework computes:
- **mIoU**: Mean Intersection over Union (excluding background)
- **F1 Score**: Harmonic mean of precision and recall
- **OA**: Overall Accuracy
- **Per-class IoU**: Individual class performance
- **Confusion Matrix**: Detailed class confusion analysis

## Output Formats

### Prediction Masks
- **Grayscale**: Class indices (0-5)
- **RGB**: Colored visualization using defined palette
- **Format**: PNG for standard data, TIFF for geospatial data

### Evaluation Reports
```
=== Evaluation Report ===

Overall Accuracy: 0.8542
Mean IoU (excluding background): 0.7321
Mean F1 (excluding background): 0.8456

Per-Class Metrics:
--------------------------------------------------

Forest land:
IoU: 0.8234
F1: 0.9021
Pixels: 152384

[...]
```

## Tips and Best Practices

1. **Data Preparation**: Use `data_diagnostics.py` to validate dataset integrity before training
2. **Mask Conversion**: Run `tools/biodiversity_mask_convert.py` to ensure masks have correct class indices
3. **Hyperparameter Tuning**: Start with smaller batch sizes (4-8) for memory-constrained GPUs
4. **GeoTIFF Handling**: The tiff dataset classes handle normalization and band conversion automatically
5. **Model Selection**: UNetFormer for speed, FTUNetFormer for accuracy, DCSwin for large-scale features
6. **Checkpoints**: Models save best checkpoints based on validation mIoU or F1 score
7. **TTA**: Use multi-scale TTA (d4) for final predictions to boost performance by 1-2%


## Citation

If you use this code in your research, please cite the original GeoSeg repository and relevant papers for the architectures used.

## License

GNU GENERAL PUBLIC LICENSE
