# Knowledge Distillation Training Guide

## Overview
This guide explains how to run knowledge distillation training where an EfficientNet-B4 U-Net teacher model helps train a smaller FTUNetFormer student model.

## Setup

### Prerequisites
- Teacher model checkpoint: `pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth`
- Training data: `data/biodiversity/Train/`
- Validation data: `data/biodiversity/Val/`

### Configuration
The KD configuration is in: `config/biodiversity/ftunetformer_kd.py`

Key parameters:
- **kd_temperature**: 2.0 (softens probability distributions)
- **kd_alpha**: 0.7 (70% KD loss, 30% hard target loss)
- **Student model**: FTUNetFormer with 6 classes
- **Teacher model**: EfficientNet-B4 U-Net with 8 classes
- **Batch size**: 4
- **Learning rate**: 6e-4 (main), 6e-5 (backbone)
- **Epochs**: 45

## Running Knowledge Distillation

### Command
```bash
python train_kd.py -c config/biodiversity/ftunetformer_kd.py
```

### What Happens During Training
1. Teacher model is loaded from pretrained weights and frozen
2. Student model learns from both:
   - Hard targets (ground truth labels) - 30%
   - Soft targets (teacher predictions) - 70%
3. The mapping matrix handles the class mismatch (8→6 classes)
4. Metrics are logged to `lightning_logs/biodiversity/ftunetformer-kd-512-crop-ms-e45/`
5. Checkpoints saved to `model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/`

## Comparing with Baseline

### Baseline (No KD)
```bash
python train_supervision.py -c config/biodiversity/ftunetformer.py
```

### Knowledge Distillation
```bash
python train_kd.py -c config/biodiversity/ftunetformer_kd.py
```

### Expected Improvements
Knowledge distillation typically provides:
- Better generalization (1-3% mIoU improvement)
- Faster convergence
- More robust per-class performance
- Better handling of ambiguous cases

## Monitoring Training

Check training progress:
```bash
# View logs
tail -f lightning_logs/biodiversity/ftunetformer-kd-512-crop-ms-e45/version_*/metrics.csv

# TensorBoard (if available)
tensorboard --logdir lightning_logs/biodiversity/
```

## Troubleshooting

### If teacher checkpoint fails to load:
- Verify path: `pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth`
- Check if model has 8 output classes
- Ensure checkpoint format matches TeacherUNet architecture

### If CUDA out of memory:
- Reduce `train_batch_size` in config (currently 4)
- Reduce `val_batch_size`
- Use gradient accumulation

### If training diverges:
- Reduce `kd_alpha` (try 0.5 or 0.3)
- Reduce `kd_temperature` (try 1.5 or 1.0)
- Lower learning rate

## Class Mapping
The teacher's 8 classes map to student's 6 classes:
- Tree → Forest
- Rangeland → Grassland (70%) + SemiNatural (30%)
- Cropland → Cropland
- Developed → Settlement
- Road → Settlement
- Bareland → SemiNatural
- Water → Background
- Background → Background
