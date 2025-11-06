# Multi-Stage Training Pipeline with OpenEarthMap Pretraining

## Overview
This pipeline implements a 3-stage training strategy to improve performance:
1. **Pretrain** student on combined data (target + OpenEarthMap)
2. **Finetune** with knowledge distillation on target-only data
3. **Ensemble** both models with multi-scale TTA

## Data Setup

### Combined Dataset
- **Location**: `data/biodiversity_combined/`
- **Train**: 4615 samples (1615 target + 3000 OpenEarthMap)
- **Val**: 500 samples (OpenEarthMap val set)
- **Created via**: `create_combined_dataset.py` (uses symlinks, original data unchanged)

### File Structure
```
data/biodiversity_combined/
├── Train/
│   ├── images_png/
│   │   ├── oem_*.png         (3000 OpenEarthMap samples)
│   │   └── target_*.png      (1615 biodiversity samples)
│   └── masks_png/
│       ├── oem_*.png         (remapped to 6 classes)
│       └── target_*.png
└── Val/
    ├── images_png/
    └── masks_png/
```

## Training Pipeline

### Step 1: Pretrain on Combined Data
**Goal**: Learn general features from larger dataset (4615 samples)

**Config**: `config/biodiversity/step1_pretrain_combined.py`

**Key Settings**:
- Resolution: 512x512
- Epochs: 50
- Loss: CE + Dice (no KD)
- Batch size: 2 (VRAM constraint)
- LR: 6e-4
- Data: Target + OpenEarthMap combined

**Run**:
```bash
python train_supervision.py -c config/biodiversity/step1_pretrain_combined.py
```

**Output**: `model_weights/biodiversity/student_pretrain_all/last.ckpt`

**Expected Time**: ~10-12 hours (depending on GPU)

---

### Step 2: Finetune on Target with KD
**Goal**: Specialize to target domain using teacher knowledge

**Config**: `config/biodiversity/step2_finetune_target_kd.py`

**Key Settings**:
- Resolution: 512x512
- Epochs: 25 (5 warmup + 20 KD)
- Loss: α·CE + (1-α)·KD (α=0.7, T=3)
- Confidence masking: Ignore teacher where max_prob < 0.5
- Warmup: 5 epochs CE-only, then add KD
- LR: 1e-4 → 3e-5 (cosine decay)
- Data: Target-only (1615 samples)
- Pretrained from: `student_pretrain_all/last.ckpt`

**Run**:
```bash
python train_kd.py -c config/biodiversity/step2_finetune_target_kd.py
```

**Output**: `model_weights/biodiversity/student_ft_target_kd/last.ckpt`

**Expected Time**: ~4-5 hours

---

### Step 3: Ensemble Inference
**Goal**: Combine predictions from both models with multi-scale TTA

**Models**:
1. `student_target_only.ckpt` (baseline, 83.93% mIoU)
2. `student_ft_target_kd.ckpt` (finetuned with KD)

**TTA Strategy**:
- Scales: 0.75, 1.0, 1.25
- Flips: Horizontal + Vertical
- Total augmentations: 3 scales × 3 augmentations × 2 models = 18 predictions averaged

**Run**:
```bash
python ensemble_inference.py \
  --input-dir data/biodiversity/Test/images_png \
  --output-dir outputs/ensemble_predictions \
  --checkpoint1 model_weights/biodiversity/ensemble_checkpoints/student_target_only.ckpt \
  --checkpoint2 model_weights/biodiversity/ensemble_checkpoints/student_ft_target_kd.ckpt \
  --scales 0.75,1.0,1.25 \
  --save-rgb
```

**Expected Improvement**: +2-4% over baseline (85-88% mIoU target)

---

## Files Created

### Datasets
- `geoseg/datasets/biodiversity_combined_dataset.py` - Dataset for 512x512 combined training

### Configs
- `config/biodiversity/step1_pretrain_combined.py` - Pretraining config
- `config/biodiversity/step2_finetune_target_kd.py` - Finetuning with KD config

### Scripts
- `create_combined_dataset.py` - Create combined dataset with symlinks
- `ensemble_inference.py` - Ensemble prediction with multi-scale TTA

### Modified
- `geoseg/utils/kd_utils.py` - Added confidence masking to `compute_kd_loss()`

### Checkpoints
- `model_weights/biodiversity/ensemble_checkpoints/student_target_only.ckpt` - Baseline (83.93%)

---

## Implementation Details

### Knowledge Distillation with Confidence Masking
The finetuning stage uses adaptive KD loss:

```python
# Warmup phase (epochs 0-4): CE + Dice only
loss = CE + Dice

# KD phase (epochs 5-24): Add teacher knowledge
teacher_prob = teacher(x)
confidence_mask = (teacher_prob.max(dim=1) >= 0.5)  # Only use confident predictions
kd_loss = KL_div(student, teacher) * confidence_mask
loss = 0.3 * CE + 0.7 * kd_loss
```

### Multi-Scale TTA
Ensemble inference averages logits across:
- 3 scales (0.75, 1.0, 1.25)
- 3 augmentations (none, H-flip, V-flip)
- 2 models (baseline + finetuned)
= 18 predictions per pixel

### Resolution Strategy
- Training: 512x512 (to fit more data in VRAM)
- Inference: Original resolution with multi-scale TTA
- This balances training efficiency with inference quality

---

## Expected Results

### Baseline (Current)
- Model: FTUNetFormer + KD
- mIoU: 83.93%
- Data: Target-only (1615 samples)

### After Pretraining (Step 1)
- Expected: ~82-84% mIoU on target val
- Benefit: Better feature representations from 4615 samples

### After Finetuning (Step 2)  
- Expected: ~84-86% mIoU on target val
- Benefit: Teacher knowledge + domain-specific tuning

### After Ensemble (Step 3)
- Expected: ~85-88% mIoU on target test
- Benefit: Model diversity + multi-scale TTA

**Target Goal**: 95% mIoU
**Gap Remaining**: ~7-10% (will need additional strategies like OpenMapData pre-training)

---

## Next Steps After This Pipeline

If results are ~85-88% mIoU:
1. **Train additional architectures** (Swin, SegFormer) for ensemble
2. **Pseudo-labeling** on unlabeled target data
3. **OpenMapData pre-training** (major effort, +5-10%)
4. **Class-weighted loss** to boost Forest/Settlement
5. **Advanced augmentations** (CutMix, MixUp, etc.)

---

## Monitoring Training

### Step 1 (Pretraining)
Watch for:
- Val mIoU should reach ~82-84%
- Training loss should decrease steadily
- Val loss should be close to train loss (good generalization)

### Step 2 (Finetuning)
Watch for:
- First 5 epochs: CE-only warmup
- After epoch 5: KD loss should activate
- Val mIoU should gradually improve to ~84-86%
- Check confidence masking: should ignore ~20-30% of teacher predictions

### Ensemble
- Compare single model vs ensemble
- Check per-class improvements (especially Forest/Settlement)
- Visualize predictions to verify quality

---

## Troubleshooting

### OOM During Training
- Reduce batch_size from 2 to 1
- Reduce resolution from 512 to 384
- Disable gradient accumulation if enabled

### Pretraining Not Improving
- Check that OpenEarthMap masks are correctly remapped (0-5)
- Verify data augmentation is working
- Try longer training (70-80 epochs)

### Finetuning Worse Than Baseline
- Check confidence threshold (try 0.3-0.7 range)
- Increase warmup epochs (try 7-10)
- Adjust α (try 0.5-0.8 range)
- Verify pretrained checkpoint is loaded correctly

### Ensemble Not Improving
- Ensure both checkpoints are from different training runs
- Try more aggressive TTA scales (0.5, 0.75, 1.0, 1.25, 1.5)
- Check that models have different predictions (diversity)

---

## Quick Start Commands

```bash
# 1. Create combined dataset
python create_combined_dataset.py

# 2. Pretrain on combined data
python train_supervision.py -c config/biodiversity/step1_pretrain_combined.py

# 3. Finetune with KD (after step 2 completes)
python train_kd.py -c config/biodiversity/step2_finetune_target_kd.py

# 4. Copy finetuned checkpoint for ensemble
cp model_weights/biodiversity/student_ft_target_kd/last.ckpt \
   model_weights/biodiversity/ensemble_checkpoints/student_ft_target_kd.ckpt

# 5. Run ensemble inference
python ensemble_inference.py \
  --input-dir data/biodiversity/Test/images_png \
  --output-dir outputs/ensemble_predictions \
  --save-rgb

# 6. Evaluate
python evaluation/model_metrics.py \
  --pred-dir outputs/ensemble_predictions \
  --gt-dir data/biodiversity/Test/masks_png
```

---

## Notes

- All original data remains unchanged (symlinks used for combined dataset)
- Checkpoints are saved in `model_weights/biodiversity/`
- Can resume training by setting `resume_ckpt_path` in configs
- Monitor training with TensorBoard: `tensorboard --logdir lightning_logs/`
