# Label Contract (Semantic Segmentation)

**Goal:** Make every experiment comparable. This document fixes class IDs, mask format, resizing rules, training flags, and how we compute metrics.

## 1) Classes & IDs (fixed)
| ID | Name                  | Used in loss/metrics?            |
|---:|-----------------------|----------------------------------|
| 0  | Background / Ignore   | **Ignored** (do not train/eval)  |
| 1  | Forest land           | Yes                              |
| 2  | Grassland             | Yes                              |
| 3  | Cropland              | Yes                              |
| 4  | Settlement            | Yes (often sparse/thin)          |
| 5  | Seminatural Grassland | Yes                              |

- `NUM_CLASSES = 6`
- `IGNORE_INDEX = 0`

## 2) Mask format
- **Type:** single-channel **uint8** PNG.
- **Valid values:** **{0,1,2,3,4,5}** only. No 255. No RGB color masks.
- **Size:** exact **H×W** match to its paired image.

## 3) Resizing / tiling
- **Masks:** **NEAREST** interpolation **only** (training & inference).
- **Images:** bilinear/bicubic is fine.
- **Sliding inference:** tile with **≥25% overlap**; blend to avoid seams.

## 4) Training flags
- Loss functions must set `ignore_index = 0`.
- Recommended losses for better overlap & class imbalance:
  - **CrossEntropy + Dice**, or
  - **Lovász-Softmax** (direct IoU surrogate).

If you use class weights, document them in the run config.

## 5) Evaluation (source of truth)
- Compute metrics over pixels where **GT ≠ 0**.
- **Per-class IoU (1..5):** `IoU_c = TP_c / (TP_c + FP_c + FN_c)`.
- **mIoU:** mean of IoU over **{1,2,3,4,5}** (background excluded).
- Also report per-class **F1** and **Overall Accuracy (OA)**.
- Pair files by **stem**; error on size mismatch or missing pairs.

## 6) File layout (example)
```
data/
  Train/
    images/*.tif
    masks/*.png    # uint8, indices {0..5}
  Val/
    images/*.tif
    masks/*.png
splits/
  iid/{train.txt,val.txt}
  xdom/{train_limerick.txt,val_dencol.txt,...}
```

### Handy constants (for code)
```python
IGNORE_INDEX = 0
CLASS_IDS = [1, 2, 3, 4, 5]
NUM_CLASSES = 6
```

## 7) (Optional) Visualization palette (for QA only)
0 Background: [11,246,210]
1 Forest: [250,62,119]
2 Grassland: [168,232,84]
3 Cropland: [242,180,92]
4 Settlement: [116,116,116]
5 Seminatural:[255,214,33]


## 8) Quick QA checklist (must pass)
- [ ] Masks are uint8, values ⊆ {0..5}
- [ ] Image/mask shapes identical (H×W)
- [ ] No mask ever resized with bilinear (edges are crisp)
- [ ] Loss uses `ignore_index=0`
- [ ] Evaluator ignores **GT==0** and reports **mIoU over 1–5**

## 9) “Definition of done” for any run
- Attach config hash, split file, seed.
- Provide `evaluation_report.txt` + `metrics.json` from the **single evaluator**.
- Include per-class IoU table, overall mIoU (1–5), and a confusion matrix plot.
