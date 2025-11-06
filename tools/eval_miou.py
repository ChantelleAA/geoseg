#!/usr/bin/env python3
import argparse, os, glob, json
import numpy as np
import cv2

def parse_args():
    p = argparse.ArgumentParser("Folder-based mIoU evaluator (ignore GT==0)")
    p.add_argument("--gt",   required=True, help="GT masks dir (uint8 indices)")
    p.add_argument("--pred", required=True, help="Pred masks dir (uint8 indices)")
    p.add_argument("--ignore", type=int, default=0)
    p.add_argument("--classes", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--out", required=True, help="Output dir for report/json")
    p.add_argument("--suffix", default=".png")
    return p.parse_args()

def fast_hist(pred, gt, nclass):
    mask = (gt != -1)  # we will map ignore to -1
    x = gt[mask] * nclass + pred[mask]
    hist = np.bincount(x, minlength=nclass*nclass).reshape(nclass, nclass)
    return hist

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Map stems
    gts = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(args.gt, f"*{args.suffix}"))}
    pds = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(args.pred, f"*{args.suffix}"))}
    stems = sorted(set(gts) & set(pds))
    if not stems:
        raise SystemExit("No matching filenames between GT and Pred.")

    nclass = max(args.classes + [args.ignore]) + 1  # 6
    hist = np.zeros((nclass, nclass), dtype=np.int64)

    for s in stems:
        gt = cv2.imread(gts[s], cv2.IMREAD_UNCHANGED)
        pd = cv2.imread(pds[s], cv2.IMREAD_UNCHANGED)
        if gt is None or pd is None:
            print(f"[WARN] skip {s}: read error"); continue
        if gt.ndim == 3: gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        if pd.ndim == 3: pd = cv2.cvtColor(pd, cv2.COLOR_BGR2GRAY)
        if gt.shape != pd.shape:
            print(f"[WARN] skip {s}: size mismatch {gt.shape} vs {pd.shape}"); continue

        gt = gt.astype(np.int64)
        pd = pd.astype(np.int64)
        gt_masked = gt.copy()
        gt_masked[gt == args.ignore] = -1  # mark ignore as -1
        pd = np.clip(pd, 0, nclass-1)      # clamp to valid range

        hist += fast_hist(pd, gt_masked, nclass)

    TP = np.diag(hist).astype(np.float64)
    FP = hist.sum(0) - TP
    FN = hist.sum(1) - TP
    eps = 1e-10
    IoU = TP / np.maximum(TP + FP + FN, eps)

    # Per-class IoU for requested classes; mIoU excludes background
    per_class = {int(c): float(IoU[c]) for c in args.classes}
    miou = float(np.mean([IoU[c] for c in args.classes]))

    # Overall accuracy over non-ignored pixels
    valid = hist.sum() - hist[args.ignore,:].sum()  # crude OA excluding gt==ignore rows
    OA = float((TP.sum() - TP[args.ignore]) / max(valid, 1.0))

    report = {
        "classes": args.classes,
        "per_class_IoU": per_class,
        "mIoU_excluding_background": miou,
        "overall_accuracy": OA,
        "confusion_matrix": hist.tolist()
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    # human-readable
    lines = []
    lines.append("Per-class IoU (exclude 0):")
    for c in args.classes:
        lines.append(f"  class {c}: {per_class[c]:.4f}")
    lines.append(f"\nMean IoU (1..5): {miou:.4f}")
    lines.append(f"Overall Accuracy (excl gt==0 rows): {OA:.4f}")
    with open(os.path.join(args.out, "evaluation_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote metrics to: {args.out}")

if __name__ == "__main__":
    main()
