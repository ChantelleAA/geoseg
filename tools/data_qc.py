#!/usr/bin/env python3
import argparse, os, glob, sys
import numpy as np
import cv2
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser("Data QC for index masks")
    p.add_argument("--images", required=True, help="Dir with images (*.tif or *.png)")
    p.add_argument("--masks",  required=True, help="Dir with masks (*.png, uint8 indices)")
    p.add_argument("--img-suffix", default=".tif")
    p.add_argument("--mask-suffix", default=".png")
    return p.parse_args()

def main():
    args = parse_args()
    imgs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(args.images, f"*{args.img_suffix}"))}
    msks = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(args.masks, f"*{args.mask_suffix}"))}

    common = sorted(set(imgs) & set(msks))
    miss_img = sorted(set(msks) - set(imgs))
    miss_msk = sorted(set(imgs) - set(msks))
    if miss_img: print(f"[WARN] masks without images: {len(miss_img)} (showing 5): {miss_img[:5]}")
    if miss_msk: print(f"[WARN] images without masks: {len(miss_msk)} (showing 5): {miss_msk[:5]}")

    ok = True
    counts = Counter()
    for stem in common:
        ip, mp = imgs[stem], msks[stem]
        im = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        mm = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if im is None or mm is None:
            print(f"[ERR] failed to read {stem}"); ok = False; continue

        # enforce 2D mask, uint8
        if mm.ndim == 3: mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
        if mm.dtype != np.uint8:
            print(f"[ERR] mask not uint8: {mp} dtype={mm.dtype}"); ok = False

        # size match
        if im.shape[0] != mm.shape[0] or im.shape[1] != mm.shape[1]:
            print(f"[ERR] size mismatch for {stem}: img {im.shape[:2]} vs mask {mm.shape[:2]}")
            ok = False

        vals, freqs = np.unique(mm, return_counts=True)
        bad = set(vals.tolist()) - {0,1,2,3,4,5}
        if bad:
            print(f"[ERR] out-of-range labels {sorted(list(bad))} in {mp}")
            ok = False
        counts.update(dict(zip(vals.tolist(), freqs.tolist())))

    tot = sum(counts.values()) or 1
    print("\nLabel distribution (across matched pairs):")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}  ({counts[k]/tot:.2%})")

    if not ok:
        print("\n[QC] FAIL. Please fix errors above.")
        sys.exit(1)
    print("\n[QC] OK. Masks look consistent with the contract.")

if __name__ == "__main__":
    main()
