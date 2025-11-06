#!/usr/bin/env python3
"""
Create combined dataset by symlinking target (biodiversity) + new (OpenEarthMap) data.
This keeps original data intact while creating a unified dataset for pretraining.
"""
import os
import glob
from pathlib import Path
from tqdm import tqdm

# Paths
TARGET_TRAIN = "/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/data/biodiversity/Train/Rural"
TARGET_VAL = "/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/data/biodiversity/Val"
OEM_CONVERTED = "/home/chantelle/Desktop/UCD/ai_sandbox/data/biodiversity/Train/Rural"  # Where prep script output OEM
OEM_BASE = "/home/chantelle/Desktop/UCD/ai_sandbox/OpenEarthMap_wo_xBD"  # For train.txt/val.txt splits
OUTPUT_BASE = "/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/data/biodiversity_combined"

def read_file_list(txt_path):
    """Read list of file stems from train.txt or val.txt"""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_symlinks(src_dir, dst_dir, file_list, prefix=""):
    """Create symlinks for images and masks"""
    os.makedirs(os.path.join(dst_dir, "images_png"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "masks_png"), exist_ok=True)
    
    for stem in tqdm(file_list, desc=f"Linking {prefix}"):
        # Link image
        src_img = os.path.join(src_dir, "images_png", f"{stem}.png")
        dst_img = os.path.join(dst_dir, "images_png", f"{prefix}{stem}.png")
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        
        # Link mask
        src_mask = os.path.join(src_dir, "masks_png", f"{stem}.png")
        dst_mask = os.path.join(dst_dir, "masks_png", f"{prefix}{stem}.png")
        if os.path.exists(src_mask) and not os.path.exists(dst_mask):
            os.symlink(src_mask, dst_mask)

def main():
    print("="*80)
    print("CREATING COMBINED DATASET (Target + OpenEarthMap)")
    print("="*80)
    
    # Get OpenEarthMap converted files directly
    oem_imgs = glob.glob(os.path.join(OEM_CONVERTED, "images_png", "*.png"))
    oem_all = [Path(p).stem for p in oem_imgs]
    
    # Split OEM: 80% train, 20% val
    import random
    random.seed(42)
    random.shuffle(oem_all)
    split_idx = int(len(oem_all) * 0.8)
    oem_train = oem_all[:split_idx]
    oem_val = oem_all[split_idx:]
    
    print(f"\nOpenEarthMap (converted):")
    print(f"  Train: {len(oem_train)} samples")
    print(f"  Val: {len(oem_val)} samples")
    
    # Count target samples
    target_train_imgs = glob.glob(os.path.join(TARGET_TRAIN, "images_png", "*.png"))
    target_val_imgs = glob.glob(os.path.join(TARGET_VAL, "images_png", "*.png"))
    target_train = [Path(p).stem for p in target_train_imgs]
    target_val = [Path(p).stem for p in target_val_imgs]
    
    print(f"\nTarget (Biodiversity):")
    print(f"  Train: {len(target_train)} samples")
    print(f"  Val: {len(target_val)} samples")
    
    print(f"\nCombined:")
    print(f"  Train: {len(oem_train) + len(target_train)} samples")
    print(f"  Val: {len(oem_val) + len(target_val)} samples")
    
    # Create train symlinks
    print("\n" + "="*80)
    print("CREATING TRAINING SYMLINKS")
    print("="*80)
    
    # OEM train data (prefix with "oem_" to avoid name conflicts)
    create_symlinks(OEM_CONVERTED, os.path.join(OUTPUT_BASE, "Train"), oem_train, prefix="oem_")
    
    # Target train data
    create_symlinks(TARGET_TRAIN, os.path.join(OUTPUT_BASE, "Train"), target_train, prefix="target_")
    
    # Create val symlinks
    print("\n" + "="*80)
    print("CREATING VALIDATION SYMLINKS")
    print("="*80)
    
    # OEM val data
    create_symlinks(OEM_CONVERTED, os.path.join(OUTPUT_BASE, "Val"), oem_val, prefix="oem_")
    
    # Target val data
    create_symlinks(TARGET_VAL, os.path.join(OUTPUT_BASE, "Val"), target_val, prefix="target_")
    
    print("\n" + "="*80)
    print("COMBINED DATASET CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nLocation: {OUTPUT_BASE}")
    print(f"  Train/images_png: {len(oem_train) + len(target_train)} images")
    print(f"  Train/masks_png: {len(oem_train) + len(target_train)} masks")
    print(f"  Val/images_png: {len(oem_val) + len(target_val)} images")
    print(f"  Val/masks_png: {len(oem_val) + len(target_val)} masks")
    print("\nUsing symlinks - original data is unchanged.")

if __name__ == "__main__":
    main()
