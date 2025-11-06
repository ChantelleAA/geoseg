#!/usr/bin/env python3
"""
Filter out images where more than 50% of the mask is settlement.
This reduces training data size and speeds up training.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

# Class indices (based on your dataset)
# Background=0, Forest=1, Grassland=2, Cropland=3, Settlement=4, SemiNatural=5
SETTLEMENT_CLASS = 4

def calculate_settlement_percentage(mask_path):
    """Calculate percentage of settlement pixels in mask."""
    mask = np.array(Image.open(mask_path))
    total_pixels = mask.size
    settlement_pixels = np.sum(mask == SETTLEMENT_CLASS)
    percentage = (settlement_pixels / total_pixels) * 100
    return percentage

def filter_high_settlement_images(
    images_dir,
    masks_dir,
    backup_dir=None,
    settlement_threshold=50.0,
    dry_run=False
):
    """
    Filter out images with more than threshold% settlement.
    
    Args:
        images_dir: Path to images directory
        masks_dir: Path to masks directory
        backup_dir: Optional backup directory for removed files
        settlement_threshold: Percentage threshold (default: 50.0)
        dry_run: If True, only report what would be removed
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Get all mask files
    mask_files = sorted(list(masks_dir.glob("*.png")))
    
    print(f"🔍 Analyzing {len(mask_files)} masks...")
    print(f"📊 Settlement threshold: {settlement_threshold}%")
    print()
    
    to_remove = []
    settlement_stats = []
    
    # Analyze all masks
    for mask_path in tqdm(mask_files, desc="Analyzing masks"):
        settlement_pct = calculate_settlement_percentage(mask_path)
        settlement_stats.append(settlement_pct)
        
        if settlement_pct > settlement_threshold:
            # Find corresponding image
            mask_name = mask_path.name
            image_path = images_dir / mask_name
            
            if image_path.exists():
                to_remove.append({
                    'image': image_path,
                    'mask': mask_path,
                    'settlement_pct': settlement_pct
                })
    
    # Statistics
    settlement_stats = np.array(settlement_stats)
    print()
    print("=" * 60)
    print("📊 SETTLEMENT STATISTICS")
    print("=" * 60)
    print(f"Total images analyzed: {len(mask_files)}")
    print(f"Mean settlement %: {settlement_stats.mean():.2f}%")
    print(f"Median settlement %: {np.median(settlement_stats):.2f}%")
    print(f"Max settlement %: {settlement_stats.max():.2f}%")
    print(f"Min settlement %: {settlement_stats.min():.2f}%")
    print()
    print(f"Images with >{settlement_threshold}% settlement: {len(to_remove)}")
    print(f"Images to keep: {len(mask_files) - len(to_remove)}")
    print(f"Reduction: {(len(to_remove)/len(mask_files)*100):.1f}%")
    print()
    
    if len(to_remove) == 0:
        print("✅ No images exceed the threshold. Nothing to remove.")
        return
    
    # Show some examples
    print("Examples of images to remove (showing top 10):")
    for i, item in enumerate(sorted(to_remove, key=lambda x: x['settlement_pct'], reverse=True)[:10]):
        print(f"  {i+1}. {item['image'].name} - {item['settlement_pct']:.1f}% settlement")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be removed")
        return
    
    # Create backup if specified
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_images = backup_dir / "images_png"
        backup_masks = backup_dir / "masks_png"
        backup_images.mkdir(parents=True, exist_ok=True)
        backup_masks.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Creating backup in {backup_dir}...")
        for item in tqdm(to_remove, desc="Backing up"):
            shutil.copy2(item['image'], backup_images / item['image'].name)
            shutil.copy2(item['mask'], backup_masks / item['mask'].name)
        print(f"✅ Backed up {len(to_remove)} pairs")
        print()
    
    # Remove files
    print("🗑️  Removing high-settlement images...")
    for item in tqdm(to_remove, desc="Removing files"):
        item['image'].unlink()
        item['mask'].unlink()
    
    print()
    print("=" * 60)
    print("✅ FILTERING COMPLETE")
    print("=" * 60)
    print(f"Removed: {len(to_remove)} image/mask pairs")
    print(f"Remaining: {len(mask_files) - len(to_remove)} image/mask pairs")
    if backup_dir:
        print(f"Backup location: {backup_dir}")
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter out high-settlement images")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/biodiversity/Train/Rural/images_png",
        help="Path to images directory"
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="data/biodiversity/Train/Rural/masks_png",
        help="Path to masks directory"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="data/biodiversity_high_settlement_backup",
        help="Backup directory for removed files (set to empty string to skip backup)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Settlement percentage threshold (default: 50.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze and report, don't remove files"
    )
    
    args = parser.parse_args()
    
    backup_dir = args.backup_dir if args.backup_dir else None
    
    filter_high_settlement_images(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        backup_dir=backup_dir,
        settlement_threshold=args.threshold,
        dry_run=args.dry_run
    )
