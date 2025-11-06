#!/usr/bin/env python3
"""Diagnostic tool to check mask labels and class distributions."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from collections import Counter
import argparse

# Match classes from dataset
CLASSES = (
    'Background' ,
    'Forest',       # 0
    'Grassland',    # 1
    'Cropland',     # 2
    'Settlement',   # 3
    'SemiNatural'  )

PALETTE = [
    [250, 62, 119],   # Forest
    [168, 232, 84],   # Grassland
    [242, 180, 92],   # Cropland
    [116, 116, 116],  # Settlement
    [255, 214, 33],   # SemiNatural
    [128, 128, 128]   # Background
]

def analyze_mask(mask_path):
    """Analyze a single mask file."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    
    # Get unique labels and their counts
    labels, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    print(f"\nAnalyzing mask: {mask_path.name}")
    print("Class distribution:")
    for label, count in zip(labels, counts):
        if label < len(CLASSES):
            class_name = CLASSES[label]
            percentage = (count / total_pixels) * 100
            print(f"  {class_name:<12} (id={label}): {percentage:6.2f}% ({count} pixels)")
        else:
            print(f"  WARNING: Invalid label {label} found ({count} pixels)")
    
    return mask, labels, counts

def visualize_mask_regions(img_path, mask_path, save_path=None):
    """Create visualization of image, mask, and overlay."""
    # Read image and mask
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    
    # Create color mask
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id, color in enumerate(PALETTE):
        mask_rgb[mask == label_id] = color
    
    # Create overlay
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1-alpha, mask_rgb, alpha, 0)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f"Analysis of {img_path.name}", fontsize=14)
    
    # Original image
    axes[0,0].imshow(img)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    
    # Mask
    axes[0,1].imshow(mask_rgb)
    axes[0,1].set_title("Segmentation Mask")
    axes[0,1].axis('off')
    
    # Overlay
    axes[1,0].imshow(overlay)
    axes[1,0].set_title("Overlay")
    axes[1,0].axis('off')
    
    # Class distribution
    labels, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    percentages = [(count/total_pixels)*100 for count in counts]
    
    class_names = [CLASSES[l] if l < len(CLASSES) else f"Unknown-{l}" for l in labels]
    axes[1,1].bar(class_names, percentages)
    axes[1,1].set_title("Class Distribution")
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_ylabel("Percentage of Image")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze mask labels and class distributions")
    parser.add_argument("--data-root", type=str, default="data/biodiversity",
                       help="Root directory containing Train/Val/Test folders")
    parser.add_argument("--split", type=str, default="Train",
                       choices=["Train", "Val", "Test"],
                       help="Which split to analyze")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of random samples to analyze")
    parser.add_argument("--output-dir", type=str, default="mask_analysis",
                       help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    split_root = data_root / args.split / "Rural"
    img_dir = split_root / "images_png"
    mask_dir = split_root / "masks_png_convert"
    
    if not img_dir.exists() or not mask_dir.exists():
        raise ValueError(f"Cannot find image/mask directories under {split_root}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image/mask pairs
    img_paths = list(img_dir.glob("*.png"))
    if not img_paths:
        raise ValueError(f"No .png images found in {img_dir}")
    
    # Select random samples
    samples = random.sample(img_paths, min(args.samples, len(img_paths)))
    
    # Analyze each sample
    all_labels = []
    for img_path in samples:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f"WARNING: No matching mask for {img_path.name}")
            continue
        
        # Analyze and visualize
        mask, labels, _ = analyze_mask(mask_path)
        all_labels.extend(labels)
        
        save_path = output_dir / f"analysis_{img_path.stem}.png"
        visualize_mask_regions(img_path, mask_path, save_path)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"Analyzed {len(samples)} samples from {args.split} split")
    print("Unique labels found:", sorted(set(all_labels)))
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()