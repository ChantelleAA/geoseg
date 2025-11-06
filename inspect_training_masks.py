import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Color mapping for visualization (matching biodiversity classes)
COLOR_MAP = {
    0: [0, 0, 0],           # Background - Black
    1: [250, 62, 119],      # Forest - Pink
    2: [168, 232, 84],      # Grassland - Light Green
    3: [242, 180, 92],      # Cropland - Orange
    4: [59, 141, 247],      # Settlement - Blue
    5: [255, 214, 33],      # SemiNatural - Yellow
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Forest',
    2: 'Grassland',
    3: 'Cropland',
    4: 'Settlement',
    5: 'SemiNatural'
}

def mask_to_rgb(mask):
    """Convert single-channel mask to RGB for visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    return rgb

def get_class_distribution(mask):
    """Get pixel count for each class."""
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    dist = {}
    for class_id, count in zip(unique, counts):
        if class_id in CLASS_NAMES:
            dist[CLASS_NAMES[class_id]] = (count, count/total * 100)
    return dist

def visualize_samples(data_dir, num_samples=6, source_filter=None):
    """
    Visualize random training samples with their masks.
    
    Args:
        data_dir: Path to combined training data
        num_samples: Number of samples to display
        source_filter: 'oem', 'target', or None for both
    """
    img_dir = Path(data_dir) / 'Train' / 'images_png'
    mask_dir = Path(data_dir) / 'Train' / 'masks_png'
    
    # Get all image files
    all_images = sorted(list(img_dir.glob('*.png')))
    
    # Filter by source if requested
    if source_filter:
        all_images = [img for img in all_images if img.name.startswith(source_filter)]
    
    if len(all_images) == 0:
        print(f"No images found in {img_dir}")
        return
    
    # Sample random images
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(sample_images):
        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_path = mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to RGB for visualization
        mask_rgb = mask_to_rgb(mask)
        
        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)
        
        # Get class distribution
        dist = get_class_distribution(mask)
        
        # Get source
        source = 'OpenEarthMap' if img_path.name.startswith('oem_') else 'Biodiversity Target'
        
        # Plot
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Image: {img_path.name[:30]}...\nSource: {source}', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask_rgb)
        axes[idx, 1].set_title('Ground Truth Mask', fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlay)
        # Build distribution text
        dist_text = '\n'.join([f'{name}: {pct:.1f}%' for name, (cnt, pct) in dist.items()])
        axes[idx, 2].set_title(f'Overlay\n{dist_text}', fontsize=8)
        axes[idx, 2].axis('off')
        
        # Print unique mask values to verify class range
        unique_vals = np.unique(mask)
        print(f"{img_path.name}: Mask values: {unique_vals}, Shape: {mask.shape}")
    
    plt.tight_layout()
    plt.savefig('training_mask_inspection.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: training_mask_inspection.png")
    plt.show()
    
    # Print legend
    print("\n=== Class Color Legend ===")
    for class_id, color in COLOR_MAP.items():
        print(f"{CLASS_NAMES[class_id]:15s}: RGB{color}")

def inspect_specific_samples(data_dir, sample_names):
    """Inspect specific samples by name."""
    img_dir = Path(data_dir) / 'Train' / 'images_png'
    mask_dir = Path(data_dir) / 'Train' / 'masks_png'
    
    found_samples = []
    for name in sample_names:
        img_path = img_dir / name
        if img_path.exists():
            found_samples.append(img_path)
        else:
            print(f"Warning: {name} not found")
    
    if found_samples:
        print(f"\nInspecting {len(found_samples)} specific samples...")
        visualize_samples_from_paths(found_samples, mask_dir)

def visualize_samples_from_paths(img_paths, mask_dir):
    """Visualize specific image paths."""
    num_samples = len(img_paths)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_path = mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        mask_rgb = mask_to_rgb(mask)
        overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)
        dist = get_class_distribution(mask)
        source = 'OpenEarthMap' if img_path.name.startswith('oem_') else 'Biodiversity Target'
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'{img_path.name}\nSource: {source}', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask_rgb)
        axes[idx, 1].set_title('Ground Truth Mask', fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlay)
        dist_text = '\n'.join([f'{name}: {pct:.1f}%' for name, (cnt, pct) in dist.items()])
        axes[idx, 2].set_title(f'Overlay\n{dist_text}', fontsize=8)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('specific_training_masks.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to: specific_training_masks.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect training masks')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/chantelle/Desktop/UCD/ai_sandbox/OpenEarthMap_wo_xBD/biodiversity_combined',
                       help='Path to combined dataset')
    parser.add_argument('--num-samples', type=int, default=6,
                       help='Number of random samples to visualize')
    parser.add_argument('--source', type=str, choices=['oem', 'target', 'both'], default='both',
                       help='Filter by data source')
    parser.add_argument('--specific', type=str, nargs='+',
                       help='Specific sample names to inspect')
    
    args = parser.parse_args()
    
    source_filter = None if args.source == 'both' else args.source
    
    print(f"Inspecting training data from: {args.data_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Source filter: {args.source}\n")
    
    if args.specific:
        inspect_specific_samples(args.data_dir, args.specific)
    else:
        visualize_samples(args.data_dir, args.num_samples, source_filter)
