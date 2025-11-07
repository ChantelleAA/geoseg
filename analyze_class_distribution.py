"""Analyze class distribution in training images to identify which ones to augment."""

import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import json

# Class mapping
CLASSES = {
    0: 'Background',
    1: 'Forest Land',
    2: 'Grassland', 
    3: 'Cropland',
    4: 'Settlement',
    5: 'SemiNatural Grassland'
}

# Target classes that need augmentation
TARGET_CLASSES = [4, 5]  # Settlement, SemiNatural

def analyze_mask(mask_path):
    """Analyze a single mask and return class pixel counts."""
    mask = np.array(Image.open(mask_path))
    unique, counts = np.unique(mask, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_pixels = mask.size
    
    # Calculate percentages
    class_percentages = {}
    for cls_id in range(6):
        count = class_counts.get(cls_id, 0)
        percentage = (count / total_pixels) * 100
        class_percentages[cls_id] = percentage
    
    return class_percentages, class_counts

def analyze_dataset(data_root):
    """Analyze all masks in a dataset."""
    mask_dir = Path(data_root) / 'Rural' / 'masks_png'
    
    results = []
    
    for mask_path in sorted(mask_dir.glob('*.png')):
        img_id = mask_path.stem
        percentages, counts = analyze_mask(mask_path)
        
        # Calculate score for target classes
        target_score = sum(percentages[cls] for cls in TARGET_CLASSES)
        
        results.append({
            'img_id': img_id,
            'mask_path': str(mask_path),
            'class_percentages': percentages,
            'class_counts': counts,
            'target_score': target_score,
            'settlement_pct': percentages[4],
            'seminatural_pct': percentages[5]
        })
    
    # Sort by target score (descending)
    results.sort(key=lambda x: x['target_score'], reverse=True)
    
    return results

def print_top_images(results, n=20):
    """Print top N images with most target class presence."""
    print(f"\nTop {n} images with highest Settlement + SemiNatural presence:")
    print("=" * 100)
    print(f"{'Image ID':<30} {'Settlement %':>12} {'SemiNatural %':>15} {'Total %':>10}")
    print("=" * 100)
    
    for item in results[:n]:
        print(f"{item['img_id']:<30} {item['settlement_pct']:>11.2f}% {item['seminatural_pct']:>14.2f}% {item['target_score']:>9.2f}%")

def save_augmentation_list(results, output_file, threshold_settlement=5.0, threshold_seminatural=5.0):
    """Save list of images to augment (replicate)."""
    
    # Images with significant Settlement (>5%)
    settlement_images = [
        item['img_id'] for item in results 
        if item['settlement_pct'] >= threshold_settlement
    ]
    
    # Images with significant SemiNatural (>5%)
    seminatural_images = [
        item['img_id'] for item in results 
        if item['seminatural_pct'] >= threshold_seminatural
    ]
    
    augmentation_list = {
        'settlement_images': settlement_images,
        'seminatural_images': seminatural_images,
        'settlement_count': len(settlement_images),
        'seminatural_count': len(seminatural_images),
        'thresholds': {
            'settlement': threshold_settlement,
            'seminatural': threshold_seminatural
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(augmentation_list, f, indent=2)
    
    print(f"\n\nSaved augmentation list to: {output_file}")
    print(f"  - Settlement images (>{threshold_settlement}%): {len(settlement_images)}")
    print(f"  - SemiNatural images (>{threshold_seminatural}%): {len(seminatural_images)}")
    
    return augmentation_list

if __name__ == '__main__':
    # Analyze training data
    print("Analyzing Training Set...")
    train_results = analyze_dataset('data/biodiversity/Train')
    print(f"Total training images: {len(train_results)}")
    
    # Print statistics
    total_settlement = sum(r['settlement_pct'] for r in train_results) / len(train_results)
    total_seminatural = sum(r['seminatural_pct'] for r in train_results) / len(train_results)
    print(f"Average Settlement percentage: {total_settlement:.2f}%")
    print(f"Average SemiNatural percentage: {total_seminatural:.2f}%")
    
    # Print top images
    print_top_images(train_results, n=30)
    
    # Save augmentation list
    augmentation_list = save_augmentation_list(
        train_results,
        'train_augmentation_list.json',
        threshold_settlement=5.0,
        threshold_seminatural=5.0
    )
