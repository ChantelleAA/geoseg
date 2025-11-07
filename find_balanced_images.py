"""Find balanced images from biodiversity_combined dataset to add to training."""

import numpy as np
from pathlib import Path
from PIL import Image
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

def analyze_mask_balance(mask_path):
    """Analyze a mask for class balance."""
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
    
    # Calculate balance metrics
    num_classes_present = sum(1 for pct in class_percentages.values() if pct > 1.0)  # Classes with >1%
    
    # Settlement and SemiNatural presence
    settlement_pct = class_percentages[4]
    seminatural_pct = class_percentages[5]
    
    # Diversity score: entropy-like measure
    non_zero_pcts = [pct for pct in class_percentages.values() if pct > 0]
    if len(non_zero_pcts) > 0:
        # Normalize to sum to 1
        probs = np.array(non_zero_pcts) / 100.0
        # Calculate entropy
        diversity_score = -np.sum(probs * np.log(probs + 1e-10))
    else:
        diversity_score = 0.0
    
    # Balance score: prefer images with minority classes AND diversity
    # Prioritize Settlement heavily, reduce SemiNatural influence
    balance_score = (
        settlement_pct * 5.0 +  # Weight settlement very heavily (increased from 2.0)
        seminatural_pct * 0.5 +  # Reduce seminatural weight (decreased from 1.5)
        diversity_score * 10.0 +  # Reward diversity
        num_classes_present * 5.0  # Reward having multiple classes
    )
    
    return {
        'class_percentages': class_percentages,
        'num_classes': num_classes_present,
        'settlement_pct': settlement_pct,
        'seminatural_pct': seminatural_pct,
        'diversity_score': diversity_score,
        'balance_score': balance_score
    }

def find_balanced_images(data_root, n_images=50):
    """Find most balanced images from combined dataset."""
    masks_dir = Path(data_root) / 'masks'
    
    if not masks_dir.exists():
        print(f"ERROR: {masks_dir} does not exist!")
        return []
    
    results = []
    
    for mask_path in masks_dir.glob('*.png'):
        img_id = mask_path.stem
        metrics = analyze_mask_balance(mask_path)
        
        # Filter criteria: balanced images with minority classes
        # 1. Must have at least 3 classes (avoid single-class dominated)
        # 2. No class should dominate >80% (avoid extreme examples)
        # 3. Prioritize Settlement: must have Settlement >2%, SemiNatural optional
        
        class_pcts = list(metrics['class_percentages'].values())
        max_class_pct = max(class_pcts)
        
        has_settlement = metrics['settlement_pct'] > 2.0
        has_minority = (metrics['settlement_pct'] > 2.0 or metrics['seminatural_pct'] > 5.0)  # Higher threshold for seminatural
        is_diverse = metrics['num_classes'] >= 3
        not_dominated = max_class_pct < 80.0
        
        if has_minority and is_diverse and not_dominated and has_settlement:  # Must have settlement
            results.append({
                'img_id': img_id,
                'mask_path': str(mask_path),
                **metrics
            })
    
    # Sort by balance score (descending)
    results.sort(key=lambda x: x['balance_score'], reverse=True)
    
    print(f"Total images analyzed: {len(results)}")
    print(f"\nTop {n_images} most balanced images:")
    print("=" * 120)
    print(f"{'Image ID':<30} {'Classes':>8} {'Settlement %':>13} {'SemiNat %':>11} {'Diversity':>10} {'Balance':>10}")
    print("=" * 120)
    
    for item in results[:n_images]:
        print(f"{item['img_id']:<30} {item['num_classes']:>8} "
              f"{item['settlement_pct']:>12.2f}% {item['seminatural_pct']:>10.2f}% "
              f"{item['diversity_score']:>10.2f} {item['balance_score']:>10.1f}")
    
    return results[:n_images]

def save_selected_images(selected_images, output_file):
    """Save list of selected balanced images."""
    image_list = {
        'selected_count': len(selected_images),
        'images': [item['img_id'] for item in selected_images],
        'metrics': [
            {
                'img_id': item['img_id'],
                'settlement_pct': item['settlement_pct'],
                'seminatural_pct': item['seminatural_pct'],
                'num_classes': item['num_classes'],
                'diversity_score': item['diversity_score'],
                'balance_score': item['balance_score']
            }
            for item in selected_images
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(image_list, f, indent=2)
    
    print(f"\n✓ Saved selected images list to: {output_file}")

if __name__ == '__main__':
    # Find balanced images from combined dataset
    selected = find_balanced_images(
        'data/Biodiversity_tiff/Train',
        n_images=50  # Select top 50 most balanced images
    )
    
    if selected:
        save_selected_images(selected, 'combined_balanced_images.json')
