"""Analyze class distribution in the training and validation datasets."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tools.cfg import py2cfg
from tqdm import tqdm
import torch

# Class names
CLASSES = ['Background', 'Forest', 'Grassland', 'Cropland', 'Settlement', 'SemiNatural']


def analyze_dataset(dataset, name='Dataset'):
    """Analyze class distribution in a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print(f"{'='*60}")
    
    # Count pixels per class
    class_counts = np.zeros(len(CLASSES), dtype=np.int64)
    total_pixels = 0
    
    print(f"Processing {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        mask = sample['gt_semantic_seg'].numpy()
        
        # Count each class
        for class_id in range(len(CLASSES)):
            class_counts[class_id] += np.sum(mask == class_id)
        
        total_pixels += mask.size
    
    # Calculate percentages
    class_percentages = (class_counts / total_pixels) * 100
    
    # Print results
    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"\nClass Distribution:")
    print(f"{'Class':<20} {'Pixels':>15} {'Percentage':>12}")
    print("-" * 50)
    for class_name, count, pct in zip(CLASSES, class_counts, class_percentages):
        print(f"{class_name:<20} {count:>15,} {pct:>11.2f}%")
    
    # Calculate class weights (inverse frequency)
    # Avoid division by zero
    class_weights = np.zeros_like(class_percentages)
    for i in range(len(class_percentages)):
        if class_percentages[i] > 0:
            class_weights[i] = 100.0 / class_percentages[i]
        else:
            class_weights[i] = 0.0
    
    # Normalize weights so minimum weight is 1.0
    if class_weights.max() > 0:
        class_weights = class_weights / class_weights[class_weights > 0].min()
    
    print(f"\nRecommended Class Weights (inverse frequency):")
    print(f"{'Class':<20} {'Weight':>10}")
    print("-" * 32)
    for class_name, weight in zip(CLASSES, class_weights):
        print(f"{class_name:<20} {weight:>10.2f}")
    
    return class_counts, class_percentages, class_weights


def plot_distribution(train_pct, val_pct, output_path='class_distribution.png'):
    """Plot class distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(CLASSES))
    width = 0.35
    
    # Plot percentages
    bars1 = ax1.bar(x - width/2, train_pct, width, label='Train', alpha=0.8)
    bars2 = ax1.bar(x + width/2, val_pct, width, label='Val', alpha=0.8)
    
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution: Train vs Val', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot log scale to see small classes better
    ax2.bar(x - width/2, train_pct, width, label='Train', alpha=0.8)
    ax2.bar(x + width/2, val_pct, width, label='Val', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%) - Log Scale', fontsize=12, fontweight='bold')
    ax2.set_title('Class Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    # Load config
    config_path = 'config/biodiversity/ftunetformer.py'
    print(f"Loading config from: {config_path}")
    config = py2cfg(config_path)
    
    # Analyze train dataset
    train_counts, train_pct, train_weights = analyze_dataset(
        config.train_dataset, 
        name='Training Dataset'
    )
    
    # Analyze val dataset
    val_counts, val_pct, val_weights = analyze_dataset(
        config.val_dataset,
        name='Validation Dataset'
    )
    
    # Plot comparison
    plot_distribution(train_pct, val_pct)
    
    # Save weights to file
    weights_file = 'recommended_class_weights.txt'
    with open(weights_file, 'w') as f:
        f.write("Recommended Class Weights (based on training set):\n")
        f.write("="*50 + "\n\n")
        f.write("For PyTorch CrossEntropyLoss:\n")
        f.write("weight = torch.FloatTensor([")
        f.write(", ".join([f"{w:.4f}" for w in train_weights]))
        f.write("])\n\n")
        
        f.write("For config file:\n")
        f.write("class_weights = [")
        f.write(", ".join([f"{w:.4f}" for w in train_weights]))
        f.write("]\n\n")
        
        f.write("Class mapping:\n")
        for i, (cls, w) in enumerate(zip(CLASSES, train_weights)):
            f.write(f"{i}: {cls:<20} weight={w:.4f}\n")
    
    print(f"\nWeights saved to: {weights_file}")
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
