"""Identify hard samples for oversampling strategy."""

import torch
import numpy as np
from pathlib import Path
from tools.cfg import py2cfg
from train_kd import KD_Train
from tqdm import tqdm
import json

def compute_sample_iou(pred, target, num_classes=6):
    """Compute per-sample IoU for each class."""
    ious = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            iou = np.nan  # Class not present
        else:
            iou = intersection / union
        ious[cls] = iou
    
    # Overall sample IoU (mean of present classes)
    present_ious = [v for v in ious.values() if not np.isnan(v)]
    mean_iou = np.mean(present_ious) if present_ious else 0.0
    
    return ious, mean_iou


def analyze_dataset(config_path, checkpoint_path):
    """Analyze which samples are hardest for the model."""
    
    print("Loading config and model...")
    config = py2cfg(config_path)
    model = KD_Train.load_from_checkpoint(checkpoint_path, config=config, map_location='cuda')
    model.eval()
    model.cuda()
    
    # Analyze training dataset
    dataset = config.train_dataset
    print(f"Analyzing {len(dataset)} training samples...")
    
    results = []
    class_names = ['Background', 'Forest', 'Grassland', 'Cropland', 'Settlement', 'SemiNatural']
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing samples"):
            sample = dataset[idx]
            img = sample['img'].unsqueeze(0).cuda()
            mask = sample['gt_semantic_seg'].numpy()
            img_id = sample['img_id']
            
            # Get prediction
            pred = model(img)
            pred = torch.softmax(pred, dim=1).argmax(dim=1)[0].cpu().numpy()
            
            # Compute per-class IoU and overall IoU
            class_ious, mean_iou = compute_sample_iou(pred, mask, num_classes=6)
            
            # Count pixels per class in ground truth
            class_counts = {}
            for cls in range(6):
                class_counts[cls] = (mask == cls).sum()
            
            results.append({
                'idx': idx,
                'img_id': img_id,
                'mean_iou': float(mean_iou),
                'class_ious': {class_names[k]: float(v) if not np.isnan(v) else None 
                               for k, v in class_ious.items()},
                'class_pixel_counts': {class_names[k]: int(v) for k, v in class_counts.items()}
            })
    
    # Save full results
    output_file = Path('hard_samples_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")
    
    # Analyze and print summary
    print("\n" + "="*80)
    print("HARD SAMPLE ANALYSIS")
    print("="*80)
    
    # Sort by mean IoU (worst first)
    results_sorted = sorted(results, key=lambda x: x['mean_iou'])
    
    print(f"\nTop 20 HARDEST samples (lowest overall IoU):")
    print("-" * 80)
    for i, r in enumerate(results_sorted[:20], 1):
        print(f"{i:2d}. {r['img_id']:30s} | mIoU: {r['mean_iou']:.3f}")
        poor_classes = [(cls, iou) for cls, iou in r['class_ious'].items() 
                       if iou is not None and iou < 0.7]
        if poor_classes:
            print(f"    Poor classes: {', '.join([f'{c}: {v:.3f}' for c, v in poor_classes])}")
    
    # Analyze per-class performance
    print(f"\n\nPer-class hard samples (samples with IoU < 0.5 for each class):")
    print("-" * 80)
    
    for cls_idx, cls_name in enumerate(class_names):
        # Get samples where this class has low IoU and is actually present
        hard_samples = [
            r for r in results 
            if r['class_ious'][cls_name] is not None 
            and r['class_ious'][cls_name] < 0.5
            and r['class_pixel_counts'][cls_name] > 100  # Class is significantly present
        ]
        
        hard_samples_sorted = sorted(hard_samples, key=lambda x: x['class_ious'][cls_name])
        
        print(f"\n{cls_name}: {len(hard_samples)} hard samples")
        if hard_samples_sorted:
            print(f"  Worst 10:")
            for i, r in enumerate(hard_samples_sorted[:10], 1):
                iou = r['class_ious'][cls_name]
                pixels = r['class_pixel_counts'][cls_name]
                print(f"    {i:2d}. {r['img_id']:30s} | IoU: {iou:.3f} | {pixels:6d} pixels")
    
    # Generate oversampling weights
    print(f"\n\nGENERATING OVERSAMPLING WEIGHTS:")
    print("-" * 80)
    
    weights = []
    for r in results:
        # Base weight = 1.0
        weight = 1.0
        
        # Increase weight for samples with low overall IoU
        if r['mean_iou'] < 0.6:
            weight += 2.0  # 3x total
        elif r['mean_iou'] < 0.7:
            weight += 1.0  # 2x total
        
        # Extra weight for weak classes (Forest, Settlement)
        forest_iou = r['class_ious']['Forest']
        settlement_iou = r['class_ious']['Settlement']
        
        if forest_iou is not None and forest_iou < 0.6 and r['class_pixel_counts']['Forest'] > 100:
            weight += 1.5
        
        if settlement_iou is not None and settlement_iou < 0.6 and r['class_pixel_counts']['Settlement'] > 100:
            weight += 1.5
        
        weights.append(weight)
    
    # Save weights
    weights_file = Path('sample_weights.txt')
    with open(weights_file, 'w') as f:
        for idx, w in enumerate(weights):
            f.write(f"{idx}\t{w}\n")
    print(f"Saved {len(weights)} sample weights to: {weights_file}")
    
    # Statistics
    print(f"\nWeight statistics:")
    print(f"  Mean weight: {np.mean(weights):.2f}")
    print(f"  Max weight: {np.max(weights):.2f}")
    print(f"  Samples with weight > 1: {sum(1 for w in weights if w > 1)} ({100*sum(1 for w in weights if w > 1)/len(weights):.1f}%)")
    print(f"  Samples with weight > 2: {sum(1 for w in weights if w > 2)} ({100*sum(1 for w in weights if w > 2)/len(weights):.1f}%)")
    print(f"  Samples with weight > 3: {sum(1 for w in weights if w > 3)} ({100*sum(1 for w in weights if w > 3)/len(weights):.1f}%)")
    
    return results, weights


if __name__ == '__main__':
    config_path = 'config/biodiversity/ftunetformer_kd.py'
    checkpoint_path = 'model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/ftunetformer-kd-512-crop-ms-e45.ckpt'
    
    results, weights = analyze_dataset(config_path, checkpoint_path)
    
    print("\n" + "="*80)
    print("DONE! Use 'sample_weights.txt' for weighted sampling in training.")
    print("="*80)
