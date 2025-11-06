"""Compare baseline FTUNetFormer vs KD FTUNetFormer predictions visually."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import random
from torch.utils.data import DataLoader
from tools.cfg import py2cfg
import argparse

# Import from train_kd to handle teacher model properly
from train_kd import KD_Train


def label2rgb(mask):
    """Convert label mask to RGB for visualization."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [11, 246, 210]   # Background
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [250, 62, 119]   # Forest
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [168, 232, 84]   # Grassland
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [242, 180, 92]   # Cropland
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [116, 116, 116]  # Settlement
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 214, 33]   # SemiNatural
    return mask_rgb


def load_model(config_path, checkpoint_path, is_kd=False):
    """Load a trained model from checkpoint."""
    config = py2cfg(config_path)
    
    if is_kd:
        # For KD model, use KD_Train wrapper
        model = KD_Train.load_from_checkpoint(checkpoint_path, config=config, map_location='cuda')
    else:
        # For baseline, load directly
        from train_supervision import Supervision_Train
        model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, map_location='cuda')
    
    model.eval()
    model.cuda()
    return model, config


def predict(model, image):
    """Get prediction from model."""
    with torch.no_grad():
        image = image.cuda()
        pred = model(image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.argmax(dim=1)
    return pred.cpu().numpy()


def compare_models(baseline_config, baseline_ckpt, kd_config, kd_ckpt, output_dir, num_samples=5, use_train_data=False):
    """Compare predictions from baseline and KD models."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading baseline model...")
    baseline_model, baseline_cfg = load_model(baseline_config, baseline_ckpt, is_kd=False)
    
    print("Loading KD model...")
    kd_model, kd_cfg = load_model(kd_config, kd_ckpt, is_kd=True)
    
    # Use validation or training dataset from baseline config (has ground truth masks)
    if use_train_data:
        print("Loading training dataset...")
        test_dataset = baseline_cfg.train_dataset
    else:
        print("Loading validation dataset...")
        test_dataset = baseline_cfg.val_dataset
    
    # Randomly select samples
    total_samples = len(test_dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"Generating comparisons for {len(indices)} samples...")
    
    for idx, sample_idx in enumerate(indices):
        sample = test_dataset[sample_idx]
        image = sample['img'].unsqueeze(0)  # Add batch dimension
        mask = sample['gt_semantic_seg'].numpy()
        img_id = sample['img_id']
        
        # Get predictions
        baseline_pred = predict(baseline_model, image)[0]
        kd_pred = predict(kd_model, image)[0]
        
        # Convert to RGB
        img_rgb = image[0].permute(1, 2, 0).cpu().numpy()
        # Denormalize image (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = (img_rgb * std + mean) * 255
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        gt_rgb = label2rgb(mask)
        baseline_rgb = label2rgb(baseline_pred)
        kd_rgb = label2rgb(kd_pred)
        
        # Create overlays (blend image with mask at 40% opacity)
        alpha = 0.4
        gt_overlay = cv2.addWeighted(img_rgb, 1-alpha, gt_rgb, alpha, 0)
        baseline_overlay = cv2.addWeighted(img_rgb, 1-alpha, baseline_rgb, alpha, 0)
        kd_overlay = cv2.addWeighted(img_rgb, 1-alpha, kd_rgb, alpha, 0)
        
        # Create comparison plot with 2 rows: masks on top, overlays on bottom
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Top row: original masks
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_rgb)
        axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(baseline_rgb)
        axes[0, 2].set_title('Baseline Prediction', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(kd_rgb)
        axes[0, 3].set_title('KD Prediction', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # Bottom row: overlays
        axes[1, 0].imshow(img_rgb)
        axes[1, 0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gt_overlay)
        axes[1, 1].set_title('GT Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(baseline_overlay)
        axes[1, 2].set_title('Baseline Overlay', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(kd_overlay)
        axes[1, 3].set_title('KD Overlay', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'Sample: {img_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / f'comparison_{idx+1}_{img_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {idx+1}/{len(indices)}: {save_path.name}")
    
    print(f"\nAll comparisons saved to: {output_dir}")
    
    # Create legend
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    classes = ['Background', 'Forest', 'Grassland', 'Cropland', 'Settlement', 'SemiNatural']
    colors = [
        [11/255, 246/255, 210/255],
        [250/255, 62/255, 119/255],
        [168/255, 232/255, 84/255],
        [242/255, 180/255, 92/255],
        [116/255, 116/255, 116/255],
        [255/255, 214/255, 33/255]
    ]
    
    for i, (cls, color) in enumerate(zip(classes, colors)):
        ax.add_patch(plt.Rectangle((i*1.5, 0), 0.4, 0.4, color=color))
        ax.text(i*1.5 + 0.5, 0.2, cls, fontsize=12, va='center')
    
    ax.set_xlim(-0.5, len(classes)*1.5)
    ax.set_ylim(-0.5, 1)
    plt.title('Class Legend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'legend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to: {output_dir / 'legend.png'}")


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs KD model predictions')
    parser.add_argument('--baseline_config', type=str, 
                        default='config/biodiversity/ftunetformer.py',
                        help='Path to baseline config')
    parser.add_argument('--baseline_ckpt', type=str,
                        default='model_weights/biodiversity/ftunetformer-512-crop-ms-e45/ftunetformer-512-crop-ms-e45.ckpt',
                        help='Path to baseline checkpoint')
    parser.add_argument('--kd_config', type=str,
                        default='config/biodiversity/ftunetformer_kd.py',
                        help='Path to KD config')
    parser.add_argument('--kd_ckpt', type=str,
                        default='model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/ftunetformer-kd-512-crop-ms-e45.ckpt',
                        help='Path to KD checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='predictions',
                        help='Directory to save comparison images')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to compare')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_train_data', action='store_true',
                        help='Use training data instead of validation data')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    compare_models(
        baseline_config=args.baseline_config,
        baseline_ckpt=args.baseline_ckpt,
        kd_config=args.kd_config,
        kd_ckpt=args.kd_ckpt,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_train_data=args.use_train_data
    )


if __name__ == '__main__':
    main()
