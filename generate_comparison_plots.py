"""Generate comparison plots: Ground Truth vs Baseline vs KD predictions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tools.cfg import py2cfg
import os
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_models(baseline_ckpt, kd_ckpt, config):
    """Load baseline and KD models."""
    from geoseg.models.FTUNetFormer import ft_unetformer
    
    # Load baseline model
    baseline = ft_unetformer(num_classes=config.num_classes)
    baseline_state = torch.load(baseline_ckpt, map_location='cpu')
    baseline.load_state_dict(baseline_state['state_dict'] if 'state_dict' in baseline_state else baseline_state)
    baseline.eval().cuda()
    
    # Load KD model
    kd_model = ft_unetformer(num_classes=config.num_classes)
    kd_state = torch.load(kd_ckpt, map_location='cpu')
    kd_model.load_state_dict(kd_state['state_dict'] if 'state_dict' in kd_state else kd_state)
    kd_model.eval().cuda()
    
    return baseline, kd_model


def create_comparison_plot(img, gt, baseline_pred, kd_pred, idx, output_dir):
    """Create a 4-panel comparison plot."""
    # Color map for classes
    colors = {
        0: [0, 0, 0],           # Background - Black
        1: [0, 100, 0],         # Forest - Dark Green
        2: [144, 238, 144],     # Grassland - Light Green
        3: [255, 255, 0],       # Cropland - Yellow
        4: [255, 0, 0],         # Settlement - Red
        5: [173, 216, 230]      # SemiNatural - Light Blue
    }
    
    def mask_to_rgb(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            rgb[mask == class_id] = color
        return rgb
    
    # Convert image to displayable format (denormalize)
    img_display = img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    # Convert masks to RGB
    gt_rgb = mask_to_rgb(gt.cpu().numpy())
    baseline_rgb = mask_to_rgb(baseline_pred.cpu().numpy())
    kd_rgb = mask_to_rgb(kd_pred.cpu().numpy())
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_display)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gt_rgb)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(baseline_rgb)
    axes[2].set_title('Baseline Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(kd_rgb)
    axes[3].set_title('KD Prediction', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_{idx:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    # Paths
    baseline_ckpt = 'model_weights/biodiversity/ftunetformer-512-crop-ms-e45/last.ckpt'
    kd_ckpt = 'model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/v5.ckpt'
    config_path = 'config/biodiversity/ftunetformer_kd.py'
    output_dir = Path('comparison_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load config
    config = py2cfg(config_path)
    
    print("Loading models...")
    baseline, kd_model = load_models(baseline_ckpt, kd_ckpt, config)
    
    # Get validation dataloader
    val_loader = config.val_loader
    
    print(f"Generating comparison plots for {len(val_loader.dataset)} validation samples...")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Generate plots
    num_samples = min(200, len(val_loader.dataset))
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Generating plots")):
            if sample_count >= num_samples:
                break
            
            imgs = batch['img'].cuda()
            masks = batch['gt_semantic_seg']
            
            # Get predictions
            baseline_preds = baseline(imgs).argmax(dim=1)
            kd_preds = kd_model(imgs).argmax(dim=1)
            
            # Process each image in batch
            for i in range(imgs.shape[0]):
                if sample_count >= num_samples:
                    break
                
                create_comparison_plot(
                    imgs[i],
                    masks[i],
                    baseline_preds[i],
                    kd_preds[i],
                    sample_count,
                    output_dir
                )
                sample_count += 1
    
    print(f"\nDone! Generated {sample_count} comparison plots in: {output_dir.absolute()}")
    print("\nClass color legend:")
    print("  Black:      Background")
    print("  Dark Green: Forest Land")
    print("  Light Green: Grassland")
    print("  Yellow:     Cropland")
    print("  Red:        Settlement")
    print("  Light Blue: SemiNatural Grassland")


if __name__ == '__main__':
    main()
