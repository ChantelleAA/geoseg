#!/usr/bin/env python3
"""
Generate predictions on training set using baseline and KD models.
Save the prediction masks for replacing ground truth.
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add geoseg to path
sys.path.insert(0, '/home/chantelle/Desktop/UCD/ai_sandbox/geoseg')

from geoseg.models.FTUNetFormer import ft_unetformer
from torch.utils.data import Dataset, DataLoader
import albumentations as albu

class InferenceDataset(Dataset):
    """Dataset for inference on images from multiple splits."""
    
    def __init__(self, base_dir, image_list=None):
        """
        Args:
            base_dir: Base biodiversity directory containing Train/Val/Test
            image_list: List of filenames to find across splits, or None for all
        """
        self.base_dir = Path(base_dir)
        
        if image_list:
            # Find images across Train/Val/Test splits
            self.image_files = []
            for filename in image_list:
                found = False
                # Check Train, Val, Test
                for split in ['Train', 'Val', 'Test']:
                    img_path = self.base_dir / split / 'Rural' / 'images_png' / filename
                    if img_path.exists():
                        self.image_files.append(img_path)
                        found = True
                        break
                if not found:
                    print(f"  Warning: {filename} not found in any split")
        else:
            # Use all images from all splits
            self.image_files = []
            for split in ['Train', 'Val', 'Test']:
                split_dir = self.base_dir / split / 'Rural' / 'images_png'
                if split_dir.exists():
                    self.image_files.extend(sorted(list(split_dir.glob('*.png'))))
        
        # Simple inference transform - resize to 512x512
        self.transform = albu.Compose([
            albu.Resize(512, 512),
            albu.Normalize(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original size for resizing back
        original_size = img.shape[:2]  # (h, w)
        
        # Transform
        augmented = self.transform(image=img)
        img = augmented['image']
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        return {
            'img': img,
            'filename': img_path.name,
            'original_size': original_size
        }

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    model = ft_unetformer(num_classes=6, decoder_channels=256)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.', 'net.', or 'teacher.' prefix if present, skip teacher weights
    new_state_dict = {}
    for k, v in state_dict.items():
        # Skip teacher model weights
        if k.startswith('teacher.'):
            continue
        # Remove prefixes
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        elif k.startswith('net.'):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model

@torch.no_grad()
def generate_predictions(model, dataloader, output_dir, device):
    """
    Generate predictions and save as PNG masks.
    
    Args:
        model: Loaded model
        dataloader: DataLoader with images
        output_dir: Where to save prediction masks
        device: cuda or cpu
    
    Returns:
        List of saved filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    print(f"\n🔮 Generating predictions...")
    print(f"Output directory: {output_dir}")
    
    for batch in tqdm(dataloader, desc="Predicting"):
        imgs = batch['img'].to(device)
        filenames = batch['filename']
        original_sizes = batch['original_size']
        
        # Forward pass
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Save each prediction
        for i, (pred, filename) in enumerate(zip(preds, filenames)):
            # Resize back to original size
            h, w = original_sizes[0][i].item(), original_sizes[1][i].item()
            pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Save as grayscale PNG
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), pred_resized)
            saved_files.append(filename)
    
    print(f"✅ Saved {len(saved_files)} predictions to {output_dir}")
    return saved_files

def get_image_ids_from_review(baseline_list, kd_list):
    """
    Convert Katherine's review lists to full image filenames.
    Returns lists of filenames that need predictions.
    """
    baseline_filenames = []
    kd_filenames = []
    
    for short_name in baseline_list:
        # Files from other regions (Ireland, Denmark, Colombia) don't have 'biodiversity_' prefix
        if any(prefix in short_name for prefix in ['ireland', 'den', 'col']):
            full_name = f"{short_name}.png"
        else:
            full_name = f"biodiversity_{short_name}.png"
        baseline_filenames.append(full_name)
    
    for short_name in kd_list:
        if any(prefix in short_name for prefix in ['ireland', 'den', 'col']):
            full_name = f"{short_name}.png"
        else:
            full_name = f"biodiversity_{short_name}.png"
        kd_filenames.append(full_name)
    
    return baseline_filenames, kd_filenames

def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions on training set for mask replacement'
    )
    parser.add_argument(
        '--baseline-checkpoint',
        type=str,
        default='/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/model_weights/biodiversity/ensemble_checkpoints/student_target_only.ckpt',
        help='Path to baseline model checkpoint'
    )
    parser.add_argument(
        '--kd-checkpoint',
        type=str,
        default='/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/model_weights/biodiversity/student_kd/last.ckpt',
        help='Path to KD model checkpoint'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/data/biodiversity',
        help='Base directory with Train/Val/Test splits'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/outputs/mask_replacement_predictions',
        help='Base output directory for predictions'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--generate-all',
        action='store_true',
        help='Generate predictions for ALL training images (not just reviewed ones)'
    )
    
    args = parser.parse_args()
    
    # Katherine's review results
    BASELINE_BETTER = [
        '1124', '1843', '1182', '1091', '1658', '1609', '0649', '1488', '0258', '2271',
        '0652', '2313', '0822', '1985', '0520', '0601', '0632', '1010', '0912', '1258',
        '1153', '1922', '1580', '0691', '0275', '1179', '1133', '1070', '1711', '0861',
        '0202', '0059', '2176', '1079', '0996', '1969', '0222', '1840', '1050', '1144',
        '1838', '1041', '1747', '1112', '1076', '0408', '1387', '0699', '0118', '2089',
        '0823', '0993', '1430', '2277', '0531', '1340', 'ireland1_0044', '0418', '1732',
        '2261', '0729', '0735', '0703', '1802', '1009', '1881', '1466', '1122', '1592',
        '0184', '2205', '0183', '1307', '0416', '1916', '0193', '1997', '0395', '2239',
        '1424', '1711', '1232'
    ]
    
    KD_BETTER = [
        '2381', '0998', '0755', '1420', '1687', '0862', '1974', '0748', '0384', '1120',
        'ireland2_0151', '1127', 'den0_0010', '0013', '0327', '1909', '1437', '1645',
        '1554', '2378', '0504', '1907', '2038', 'den3_0016', '0051', '1569', '1894',
        'col1_0035', '0647', '0267', '0122', '0824', '0551', '0986', '0577', '0023',
        '0908', '2389', '1422', '0105', '0070', '0019', 'ireland2_0070', '0163', '2404',
        '2056', '2115', '1593', '2241', '2158', '2421', '0300', '0113', 'den1_0011',
        '0026', '0400', '1435', '1682', '2148', '1136', '1035', 'ireland1_0055', '0928',
        '0610', 'den0_0019', '1990', '0914', '2368', '1765', '0274', '1880', '1588',
        '1956', '1607', '1568', '1330', 'ireland2_0038', '0550', '1493', 'den0_0006',
        'den0_0042', '1418', 'den0_0005', '0542', '0191', '2054', '0585', '2249', '1911',
        '2307', '2137', '2226', '2206', '0297', '2237', '0421', 'ireland2_0196', '2394',
        '1233', '1623', '1781', 'ireland2_0142', '0855', 'ireland2_0028', '1462',
        'ireland2_0090', '2154', '2258'
    ]
    
    print("="*70)
    print("🎯 GENERATING PREDICTIONS FOR GROUND TRUTH REPLACEMENT")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Check if checkpoints exist
    if not Path(args.baseline_checkpoint).exists():
        print(f"❌ Baseline checkpoint not found: {args.baseline_checkpoint}")
        return
    
    if not Path(args.kd_checkpoint).exists():
        print(f"❌ KD checkpoint not found: {args.kd_checkpoint}")
        return
    
    if not Path(args.image_dir).exists():
        print(f"❌ Image directory not found: {args.image_dir}")
        return
    
    device = torch.device(args.device)
    
    # Prepare output directories
    baseline_output = Path(args.output_dir) / 'baseline'
    kd_output = Path(args.output_dir) / 'kd'
    
    baseline_output.mkdir(parents=True, exist_ok=True)
    kd_output.mkdir(parents=True, exist_ok=True)
    
    # Get image lists
    baseline_filenames, kd_filenames = get_image_ids_from_review(BASELINE_BETTER, KD_BETTER)
    
    print(f"\n📋 Katherine's Review:")
    print(f"   - Baseline better: {len(baseline_filenames)} images")
    print(f"   - KD better: {len(kd_filenames)} images")
    print(f"   - Total to predict: {len(set(baseline_filenames) | set(kd_filenames))} unique images")
    
    # Generate baseline predictions
    print(f"\n{'='*70}")
    print("🔵 GENERATING BASELINE MODEL PREDICTIONS")
    print(f"{'='*70}")
    
    baseline_model = load_model(args.baseline_checkpoint, device)
    
    if args.generate_all:
        print("Generating predictions for ALL training images...")
        baseline_dataset = InferenceDataset(args.image_dir, image_list=None)
    else:
        print(f"Generating predictions for {len(baseline_filenames)} reviewed images...")
        # baseline_filenames already has full filenames like 'biodiversity_1843.png'
        baseline_dataset = InferenceDataset(args.image_dir, image_list=baseline_filenames)
    
    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    baseline_saved = generate_predictions(baseline_model, baseline_loader, baseline_output, device)
    
    # Clear memory
    del baseline_model
    torch.cuda.empty_cache()
    
    # Generate KD predictions
    print(f"\n{'='*70}")
    print("🟢 GENERATING KD MODEL PREDICTIONS")
    print(f"{'='*70}")
    
    kd_model = load_model(args.kd_checkpoint, device)
    
    if args.generate_all:
        print("Generating predictions for ALL training images...")
        kd_dataset = InferenceDataset(args.image_dir, image_list=None)
    else:
        print(f"Generating predictions for {len(kd_filenames)} reviewed images...")
        # kd_filenames already has full filenames like 'biodiversity_2381.png'
        kd_dataset = InferenceDataset(args.image_dir, image_list=kd_filenames)
    
    kd_loader = DataLoader(
        kd_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    kd_saved = generate_predictions(kd_model, kd_loader, kd_output, device)
    
    del kd_model
    torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PREDICTION GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Statistics:")
    print(f"   - Baseline predictions: {len(baseline_saved)} files")
    print(f"   - KD predictions: {len(kd_saved)} files")
    print(f"\n📁 Saved to:")
    print(f"   - Baseline: {baseline_output}")
    print(f"   - KD: {kd_output}")
    print(f"\n⏭️  Next step: Run the replacement script")
    print(f"   python replace_ground_truth_masks.py \\")
    print(f"     --baseline-predictions {baseline_output} \\")
    print(f"     --kd-predictions {kd_output} \\")
    print(f"     --dry-run  # Remove --dry-run to actually replace")
    print("="*70)

if __name__ == '__main__':
    main()
