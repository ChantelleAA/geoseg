"""
Step 4: Ensemble inference with multi-scale TTA
Combines:
  - student_target_only.ckpt (baseline, 83.93% mIoU)
  - student_ft_target_kd.ckpt (finetuned with KD)
  
TTA: H/V flips + multi-scale (0.75, 1.0, 1.25)
Combines logits via averaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import cv2
import argparse
from pathlib import Path

from geoseg.models.FTUNetFormer import ft_unetformer


def load_model(checkpoint_path, num_classes=6):
    """Load model from checkpoint"""
    model = ft_unetformer(num_classes=num_classes, decoder_channels=256)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'net.' prefix if present (from Lightning)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def resize_image(img, scale_factor):
    """Resize image by scale factor"""
    if scale_factor == 1.0:
        return img
    
    h, w = img.shape[-2:]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


def predict_with_tta(models, image, scales=(0.75, 1.0, 1.25), use_flips=True):
    """
    Predict with test-time augmentation
    
    Args:
        models: list of models
        image: torch.Tensor (1, C, H, W)
        scales: tuple of scale factors
        use_flips: whether to use horizontal/vertical flips
    
    Returns:
        torch.Tensor (H, W) - predicted class indices
    """
    device = image.device
    orig_h, orig_w = image.shape[-2:]
    logits_sum = torch.zeros(1, 6, orig_h, orig_w, device=device)
    count = 0
    
    # Augmentation transforms
    augmentations = [
        (lambda x: x, lambda x: x),  # No augmentation
    ]
    
    if use_flips:
        augmentations.extend([
            (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),  # Horizontal flip
            (lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [3])),  # Vertical flip
        ])
    
    # Iterate over scales
    for scale in scales:
        # Resize image
        scaled_img = resize_image(image, scale)
        
        # Iterate over augmentations
        for aug_fn, unaug_fn in augmentations:
            # Apply augmentation
            aug_img = aug_fn(scaled_img)
            
            # Predict with each model
            for model in models:
                with torch.no_grad():
                    logits = model(aug_img)
                
                # Undo augmentation
                logits = unaug_fn(logits)
                
                # Resize back to original size
                if scale != 1.0:
                    logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                
                # Accumulate logits
                logits_sum += logits
                count += 1
    
    # Average logits
    avg_logits = logits_sum / count
    
    # Get predictions
    pred = avg_logits.argmax(dim=1).squeeze(0)  # (H, W)
    return pred


def preprocess_image(img_path):
    """Load and preprocess image"""
    # Load image
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    
    # Normalize (same as training)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # To tensor (C, H, W)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0)  # (1, C, H, W)
    
    return img


def colorize_mask(mask, palette):
    """Convert mask to RGB visualization"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(palette):
        rgb[mask == class_idx] = color
    
    return rgb


# Color palette
PALETTE = [
    [11, 246, 210],   # 0 Background
    [250, 62, 119],   # 1 Forest land
    [168, 232, 84],   # 2 Grassland
    [242, 180, 92],   # 3 Cropland
    [116, 116, 116],  # 4 Settlement
    [255, 214, 33],   # 5 Seminatural Grassland
]


def main():
    parser = argparse.ArgumentParser(description="Ensemble inference with multi-scale TTA")
    parser.add_argument('--input-dir', required=True, help='Input directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory for predictions')
    parser.add_argument('--checkpoint1', default='model_weights/biodiversity/ensemble_checkpoints/student_target_only.ckpt',
                       help='Path to first checkpoint (baseline)')
    parser.add_argument('--checkpoint2', default='model_weights/biodiversity/ensemble_checkpoints/student_ft_target_kd.ckpt',
                       help='Path to second checkpoint (finetuned with KD)')
    parser.add_argument('--scales', default='0.75,1.0,1.25', help='Comma-separated scale factors')
    parser.add_argument('--no-flips', action='store_true', help='Disable horizontal/vertical flips')
    parser.add_argument('--img-suffix', default='.png', help='Image file suffix')
    parser.add_argument('--save-rgb', action='store_true', help='Save RGB visualizations')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Parse scales
    scales = [float(s) for s in args.scales.split(',')]
    print(f"Using scales: {scales}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_rgb:
        os.makedirs(os.path.join(args.output_dir, 'rgb'), exist_ok=True)
    
    # Load models
    print("Loading models...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    models = []
    if os.path.exists(args.checkpoint1):
        print(f"  Loading model 1: {args.checkpoint1}")
        model1 = load_model(args.checkpoint1)
        model1.to(device)
        model1.eval()
        models.append(model1)
    else:
        print(f"  Warning: Checkpoint 1 not found: {args.checkpoint1}")
    
    if os.path.exists(args.checkpoint2):
        print(f"  Loading model 2: {args.checkpoint2}")
        model2 = load_model(args.checkpoint2)
        model2.to(device)
        model2.eval()
        models.append(model2)
    else:
        print(f"  Warning: Checkpoint 2 not found: {args.checkpoint2}")
    
    if len(models) == 0:
        print("Error: No valid checkpoints found!")
        return
    
    print(f"Loaded {len(models)} model(s)")
    
    # Get list of images
    img_paths = glob.glob(os.path.join(args.input_dir, f"*{args.img_suffix}"))
    print(f"Found {len(img_paths)} images")
    
    # Process images
    for img_path in tqdm(img_paths, desc="Processing"):
        # Load and preprocess image
        img = preprocess_image(img_path)
        img = img.to(device)
        
        # Predict with TTA
        pred = predict_with_tta(models, img, scales=scales, use_flips=not args.no_flips)
        
        # Save prediction
        pred_np = pred.cpu().numpy().astype(np.uint8)
        stem = Path(img_path).stem
        
        # Save grayscale mask
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}.png"), pred_np)
        
        # Save RGB visualization if requested
        if args.save_rgb:
            rgb = colorize_mask(pred_np, PALETTE)
            cv2.imwrite(os.path.join(args.output_dir, 'rgb', f"{stem}.png"), 
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    print(f"\nDone! Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
