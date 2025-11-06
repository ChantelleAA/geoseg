"""Ensemble multiple models for improved biodiversity segmentation."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path

# Import configurations
import sys
sys.path.append('.')
from config.biodiversity.ftunetformer import net as baseline_net, val_loader, classes, num_classes


def load_model(checkpoint_path, model):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'net.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('net.', '') if key.startswith('net.') else key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model


class EnsembleModel:
    """Ensemble multiple segmentation models."""
    
    def __init__(self, models, weights=None, ensemble_mode='weighted_avg'):
        """
        Args:
            models: List of models
            weights: List of weights for each model (must sum to 1.0)
            ensemble_mode: 'weighted_avg', 'voting', or 'max'
        """
        self.models = models
        self.ensemble_mode = ensemble_mode
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        assert len(weights) == len(models), "Number of weights must match number of models"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        self.weights = torch.FloatTensor(weights)
        
        # Move models to GPU
        for model in self.models:
            model.cuda()
            model.eval()
    
    def predict(self, images):
        """
        Ensemble prediction.
        
        Args:
            images: (N, C, H, W) input images
            
        Returns:
            (N, num_classes, H, W) ensemble predictions
        """
        with torch.no_grad():
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model(images)
                # Apply softmax to get probabilities
                pred_prob = F.softmax(pred, dim=1)
                predictions.append(pred_prob)
            
            # Ensemble the predictions
            if self.ensemble_mode == 'weighted_avg':
                # Weighted average of probabilities
                weights = self.weights.view(1, -1, 1, 1, 1).cuda()
                ensemble_pred = sum(w * p for w, p in zip(weights.squeeze(0), predictions))
                
            elif self.ensemble_mode == 'voting':
                # Majority voting on class predictions
                pred_classes = [torch.argmax(p, dim=1) for p in predictions]
                stacked = torch.stack(pred_classes, dim=0)
                ensemble_class = torch.mode(stacked, dim=0)[0]
                # Convert back to probabilities (one-hot)
                ensemble_pred = F.one_hot(ensemble_class, num_classes=predictions[0].shape[1])
                ensemble_pred = ensemble_pred.permute(0, 3, 1, 2).float()
                
            elif self.ensemble_mode == 'max':
                # Take maximum probability for each class
                stacked = torch.stack(predictions, dim=0)
                ensemble_pred = torch.max(stacked, dim=0)[0]
            
            else:
                raise ValueError(f"Unknown ensemble mode: {self.ensemble_mode}")
            
            return ensemble_pred


def evaluate_ensemble(ensemble, val_loader, num_classes=6):
    """Evaluate ensemble model."""
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for batch in tqdm(val_loader, desc="Evaluating ensemble"):
        images = batch['img'].cuda()
        masks = batch['gt_semantic_seg'].cuda()
        
        # Get ensemble prediction
        outputs = ensemble.predict(images)
        preds = torch.argmax(outputs, dim=1)
        
        # Update confusion matrix
        preds_np = preds.cpu().numpy().flatten()
        masks_np = masks.cpu().numpy().flatten()
        
        for pred, mask in zip(preds_np, masks_np):
            if mask != 0:  # ignore background
                confusion_matrix[mask, pred] += 1
    
    # Calculate metrics
    iou_per_class = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        iou = intersection / (union + 1e-10)
        iou_per_class.append(iou)
    
    mean_iou = np.mean(iou_per_class)
    
    # Calculate accuracy
    total = confusion_matrix.sum()
    correct = np.diag(confusion_matrix).sum()
    accuracy = correct / total
    
    return {
        'mIoU': mean_iou,
        'IoU_per_class': iou_per_class,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix
    }


def find_optimal_weights(models, val_loader, num_classes=6, search_steps=10):
    """Grid search for optimal ensemble weights (2 models only)."""
    if len(models) != 2:
        print("Optimal weight search only supports 2 models. Using equal weights.")
        return [1.0 / len(models)] * len(models)
    
    best_miou = 0
    best_weights = [0.5, 0.5]
    
    print("\n" + "="*60)
    print("Searching for optimal ensemble weights...")
    print("="*60)
    
    for i in range(search_steps + 1):
        w1 = i / search_steps
        w2 = 1.0 - w1
        
        ensemble = EnsembleModel(models, weights=[w1, w2], ensemble_mode='weighted_avg')
        results = evaluate_ensemble(ensemble, val_loader, num_classes)
        
        print(f"Weights: [{w1:.2f}, {w2:.2f}] -> mIoU: {results['mIoU']:.4f}")
        
        if results['mIoU'] > best_miou:
            best_miou = results['mIoU']
            best_weights = [w1, w2]
    
    print("="*60)
    print(f"Best weights: {best_weights} with mIoU: {best_miou:.4f}")
    print("="*60)
    
    return best_weights


if __name__ == '__main__':
    print("="*60)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*60)
    
    # Load baseline model
    print("\n1. Loading baseline model...")
    baseline_checkpoint = "model_weights/biodiversity/ftunetformer-512-crop-ms-e45/ftunetformer-512-crop-ms-e45.ckpt"
    from geoseg.models.FTUNetFormer import ft_unetformer
    baseline_model = ft_unetformer(num_classes=num_classes, decoder_channels=256)
    baseline_model = load_model(baseline_checkpoint, baseline_model)
    print("✓ Baseline model loaded")
    
    # Load KD model
    print("\n2. Loading KD model...")
    kd_checkpoint = "model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/ftunetformer-kd-512-crop-ms-e45.ckpt"
    kd_model = ft_unetformer(num_classes=num_classes, decoder_channels=256)
    kd_model = load_model(kd_checkpoint, kd_model)
    print("✓ KD model loaded")
    
    models = [baseline_model, kd_model]
    
    # Evaluate individual models first
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    print("\nBaseline model:")
    baseline_ensemble = EnsembleModel([baseline_model], weights=[1.0])
    baseline_results = evaluate_ensemble(baseline_ensemble, val_loader, num_classes)
    print(f"  mIoU: {baseline_results['mIoU']:.4f}")
    print(f"  OA:   {baseline_results['accuracy']:.4f}")
    for i, class_name in enumerate(classes):
        print(f"    {class_name:25s}: {baseline_results['IoU_per_class'][i]:.4f}")
    
    print("\nKD model:")
    kd_ensemble = EnsembleModel([kd_model], weights=[1.0])
    kd_results = evaluate_ensemble(kd_ensemble, val_loader, num_classes)
    print(f"  mIoU: {kd_results['mIoU']:.4f}")
    print(f"  OA:   {kd_results['accuracy']:.4f}")
    for i, class_name in enumerate(classes):
        print(f"    {class_name:25s}: {kd_results['IoU_per_class'][i]:.4f}")
    
    # Find optimal weights
    optimal_weights = find_optimal_weights(models, val_loader, num_classes, search_steps=10)
    
    # Evaluate with optimal weights
    print("\n" + "="*60)
    print("ENSEMBLE WITH OPTIMAL WEIGHTS")
    print("="*60)
    ensemble = EnsembleModel(models, weights=optimal_weights, ensemble_mode='weighted_avg')
    ensemble_results = evaluate_ensemble(ensemble, val_loader, num_classes)
    
    print(f"\nEnsemble Results (weights: {optimal_weights}):")
    print(f"  mIoU: {ensemble_results['mIoU']:.4f}")
    print(f"  OA:   {ensemble_results['accuracy']:.4f}")
    print("\nPer-class IoU:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name:25s}: {ensemble_results['IoU_per_class'][i]:.4f}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline:  {baseline_results['mIoU']:.4f}")
    print(f"KD:        {kd_results['mIoU']:.4f}")
    print(f"Ensemble:  {ensemble_results['mIoU']:.4f}")
    improvement = (ensemble_results['mIoU'] - baseline_results['mIoU']) * 100
    print(f"Improvement: {improvement:+.2f}%")
    print("="*60)
