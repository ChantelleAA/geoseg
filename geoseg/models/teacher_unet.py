"""Teacher U-Net model based on EfficientNet-B4 backbone using segmentation_models_pytorch."""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("Warning: segmentation_models_pytorch not installed. Install with: pip install segmentation-models-pytorch")


class TeacherUNet(nn.Module):
    """U-Net with EfficientNet-B4 encoder for the teacher model.
    
    This uses the segmentation_models_pytorch library which matches
    the checkpoint format from the pretrained weights.
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        
        if not HAS_SMP:
            raise ImportError(
                "segmentation_models_pytorch is required for TeacherUNet. "
                "Install with: pip install segmentation-models-pytorch"
            )
        
        # Create U-Net with EfficientNet-B4 encoder
        # This matches the architecture of the pretrained checkpoint
        self.model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            activation=None  # No activation, we'll apply softmax in loss
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net model."""
        return self.model(x)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load the state dict
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded teacher checkpoint from {checkpoint_path}")
        except RuntimeError as e:
            print(f"Warning: Could not load checkpoint with strict=True: {e}")
            print("Attempting to load with strict=False...")
            self.model.load_state_dict(state_dict, strict=False)
            print("Loaded checkpoint with strict=False (some weights may be missing)")
    
    def freeze(self):
        """Freeze all parameters and set to eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print("Teacher model frozen and set to eval mode")