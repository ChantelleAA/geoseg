"""
Step 2: Finetune pretrained student on target-only data with KD from teacher
- Start from pretrained checkpoint (student_pretrain_all.pt)
- Train only on target biodiversity data (no OpenEarthMap)
- Use KD with confidence masking (threshold=0.5)
- 3-5 epoch CE-only warmup, then switch to CE+KD
- α=0.7, T=3, Lower LR (1e-4 → 3e-5)
- Output: student_ft_target_kd.pt
"""

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.models.teacher_unet import TeacherUNet
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix
from tools.utils import Lookahead
from tools.utils import process_model_params
import torch
import torch.nn as nn

# Training hyperparameters
max_epoch = 25  # Shorter finetune (3-5 warmup + 20-22 KD epochs)
ignore_index = 0
train_batch_size = 2  # Same as pretraining
val_batch_size = 2
lr = 1e-4  # Lower LR for finetuning
weight_decay = 2.5e-4
backbone_lr = 3e-5  # Lower backbone LR
backbone_weight_decay = 2.5e-4
num_classes = 6
classes = CLASSES

# Knowledge distillation params
kd_temperature = 3.0  # T=3 as requested
kd_alpha = 0.7  # α=0.7 as requested (weight for KD loss)
rangeland_split_alpha = 0.7  # Rangeland → Grassland (70%) + SemiNatural (30%)
confidence_threshold = 0.5  # Ignore teacher predictions where max prob < 0.5
teacher_checkpoint = 'pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth'

# Warmup settings
warmup_epochs = 5  # CE-only warmup before adding KD
current_epoch = 0  # This will be tracked by the training loop

# Training config
weights_name = "student_ft_target_kd"
weights_path = "model_weights/biodiversity/{}".format(weights_name)
test_weights_name = "student_ft_target_kd"
log_name = 'biodiversity/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = 'model_weights/biodiversity/student_pretrain_all/last.ckpt'  # Load pretrained student
gpus = 'auto'
resume_ckpt_path = None

# Define student network (load from pretrained checkpoint)
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# Define teacher network (9 classes, frozen)
teacher = TeacherUNet(num_classes=9, pretrained=True)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

# Create KD helper with confidence masking
mapping_matrix = create_mapping_matrix(alpha=rangeland_split_alpha)
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)

# Define hard loss (CE + Dice)
hard_loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0),
    DiceLoss(smooth=0.05, ignore_index=0), 
    1.0, 1.0
)


class AdaptiveKDLoss(nn.Module):
    """
    Combined loss with warmup and confidence masking.
    - First warmup_epochs: CE + Dice only (no KD)
    - After warmup: (1-α)·CE + α·KD with confidence masking
    """
    
    def __init__(self, hard_loss, kd_helper, alpha=0.7, warmup_epochs=5, confidence_threshold=0.5):
        super().__init__()
        self.hard_loss = hard_loss
        self.kd_helper = kd_helper
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.confidence_threshold = confidence_threshold
        self.current_epoch = 0
    
    def forward(self, student_logits, targets, teacher_logits):
        # Always compute hard loss
        hard_loss_val = self.hard_loss(student_logits, targets)
        
        # During warmup: only hard loss
        if self.current_epoch < self.warmup_epochs:
            return hard_loss_val
        
        # After warmup: add KD loss with confidence masking
        kd_loss = self.kd_helper.compute_kd_loss(
            student_logits, 
            teacher_logits,
            confidence_threshold=self.confidence_threshold
        )
        
        return (1 - self.alpha) * hard_loss_val + self.alpha * kd_loss


loss = AdaptiveKDLoss(
    hard_loss, 
    kd_helper, 
    alpha=kd_alpha, 
    warmup_epochs=warmup_epochs,
    confidence_threshold=confidence_threshold
)

use_aux_loss = False

# Define dataloaders - TARGET DATA ONLY (no OpenEarthMap)
train_dataset = BiodiversityTrainDataset(
    data_root='data/biodiversity/Train',
    mosaic_ratio=0.25,
    transform=train_aug
)

val_dataset = BiodiversityTrainDataset(
    data_root='data/biodiversity/Val',
    mosaic_ratio=0.0,
    transform=val_aug
)

test_dataset = BiodiversityTestDataset(
    data_root='data/biodiversity/Test'
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# Define optimizer with lower learning rate
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

# LR schedule: start at 1e-4, decay to 3e-5
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_epoch - warmup_epochs,
    eta_min=3e-5
)
