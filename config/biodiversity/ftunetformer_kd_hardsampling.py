"""Configuration for FTUNetFormer with KD + Hard Sample Oversampling."""

from torch.utils.data import DataLoader, WeightedRandomSampler
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.models.teacher_unet import TeacherUNet
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix
from tools.utils import Lookahead
from tools.utils import process_model_params
import torch

# training hparam
max_epoch = 65
ignore_index = 0
train_batch_size = 2  # reduced from 4 to avoid OOM with KD
val_batch_size = 2
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = 6
classes = CLASSES

# knowledge distillation params
kd_temperature = 2.0
kd_alpha = 0.5  # Optimal from sweep
rangeland_split_alpha = 0.7
teacher_checkpoint = 'pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth'

# training config
weights_name = "ftunetformer-kd-hardsampling-512-crop-ms-e45"
weights_path = "model_weights/biodiversity/{}".format(weights_name)
test_weights_name = "ftunetformer-kd-hardsampling-512-crop-ms-e45"
log_name = 'biodiversity/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'
# Start from scratch with hard sample oversampling
resume_ckpt_path = None

# define the student network
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# define the teacher network
teacher = TeacherUNet(num_classes=9, pretrained=True)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

# create KD helper
mapping_matrix = create_mapping_matrix(alpha=rangeland_split_alpha)
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)

# define the combined loss
hard_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0),
                      DiceLoss(smooth=0.05, ignore_index=0), 1.0, 1.0)

class KDLoss(nn.Module):
    """Combined hard target and knowledge distillation loss."""
    
    def __init__(self, hard_loss, kd_helper, alpha=0.5):
        super().__init__()
        self.hard_loss = hard_loss
        self.kd_helper = kd_helper
        self.alpha = alpha
    
    def forward(self, student_logits, targets, teacher_logits):
        hard_loss = self.hard_loss(student_logits, targets)
        kd_loss = self.kd_helper.compute_kd_loss(student_logits, teacher_logits)
        return (1 - self.alpha) * hard_loss + self.alpha * kd_loss

loss = KDLoss(hard_loss, kd_helper, alpha=kd_alpha)
use_aux_loss = False

# Load sample weights from analysis
sample_weights_path = 'sample_weights.txt'
sample_weights = []
with open(sample_weights_path, 'r') as f:
    for line in f:
        idx, weight = line.strip().split('\t')
        sample_weights.append(float(weight))

print(f"Loaded {len(sample_weights)} sample weights from {sample_weights_path}")
print(f"Weight range: {min(sample_weights):.2f} - {max(sample_weights):.2f}")
print(f"Mean weight: {sum(sample_weights)/len(sample_weights):.2f}")

# define the datasets
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
    data_root='data/biodiversity/Test')

# Create weighted sampler for oversampling hard samples
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True  # Allow sampling same image multiple times per epoch
)

# Use sampler instead of shuffle
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    sampler=sampler,  # Use weighted sampler
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

# Get optimizer params with layerwise learning rates (exactly like the working config)
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
