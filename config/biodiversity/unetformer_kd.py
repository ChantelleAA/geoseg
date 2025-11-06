"""Configuration for UNetFormer with knowledge distillation from FTUNetFormer teacher."""

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 80
ignore_index = 0
train_batch_size = 4  # UNetFormer is lighter than FTUNetFormer
val_batch_size = 4
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = 6
classes = CLASSES

# knowledge distillation params
kd_temperature = 2.0
kd_alpha = 0.5  # Proven optimal from previous experiments
# Teacher: FTUNetFormer trained with KD from EfficientNet
teacher_checkpoint = 'model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/ftunetformer-kd-512-crop-ms-e45-v5.ckpt'

# training config
weights_name = "unetformer-kd-from-ftunet-512-crop-ms-e45"
weights_path = "model_weights/biodiversity/{}".format(weights_name)
test_weights_name = "unetformer-kd-from-ftunet-512-crop-ms-e45"
log_name = 'biodiversity/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = 'model_weights/biodiversity/unetformer-kd-from-ftunet-512-crop-ms-e45/last.ckpt'
gpus = 'auto'
resume_ckpt_path = 'model_weights/biodiversity/unetformer-kd-from-ftunet-512-crop-ms-e45/unetformer-kd-from-ftunet-512-crop-ms-e45.ckpt'

# define the student network (UNetFormer with ResNet18)
net = UNetFormer(
    decode_channels=64,
    dropout=0.1,
    backbone_name='swsl_resnet18',
    pretrained=True,  # Use ImageNet pretrained backbone
    window_size=8,
    num_classes=num_classes
)

# define the teacher network (FTUNetFormer trained with KD)
teacher = ft_unetformer(num_classes=num_classes, decoder_channels=256)
# Load teacher weights
teacher_ckpt = torch.load(teacher_checkpoint)
teacher_state_dict = {}
for key, value in teacher_ckpt['state_dict'].items():
    if key.startswith('net.'):
        new_key = key[4:]  # Remove 'net.' prefix
        teacher_state_dict[new_key] = value

teacher.load_state_dict(teacher_state_dict)
# Freeze teacher
for param in teacher.parameters():
    param.requires_grad = False
teacher.eval()

# define the combined loss
hard_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0),
                      DiceLoss(smooth=0.05, ignore_index=0), 1.0, 1.0)


class KDLoss(nn.Module):
    """Knowledge distillation loss for same number of classes (no mapping needed)."""
    
    def __init__(self, hard_loss, temperature=2.0, alpha=0.5):
        super().__init__()
        self.hard_loss = hard_loss
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, targets, teacher_logits):
        # Hard label loss
        hard_loss = self.hard_loss(student_logits, targets)
        
        # KD loss (soft targets from teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        return (1 - self.alpha) * hard_loss + self.alpha * kd_loss


loss = KDLoss(hard_loss, temperature=kd_temperature, alpha=kd_alpha)
use_aux_loss = True  # UNetFormer has auxiliary head

# define the dataloader
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
    pin_memory=True,
    shuffle=False,
    drop_last=False
)

# define the optimizer & scheduler
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}

net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
