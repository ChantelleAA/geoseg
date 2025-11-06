"""Configuration for FTUNetFormer with knowledge distillation from EfficientNet-B4 U-Net teacher."""

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.models.teacher_unet import TeacherUNet
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 45
ignore_index = 0
train_batch_size = 2  # reduced from 4 to avoid OOM with KD
val_batch_size = 2    # reduced from 4 to avoid OOM with KD
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = 6  # new taxonomy: Forest, Grassland, Cropland, Settlement, SemiNatural, Background
classes = CLASSES  # updated in biodiversity_dataset.py

# knowledge distillation params
kd_temperature = 2.0
kd_alpha = 0.5  # OPTIMIZED: weight for KD loss (1-alpha for hard targets) - sweep showed 0.5 best
rangeland_split_alpha = 0.7  # probability of mapping Rangeland to Grassland vs SemiNatural
teacher_checkpoint = 'pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth'  # path to teacher checkpoint

# training config
weights_name = "ftunetformer-kd-512-crop-ms-e45"
weights_path = "model_weights/biodiversity/{}".format(weights_name)
test_weights_name = "ftunetformer-kd-512-crop-ms-e45"
log_name = 'biodiversity/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None  # Train from scratch with corrected ground truth

# define the student network (6 classes)
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# define the teacher network (using smp U-Net with EfficientNet-B4)
# The checkpoint has 9 classes, which we'll remap to our 6 classes
teacher = TeacherUNet(num_classes=9, pretrained=True)  # 9 classes in checkpoint!
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()  # freeze and set to eval mode

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
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

# Optional: Cache teacher predictions
def cache_teacher_predictions():
    """Cache teacher predictions for the training set to save compute."""
    teacher.cuda()
    train_loader_nocache = DataLoader(
        BiodiversityTrainDataset(
            data_root='data/biodiversity/Train',
            mosaic_ratio=0.0,  # disable mosaic for caching
            transform=val_aug,
            use_original_classes=True  # use original 8 classes for teacher
        ),
        batch_size=val_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch in train_loader_nocache:
            img_ids = batch['img_id']
            teacher_logits = teacher(batch['img'].cuda())
            for img_id, logits in zip(img_ids, teacher_logits):
                kd_helper.cache_teacher_probs(img_id, logits.unsqueeze(0))
    
    teacher.cpu()  # free GPU memory

# Uncomment to cache teacher predictions before training
# cache_teacher_predictions()