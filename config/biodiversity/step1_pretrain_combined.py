"""
Step 1: Pretrain FTUNetFormer on combined dataset (target + OpenEarthMap)
- Data: Target (1615) + OpenEarthMap (3000) = 4615 samples
- Resolution: 512x512
- Loss: CE + Dice (no KD)
- Output: student_pretrain_all.pt
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_combined_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead
from tools.utils import process_model_params
import torch

# Training hyperparameters
max_epoch = 50  # More epochs since we have more data
ignore_index = 0
train_batch_size = 2  # Reduced for 512x512 resolution (VRAM constraint)
val_batch_size = 2
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = 6
classes = CLASSES

weights_name = "student_pretrain_all"
weights_path = "model_weights/biodiversity/{}".format(weights_name)
test_weights_name = "student_pretrain_all"
log_name = 'biodiversity/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # Train from scratch
gpus = 'auto'
resume_ckpt_path = "/home/chantelle/Desktop/UCD/ai_sandbox/geoseg/model_weights/biodiversity/student_pretrain_all/student_pretrain_all.ckpt"

# Define network
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# Define loss: CE + Dice (no KD for pretraining)
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0),
    DiceLoss(smooth=0.05, ignore_index=0), 
    1.0, 1.0
)

use_aux_loss = False

# Define dataloaders - using combined dataset at 512x512
train_dataset = BiodiversityCombinedDataset(
    data_root='data/biodiversity_combined/Train',
    img_dir='images_png',
    mask_dir='masks_png',
    transform=train_aug_512,
    img_size=(512, 512)
)

val_dataset = BiodiversityValDataset(
    data_root='data/biodiversity_combined/Val',
    img_dir='images_png',
    mask_dir='masks_png',
    transform=val_aug_512,
    img_size=(512, 512)
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

# Define optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
