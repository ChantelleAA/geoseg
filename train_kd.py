"""Knowledge Distillation Training Script."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class KD_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net  # student model
        self.teacher = config.teacher  # teacher model
        
        # Move teacher to same device as student will be
        # This happens automatically when we register it as a module
        # but we explicitly freeze and set to eval
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.loss = config.loss
        self.kd_helper = config.kd_helper
        self.classes = config.classes
        # Don't save hyperparameters - config contains unpicklable objects
        
        # Create evaluator for metrics
        self.train_evaluator = Evaluator(num_class=len(self.classes))
        self.val_evaluator = Evaluator(num_class=len(self.classes))
        
        # Store predictions and targets for epoch-end metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.net(x)

    def on_train_epoch_start(self):
        self.train_evaluator.reset()
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        img = batch['img']
        mask = batch['gt_semantic_seg']
        
        # Student forward pass
        student_logits = self.forward(img)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(img)
        
        # Compute KD loss
        loss = self.loss(student_logits, mask, teacher_logits)
        
        # Get predictions for metrics
        student_pred = torch.argmax(student_logits, dim=1)
        
        self.training_step_outputs.append({
            'loss': loss,
            'pred': student_pred.cpu().numpy(),
            'target': mask.cpu().numpy()
        })
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Compute metrics
        for output in self.training_step_outputs:
            self.train_evaluator.add_batch(output['target'], output['pred'])
        
        IoU_per_class = self.train_evaluator.Intersection_over_Union()
        mIoU = np.nanmean(IoU_per_class)
        F1_per_class = self.train_evaluator.F1()
        F1 = np.nanmean(F1_per_class)
        OA = self.train_evaluator.OA()
        
        # Log metrics with on_epoch=True to ensure they're saved
        self.log('train_mIoU', mIoU, on_epoch=True, prog_bar=True)
        self.log('train_F1', F1, on_epoch=True, prog_bar=True)
        self.log('train_OA', OA, on_epoch=True, prog_bar=True)
        
        # Print detailed metrics
        print(f"\ntrain: {{'mIoU': {mIoU}, 'F1': {F1}, 'OA': {OA}}}")
        class_metrics = {self.classes[i]: IoU_per_class[i] for i in range(len(self.classes))}
        print(class_metrics)
        
        self.training_step_outputs.clear()

    def on_validation_epoch_start(self):
        self.val_evaluator.reset()
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        img = batch['img']
        mask = batch['gt_semantic_seg']
        
        # Student forward pass
        student_logits = self.forward(img)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(img)
        
        # Compute KD loss
        loss = self.loss(student_logits, mask, teacher_logits)
        
        # Get predictions for metrics
        student_pred = torch.argmax(student_logits, dim=1)
        
        self.validation_step_outputs.append({
            'loss': loss,
            'pred': student_pred.cpu().numpy(),
            'target': mask.cpu().numpy()
        })
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute metrics
        for output in self.validation_step_outputs:
            self.val_evaluator.add_batch(output['target'], output['pred'])
        
        IoU_per_class = self.val_evaluator.Intersection_over_Union()
        mIoU = np.nanmean(IoU_per_class)
        F1_per_class = self.val_evaluator.F1()
        F1 = np.nanmean(F1_per_class)
        OA = self.val_evaluator.OA()
        
        # Log metrics with on_epoch=True to ensure they're saved
        self.log('val_mIoU', mIoU, on_epoch=True, prog_bar=True)
        self.log('val_F1', F1, on_epoch=True, prog_bar=True)
        self.log('val_OA', OA, on_epoch=True, prog_bar=True)
        
        # Print detailed metrics
        print(f"\nval: {{'mIoU': {mIoU}, 'F1': {F1}, 'OA': {OA}}}")
        class_metrics = {self.classes[i]: IoU_per_class[i] for i in range(len(self.classes))}
        print(class_metrics)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    
    seed_everything(42)
    
    # Create model
    model = KD_Train(config)
    
    # Load pretrained weights if specified (for fine-tuning)
    if hasattr(config, 'pretrained_ckpt_path') and config.pretrained_ckpt_path:
        print(f"Loading pretrained weights from: {config.pretrained_ckpt_path}")
        checkpoint = torch.load(config.pretrained_ckpt_path, map_location='cpu')
        # Load only the student model weights
        if 'state_dict' in checkpoint:
            # Filter out non-model keys
            model_state = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('net.')}
            model.net.load_state_dict(model_state)
            print("Loaded student model weights successfully!")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=config.weights_name,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=config.save_last
    )
    
    # Setup loggers
    csv_logger = CSVLogger(
        save_dir='lightning_logs',
        name=config.log_name
    )
    
    tb_logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=config.log_name
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epoch,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tb_logger],  # Use both loggers
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,
    )
    
    # Train
    if config.resume_ckpt_path:
        trainer.fit(model, config.train_loader, config.val_loader, ckpt_path=config.resume_ckpt_path)
    else:
        trainer.fit(model, config.train_loader, config.val_loader)


if __name__ == "__main__":
    main()
