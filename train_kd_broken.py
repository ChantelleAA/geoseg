"""Knowledge Distillation Training Script."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
from tools.metrics_plotter import MetricsPlotterCallback
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random





def seed_everything(seed):def seed_everything(seed):

    random.seed(seed)    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)    np.random.seed(seed)

    torch.manual_seed(seed)    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True    torch.backends.cudnn.benchmark = True





def get_args():def get_args():

    parser = argparse.ArgumentParser()    parser = argparse.ArgumentParser()

    arg = parser.add_argument    arg = parser.add_argument

    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)

    return parser.parse_args()    return parser.parse_args()





class KD_Train(pl.LightningModule):class KD_Train(pl.LightningModule):

    def __init__(self, config):    def __init__(self, config):

        super().__init__()        super().__init__()

        self.config = config        self.config = config

        self.net = config.net  # student model        self.net = config.net  # student model

        self.teacher = config.teacher  # teacher model        self.teacher = config.teacher  # teacher model

                

        # Move teacher to same device as student will be        # Move teacher to same device as student will be

        # This happens automatically when we register it as a module        # This happens automatically when we register it as a module

        # but we explicitly freeze and set to eval        # but we explicitly freeze and set to eval

        self.teacher.eval()        self.teacher.eval()

        for param in self.teacher.parameters():        for param in self.teacher.parameters():

            param.requires_grad = False            param.requires_grad = False



        self.loss = config.loss        self.loss = config.loss

        self.kd_helper = config.kd_helper if hasattr(config, 'kd_helper') else None        self.kd_helper = config.kd_helper if hasattr(config, 'kd_helper') else None



        self.metrics_train = Evaluator(num_class=config.num_classes)        self.metrics_train = Evaluator(num_class=config.num_classes)

        self.metrics_val = Evaluator(num_class=config.num_classes)        self.metrics_val = Evaluator(num_class=config.num_classes)



    def forward(self, x):    def forward(self, x):

        # only student net is used in the prediction/inference        # only student net is used in the prediction/inference

        seg_pre = self.net(x)        seg_pre = self.net(x)

        return seg_pre        return seg_pre



    def training_step(self, batch, batch_idx):    def training_step(self, batch, batch_idx):

        img, mask = batch['img'], batch['gt_semantic_seg']        img, mask = batch['img'], batch['gt_semantic_seg']



        # Get student predictions        # Get student predictions

        student_prediction = self.net(img)        student_prediction = self.net(img)

                

        # Get teacher predictions (no gradient)        # Get teacher predictions (no gradient)

        # Ensure teacher is on same device as input        # Ensure teacher is on same device as input

        with torch.no_grad():        with torch.no_grad():

            # Move teacher to same device if needed            # Move teacher to same device if needed

            if next(self.teacher.parameters()).device != img.device:            if next(self.teacher.parameters()).device != img.device:

                self.teacher = self.teacher.to(img.device)                self.teacher = self.teacher.to(img.device)

            teacher_prediction = self.teacher(img)            teacher_prediction = self.teacher(img)

                

        # Compute KD loss        # Compute KD loss

        loss = self.loss(student_prediction, mask, teacher_prediction)        loss = self.loss(student_prediction, mask, teacher_prediction)



        # For metrics, use student predictions        # For metrics, use student predictions

        if self.config.use_aux_loss:        if self.config.use_aux_loss:

            pre_mask = nn.Softmax(dim=1)(student_prediction[0])            pre_mask = nn.Softmax(dim=1)(student_prediction[0])

        else:        else:

            pre_mask = nn.Softmax(dim=1)(student_prediction)            pre_mask = nn.Softmax(dim=1)(student_prediction)



        pre_mask = pre_mask.argmax(dim=1)        pre_mask = pre_mask.argmax(dim=1)

        for i in range(mask.shape[0]):        for i in range(mask.shape[0]):

            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())



        return {"loss": loss}        return {"loss": loss}



    def on_train_epoch_end(self):    def on_train_epoch_end(self):

        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())

        F1 = np.nanmean(self.metrics_train.F1())        F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())        OA = np.nanmean(self.metrics_train.OA())

        iou_per_class = self.metrics_train.Intersection_over_Union()        iou_per_class = self.metrics_train.Intersection_over_Union()

                

        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}

        print('train:', eval_value)        print('train:', eval_value)



        iou_value = {}        iou_value = {}

        for class_name, iou in zip(self.config.classes, iou_per_class):        for class_name, iou in zip(self.config.classes, iou_per_class):

            iou_value[class_name] = iou            iou_value[class_name] = iou

        print(iou_value)        print(iou_value)

                

        self.metrics_train.reset()        self.metrics_train.reset()

        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}

        self.log_dict(log_dict, prog_bar=True)        self.log_dict(log_dict, prog_bar=True)



    def validation_step(self, batch, batch_idx):    def validation_step(self, batch, batch_idx):

        img, mask = batch['img'], batch['gt_semantic_seg']        img, mask = batch['img'], batch['gt_semantic_seg']

                

        # Only use student for validation        # Only use student for validation

        prediction = self.forward(img)        prediction = self.forward(img)

        pre_mask = nn.Softmax(dim=1)(prediction)        pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)        pre_mask = pre_mask.argmax(dim=1)

                

        for i in range(mask.shape[0]):        for i in range(mask.shape[0]):

            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())



        # For validation loss, we can use just the hard target loss        # For validation loss, we can use just the hard target loss

        # or the full KD loss - let's use a simple loss for validation        # or the full KD loss - let's use a simple loss for validation

        with torch.no_grad():        with torch.no_grad():

            teacher_prediction = self.teacher(img)            teacher_prediction = self.teacher(img)

        loss_val = self.loss(prediction, mask, teacher_prediction)        loss_val = self.loss(prediction, mask, teacher_prediction)

                

        return {"loss_val": loss_val}        return {"loss_val": loss_val}



    def on_validation_epoch_end(self):    def on_validation_epoch_end(self):

        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())

        F1 = np.nanmean(self.metrics_val.F1())        F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())        OA = np.nanmean(self.metrics_val.OA())

        iou_per_class = self.metrics_val.Intersection_over_Union()        iou_per_class = self.metrics_val.Intersection_over_Union()



        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}

        print('val:', eval_value)        print('val:', eval_value)

                

        iou_value = {}        iou_value = {}

        for class_name, iou in zip(self.config.classes, iou_per_class):        for class_name, iou in zip(self.config.classes, iou_per_class):

            iou_value[class_name] = iou            iou_value[class_name] = iou

        print(iou_value)        print(iou_value)



        self.metrics_val.reset()        self.metrics_val.reset()

        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}

        self.log_dict(log_dict, prog_bar=True)        self.log_dict(log_dict, prog_bar=True)



    def configure_optimizers(self):    def configure_optimizers(self):

        optimizer = self.config.optimizer        optimizer = self.config.optimizer

        lr_scheduler = self.config.lr_scheduler        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]        return [optimizer], [lr_scheduler]



    def train_dataloader(self):    def train_dataloader(self):

        return self.config.train_loader        return self.config.train_loader



    def val_dataloader(self):    def val_dataloader(self):

        return self.config.val_loader        return self.config.val_loader





def main():def main():

    args = get_args()    args = get_args()

    config = py2cfg(args.config_path)    config = py2cfg(args.config_path)

    seed_everything(42)    seed_everything(42)



    checkpoint_callback = ModelCheckpoint(    checkpoint_callback = ModelCheckpoint(

        save_top_k=config.save_top_k,         save_top_k=config.save_top_k, 

        monitor=config.monitor,        monitor=config.monitor,

        save_last=config.save_last,         save_last=config.save_last, 

        mode=config.monitor_mode,        mode=config.monitor_mode,

        dirpath=config.weights_path,        dirpath=config.weights_path,

        filename=config.weights_name        filename=config.weights_name

    )    )

        

    # Create metrics plotter callback    # Create metrics plotter callback

    plot_dir = os.path.join('plots', config.log_name)    plot_dir = os.path.join('plots', config.log_name)

    metrics_plotter = MetricsPlotterCallback(save_dir=plot_dir, plot_every_n_epochs=1)    metrics_plotter = MetricsPlotterCallback(save_dir=plot_dir, plot_every_n_epochs=1)

        

    logger = CSVLogger('lightning_logs', name=config.log_name)    logger = CSVLogger('lightning_logs', name=config.log_name)



    model = KD_Train(config)    model = KD_Train(config)

        

    if config.pretrained_ckpt_path:    if config.pretrained_ckpt_path:

        model = KD_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)        model = KD_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)



    trainer = pl.Trainer(    trainer = pl.Trainer(

        devices=config.gpus,         devices=config.gpus, 

        max_epochs=config.max_epoch,         max_epochs=config.max_epoch, 

        accelerator='auto',        accelerator='auto',

        check_val_every_n_epoch=config.check_val_every_n_epoch,        check_val_every_n_epoch=config.check_val_every_n_epoch,

        callbacks=[checkpoint_callback, metrics_plotter],         callbacks=[checkpoint_callback, metrics_plotter], 

        strategy='auto',        strategy='auto',

        logger=logger        logger=logger

    )    )

        

    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)





if __name__ == "__main__":if __name__ == "__main__":

    main()    main()

