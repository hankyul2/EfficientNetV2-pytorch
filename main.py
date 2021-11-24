import os
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch import nn
from torch.optim import SGD
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class, LightningCLI
from torchmetrics import MetricCollection, Accuracy

from utils.cifar import CIFAR
from efficientnetv2.efficientnet_v2 import get_efficientnet_v2
from utils.lr_scheduler import CosineLR
from utils.data_augmentation import cutmix, cutout, mixup


class BaseVisionSystem(LightningModule):
    def __init__(self, model_name: str, model_args: dict, num_classes: int, num_step: int, gpus: str, max_epochs: int,
                 optimizer_init: dict, lr_scheduler_init: dict, augmentation: str):
        """ Define base vision classification system
        :arg
            backbone_init: feature extractor
            num_classes: number of class of dataset
            num_step: number of step
            gpus: gpus id
            max_epoch: max number of epoch
            optimizer_init: optimizer class path and init args
            lr_scheduler_init: learning rate scheduler class path and init args
        """
        super(BaseVisionSystem, self).__init__()

        # step 1. save data related info (not defined here)
        self.augmentation = augmentation
        self.num_step = num_step // (len(gpus.split(','))-1)
        self.max_epochs = max_epochs

        # step 2. define model
        self.backbone = get_efficientnet_v2(model_name, **model_args)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = nn.CrossEntropyLoss()

        # step 4. define metric
        metrics = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, x):
        return self.fc(self.backbone(x))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(batch, self.train_metric, 'train', add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.valid_metric, 'valid', add_dataloader_idx=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.test_metric, 'test', add_dataloader_idx=True)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        x, y = batch
        loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
        metric = metric(y_hat, y)
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx)
        self.log_dict(metric, add_dataloader_idx=add_dataloader_idx)
        return loss

    def compute_loss(self, x, y):
        if self.augmentation == 'default':
            return self.compute_loss_eval(x, y)

        elif self.augmentation == 'mixup':
            x, y1, y2, ratio = mixup(x, y)
            y_hat = self.fc(self.backbone(x))
            loss = self.criterion(y_hat, y1) * ratio + self.criterion(y_hat, y2) * (1 - ratio)
            return loss, y_hat

        elif self.augmentation == 'cutout':
            x, y, ratio = cutout(x, y)
            return self.compute_loss_eval(x, y)

        elif self.augmentation == 'cutmix':
            x, y1, y2, ratio = cutmix(x, y)
            y_hat = self.fc(self.backbone(x))
            loss = self.criterion(y_hat, y1) * ratio + self.criterion(y_hat, y2) * (1 - ratio)
            return loss, y_hat

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.backbone(x))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_and_get_lr_scheduler_config(self):
        if 'num_step' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['num_step'] = self.num_step
        if 'max_epochs' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['max_epochs'] = self.max_epochs
        return self.lr_scheduler_init_config


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 1. link argument
        parser.link_arguments('data.num_classes', 'model.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.max_epochs', apply_on='parse')
        parser.link_arguments('trainer.gpus', 'model.gpus', apply_on='parse')

        # 2. add optimizer & scheduler argument
        parser.add_optimizer_args((SGD,), link_to='model.optimizer_init')
        parser.add_lr_scheduler_args((CosineLR,), link_to='model.lr_scheduler_init')


if __name__ == '__main__':
    cli = MyLightningCLI(BaseVisionSystem, CIFAR, save_config_overwrite=True)
    cli.trainer.test(ckpt_path='best')