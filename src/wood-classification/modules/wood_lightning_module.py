# Pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import pytorch_lightning as pl
import torchmetrics


class WoodLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=2 * 1e-4, use_pretrained=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.model = self.pretrained_model()
        else:
            self.model = model

    def pretrained_model(self):
        net_pretrained = models.resnet34(pretrained=True)
        # zamrożenie parametrów sieci
        for param in net_pretrained.parameters():
            param.requires_grad = False
        num_in_features = net_pretrained.fc.in_features  # liczba cech wejściowych się nie zmienia, natomiast liczbę cech wyjściowych podmienimy na num_classes = 2
        net_pretrained.fc = nn.Linear(num_in_features,
                                      self.num_classes)  # nadpisanie warstwy fc nową warstwą - w tym przykładzie tylko ta byłaby trenowana
        return net_pretrained

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
