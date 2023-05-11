# standardowe pakiety
import os
import numpy as np

# Pytorch 
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
from torchvision.datasets import ImageFolder

# Pytorch Lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, Timer
import torchmetrics

# Hydra
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    early_stop_callback = EarlyStopping(
        monitor=cfg.early_stop_callback.monitor,
        patience=cfg.early_stop_callback.patience,
        verbose=cfg.early_stop_callback.verbose,
        mode=cfg.early_stop_callback.mode
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint_callback.monitor,
        dirpath=cfg.checkpoint_callback.dirpath,
        filename=cfg.checkpoint_callback.filename,
        save_top_k=cfg.checkpoint_callback.save_top_k,
        mode=cfg.checkpoint_callback.mode)

    model_summary_callback = ModelSummary(max_depth=cfg.model_summary_callback.max_depth)

    timer = Timer(duration=cfg.timer.duration)

    # Inicjalizacja modelu
    lightning_module = instantiate(cfg.modules)
    # Inicjalizacja trenera
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus,
                         callbacks=[early_stop_callback, checkpoint_callback, model_summary_callback, timer])
    # Trenowanie modelu
    trainer.fit(lightning_module, data_module)
    # Ewaluacja modelu
    trainer.test(model=lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()