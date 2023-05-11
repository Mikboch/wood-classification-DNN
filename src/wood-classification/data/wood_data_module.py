import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import gdown
import zipfile


class WoodDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = '/content/wood-classification-DNN/data/external', raw_data_dir='/content/wood-classification-DNN/data/raw',
                 train_dataset_path='/content/wood-classification-DNN/data/external/wood_dataset/train',
                 test_dataset_path='/content/wood-classification-DNN/data/external/wood_dataset/val', image_size=(228, 228), num_classes=12,
                 split_ratio=0.9, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.batch_size = batch_size
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.imagenet_transform = Compose(
            [Resize(self.image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def prepare_data(self):
        current_path = os.getcwd()
        os.chdir(self.raw_data_dir)
        zip_name = 'wood_dataset.zip'
        gdown.download('https://drive.google.com/uc?id=1lbYAc5fUoKX06hktIghdma6LOyxpFFg8&confirm=t', output=zip_name,
                       quiet=False)
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        os.chdir(current_path)

    def setup(self, stage=None):
        # train/val
        if stage == 'fit' or stage is None:
            dataset = ImageFolder(self.train_dataset_path, transform=self.imagenet_transform)
            train_dataset_size = int(len(dataset) * self.split_ratio)
            self.train_dataset, self.val_dataset = random_split(dataset,
                                                                [train_dataset_size, len(dataset) - train_dataset_size])
        # test
        if stage == 'test' or stage is None:
            self.test_dataset = ImageFolder(self.test_dataset_path, transform=self.imagenet_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
