import os
import zipfile

import gdown
import torch
import torchvision
from torchvision import datasets


class WoodTypesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_type, data_transforms=None):
        self.dataset_type = dataset_type
        print("Current working directory: {0}".format(os.getcwd()))
        self.data = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
        self.labels = self.data.targets
        self.transforms = data_transforms
        print(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.data.samples[index]
        file_name = sample[0]
        label = sample[1]
        image = torchvision.io.read_image(file_name)  # torch.ops.image.read_file(file_name)
        if self.transforms:
            image = self.transforms(image)
        return image, label
