from torch.utils import data
from torchvision import transforms

from src.config.consts import *
from src.data.models.wood_type_dataset import WoodTypesDataset


def get_data_loaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(228),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(200),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    train_set = WoodTypesDataset(data_dir="../data/external/wood_dataset/train", dataset_type="train",
                                 data_transforms=data_transforms["train"])
    val_set = WoodTypesDataset(data_dir="../data/external/wood_dataset/val", dataset_type="val",
                               data_transforms=data_transforms["val"])
    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
    return train_loader, val_loader
