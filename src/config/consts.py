import os

train_data_dir = "../data/external/wood_dataset/train"
val_data_dir = "../data/external/wood_dataset/val"
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "models/ConvNets")