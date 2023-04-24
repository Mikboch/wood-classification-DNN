import torch.nn as nn

from src.models.net.w_imcnn_block import WIMCNNBlock


# Warstwa debugowa do wyświetlania rozmiarów warstwa po warstwie w Sequential
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


# TODO: Ogarnąć:
# 1) Stride i paddingi, czy nie da się tego fajniej wykonać
# (jeżeli nie, to trzeba to dobrze uzaadnić, a na razie nie mam pomysłu)
# (nie wiem jak w artykule uzyskiwali takie rozmiary obrazów z kroku na krok)
# 2) Przekazywanie parametrów
class WIMCNNModel(nn.Module):
    def __init__(self, num_classes=12, in_channels=3, num_blocks=[3, 3, 3], c_hidden=[16, 32, 64], act_fn_name="relu",
                 block_name="ResNetBlock", **kwargs):
        super().__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
            PrintLayer(),  # debug
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            PrintLayer(),  # debug
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            PrintLayer(),  # debug
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(3, stride=2),
            PrintLayer(),  # debug
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            PrintLayer(),  # debug
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            PrintLayer(),  # debug
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            PrintLayer()  # debug
        )
        self.WIMCNNUnit = nn.Sequential(
            WIMCNNBlock(256, 64, (96, 128), (16, 24, 32), 32),
            WIMCNNBlock(480, 128, (128, 192), (32, 64, 96), 64),
            WIMCNNBlock(512, 192, (96, 208), (16, 32, 48), 64),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(192, 512, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True)
        )
        self.AvgPool = nn.AvgPool2d(13, ceil_mode=True)
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x):
        print("x = ", x.shape)
        out = self.convBlock1(x)
        print("out = ", out.shape)
        out = self.convBlock2(out)
        print("out = ", out.shape)
        WIMCNNUnitOut = self.WIMCNNUnit(out)
        print("WIMCNNUnitOut = ", WIMCNNUnitOut.shape)
        out_conv = self.out_conv(out)
        out = out_conv + WIMCNNUnitOut
        print("out = ", out.shape)
        out = self.AvgPool(out)
        print("out = ", out.shape)
        result = self.linear(out)

        return result
