from w_imcnn_block import WIMCNNBlock
from torch import nn


class WIMCNNModel(nn.Module):
    def __init__(self, num_classes=16, in_channels=3):
        super().__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.WIMCNNUnit = nn.Sequential(
            WIMCNNBlock(256, 64, (96, 128), (16, 24, 32), 32),
            WIMCNNBlock(480, 128, (128, 192), (32, 64, 96), 64),
            WIMCNNBlock(512, 192, (96, 208), (16, 32, 48), 64),
        )
        self.AvgPool = nn.AvgPool2d(13)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.convBlock1(x)
        out = self.convBlock2(out)
        WIMCNNUnitOut = self.WIMCNNUnit(out)
        out = out + WIMCNNUnitOut
        out = self.AvgPool(out)
        result = self.linear(out)

        return result
