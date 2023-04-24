import torch
import torch.nn as nn


class WIMCNNBlock(nn.Module):
    def __init__(self, out_channels, c1, c2, c3, c4):
        super().__init__()
        self.residual = nn.Sequential(
            # Conv1x1
            nn.LazyConv2d(out_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True)
        )
        self.conv1x1 = nn.Sequential(
            # Conv1x1
            nn.LazyConv2d(c1, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            # Conv1x1
            nn.LazyConv2d(c2[0], kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            # ConvDW3x3
            nn.Conv2d(c2[0], c2[0], kernel_size=3, padding=1, groups=c2[0]),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU6(inplace=True),
            # Conv1x1
            nn.LazyConv2d(c2[1], kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.conv5x5 = nn.Sequential(
            # Conv1x1
            nn.LazyConv2d(c3[0], kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            # ConvDW3x3
            nn.Conv2d(c3[0], c3[0], kernel_size=3, padding=1, groups=c3[0]),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU6(inplace=True),
            # Conv1x1
            nn.LazyConv2d(c3[1], kernel_size=1),
            nn.ReLU6(inplace=True),
            # ConvDW3x3
            nn.Conv2d(c3[1], c3[1], kernel_size=3, padding=1, groups=c3[1]),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU6(inplace=True),
            # Conv1x1
            nn.LazyConv2d(c3[2], kernel_size=1),
            nn.ReLU6(inplace=True)
        )
        self.maxPool = nn.Sequential(
            nn.MaxPool2d(1),
            # Conv1x1
            nn.LazyConv2d(c4, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        residual = self.residual(x)
        branch_1 = self.conv1x1(x)
        branch_2 = self.conv3x3(x)
        branch_3 = self.conv5x5(x)
        branch_4 = self.maxPool(x)
        concat = torch.cat((branch_1, branch_2, branch_3, branch_4), 1)

        return residual + concat
