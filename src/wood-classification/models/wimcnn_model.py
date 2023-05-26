import torch
from torch import nn
from torch.nn import functional as F


class WIMCNNModel(nn.Module):
    def __init__(self, num_classes=12, in_channels=3,
                 channels_before_wimcnn_unit=64,
                 wimcnn_unit_in_channels=192,
                 wimcnn_first_block_out_channels=256,
                 wimcnn_second_block_out_channels=480,
                 out_channels=512, **kwargs):
        super().__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_before_wimcnn_unit, kernel_size=3, stride=2),
            nn.ReLU6(),

            nn.Conv2d(channels_before_wimcnn_unit, channels_before_wimcnn_unit, kernel_size=3, stride=2),
            nn.ReLU6(),

            nn.Conv2d(channels_before_wimcnn_unit, channels_before_wimcnn_unit, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_before_wimcnn_unit),
            nn.ReLU6(),

            nn.MaxPool2d(3, stride=2),
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(channels_before_wimcnn_unit, channels_before_wimcnn_unit, kernel_size=1),
            nn.ReLU6(),

            nn.Conv2d(channels_before_wimcnn_unit, wimcnn_unit_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(wimcnn_unit_in_channels),
            nn.ReLU6(),

            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )
        # Parameters for each block based on GoogLeNet and "Going Deeper with Convolutions" work (Szegedy et al., 2015)
        self.WIMCNNUnit = nn.Sequential(
            # By default: (192, 256, 64, (96, 128), (16, 24, 32), 32); ratio 2:4:1:1
            WIMCNNBlock(wimcnn_unit_in_channels, wimcnn_first_block_out_channels,
                        int(wimcnn_first_block_out_channels / 4),
                        (int(wimcnn_first_block_out_channels * 3 / 8), int(wimcnn_first_block_out_channels / 2)),
                        (int(wimcnn_first_block_out_channels / 16), int(wimcnn_first_block_out_channels * 3 / 32),
                         int(wimcnn_first_block_out_channels / 8)),
                        int(wimcnn_first_block_out_channels / 8)),
            nn.ReLU6(),
            # By default: (256, 480, 128, (128, 192), (32, 64, 96), 64); ratio 4:6:3:2
            WIMCNNBlock(wimcnn_first_block_out_channels, wimcnn_second_block_out_channels,
                        int(wimcnn_second_block_out_channels * 4 / 15),
                        (int(wimcnn_second_block_out_channels * 4 / 15), int(wimcnn_second_block_out_channels * 2 / 5)),
                        (int(wimcnn_second_block_out_channels / 15), int(wimcnn_second_block_out_channels * 2 / 15),
                         int(wimcnn_second_block_out_channels / 5)),
                        int(wimcnn_second_block_out_channels * 2 / 15)),
            nn.ReLU6(),
            # By default: (480, 512, 128, (128, 256), (24, 48, 64), 64); ratio 2:4:1:1
            WIMCNNBlock(wimcnn_second_block_out_channels, out_channels,
                        int(out_channels / 4),
                        (int(out_channels / 4), int(out_channels / 2)),
                        (int(out_channels / 32), int(out_channels / 16), int(out_channels / 8)),
                        int(out_channels / 8)),
        )

        self.residual_conv = nn.Conv2d(wimcnn_unit_in_channels, out_channels, kernel_size=1)
        self.AvgPool = nn.AvgPool2d(13, ceil_mode=True)
        self.linear = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        # First convolution block
        out = self.convBlock1(x)

        # Second convolution block
        out = self.convBlock2(out)

        # WIMCNN Unit Computation
        WIMCNNUnitOut = self.WIMCNNUnit(out)
        residual_conv = self.residual_conv(out)

        # Splice after unit computation
        out = residual_conv + WIMCNNUnitOut
        out = F.relu6(out)
        out = self.AvgPool(out)

        # Fully Connected
        out = out.view(out.size(0), -1)
        result = self.linear(out)

        return result


class WIMCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, c1, c2, c3, c4):
        super().__init__()
        self.residual = nn.Sequential(
            # Conv1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU6(),
        )
        self.conv1x1 = nn.Sequential(
            # Conv1x1
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU6(),
        )
        self.conv3x3 = nn.Sequential(
            # Conv1x1
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU6(),

            # ConvDW3x3
            nn.Conv2d(c2[0], c2[0], kernel_size=3, padding=1, groups=c2[0], bias=False),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU6(),

            # Conv1x1
            nn.Conv2d(c2[0], c2[1], kernel_size=1),
            nn.ReLU6(),
        )
        self.conv5x5 = nn.Sequential(
            # Conv1x1
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU6(),

            # ConvDW3x3
            nn.Conv2d(c3[0], c3[0], kernel_size=3, padding=1, groups=c3[0], bias=False),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU6(),

            # Conv1x1
            nn.Conv2d(c3[0], c3[1], kernel_size=1),
            nn.ReLU6(),

            # ConvDW3x3
            nn.Conv2d(c3[1], c3[1], kernel_size=3, padding=1, groups=c3[1], bias=False),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU6(),

            # Conv1x1
            nn.Conv2d(c3[1], c3[2], kernel_size=1),
            nn.ReLU6(),
        )
        self.maxPool = nn.Sequential(
            nn.MaxPool2d(1),

            # Conv1x1
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU6(),
        )

    def forward(self, x):
        residual = self.residual(x)
        branch_1 = self.conv1x1(x)
        branch_2 = self.conv3x3(x)
        branch_3 = self.conv5x5(x)
        branch_4 = self.maxPool(x)
        concat = torch.cat((branch_1, branch_2, branch_3, branch_4), dim=1)

        return residual + concat
