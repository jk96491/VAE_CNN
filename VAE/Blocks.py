import torch.nn as nn
import torch.nn.functional as F


class residual_down_sampling_block(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(residual_down_sampling_block, self).__init__()

        self.conv1_layer = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BatchNorm_Layer1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2_layer = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BatchNorm_Layer2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = F.rrelu(self.BatchNorm_Layer1(self.conv1_layer(x)))
        x = self.AvePool(x)
        x = self.BatchNorm_Layer2(self.conv2_layer(x))

        x = F.rrelu(x + skip)
        return x


class residual_up_sampling_block(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(residual_up_sampling_block, self).__init__()

        self.conv1_layer = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BatchNorm_Layer1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2_layer = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BatchNorm_Layer2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.UpNN = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))

        x = F.rrelu(self.BatchNorm_Layer1(self.conv1_layer(x)))
        x = self.UpNN(x)
        x = self.BatchNorm_Layer2(self.conv2_layer(x))

        x = F.rrelu(x + skip)
        return x