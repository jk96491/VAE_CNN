import torch
import torch.nn as nn
import torch.utils.data
from VAE.Blocks import residual_down_sampling_block
from VAE.Blocks import residual_up_sampling_block


class Encoder(nn.Module):
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = residual_down_sampling_block(channels, ch)  # 64
        self.conv2 = residual_down_sampling_block(ch, 2 * ch)  # 32
        self.conv3 = residual_down_sampling_block(2 * ch, 4 * ch)  # 16
        self.conv4 = residual_down_sampling_block(4 * ch, 8 * ch)  # 8
        self.conv5 = residual_down_sampling_block(8 * ch, 8 * ch)  # 4
        self.conv_mu = nn.Conv2d(8 * ch, z, 2, 2)  # 2
        self.conv_logvar = nn.Conv2d(8 * ch, z, 2, 2)  # 2

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, Train=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if Train:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            x = self.sample(mu, logvar)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        return x, mu, logvar


class Decoder(nn.Module):
    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = residual_up_sampling_block(z, ch * 8)
        self.conv2 = residual_up_sampling_block(ch * 8, ch * 8)
        self.conv3 = residual_up_sampling_block(ch * 8, ch * 4)
        self.conv4 = residual_up_sampling_block(ch * 4, ch * 2)
        self.conv5 = residual_up_sampling_block(ch * 2, ch)
        self.conv6 = residual_up_sampling_block(ch, ch // 2)
        self.conv7 = nn.Conv2d(ch // 2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x