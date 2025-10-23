"""
PyTorch implementation of the 3D Improved UNet (Isensee et al., 2018).
This version replicates the TensorFlow architecture used in the COMP3710 reference project.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------
#  Building Blocks
# -----------------------------------

class ContextModule(nn.Module):
    """Residual context block with dropout and instance norm"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.drop = nn.Dropout3d(dropout)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        self.residual = (in_channels == out_channels)

    def forward(self, x):
        residual = x if self.residual else None
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        if residual is not None:
            x = x + residual
        return x


class DownBlock(nn.Module):
    """Downsampling path with residual context"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.context = ContextModule(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        context = self.context(x)
        return x + context

class LocalizationBlock(nn.Module):
    """Localisation module: 3x3x3 + 1x1x1 convs"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 1)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
    


