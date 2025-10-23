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
    

class UpBlock(nn.Module):
    """Upsampling block followed by concatenation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.cat([x, skip], dim=1)
        return x


class SegmentationLayer(nn.Module):
    """1x1x1 conv with optional upscaling and addition"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x, prev_seg=None, upscale=False, add=False):
        seg = self.conv(x)
        if add and prev_seg is not None:
            # Upsample previous segmentation first, then add
            prev_up = F.interpolate(prev_seg, size=seg.shape[2:], mode='trilinear', align_corners=False)
            seg = seg + prev_up
        if upscale:
            seg = F.interpolate(seg, scale_factor=2, mode='trilinear', align_corners=False)
        return seg


# -----------------------------------
#  Full Model
# -----------------------------------

class ImprovedUNet3D(nn.Module):
    def __init__(self, num_classes=6, base_filters=16):
        super().__init__()
        # Encoder (Down)
        self.down1 = DownBlock(1, base_filters, stride=1)
        self.down2 = DownBlock(base_filters, base_filters*2, stride=2)
        self.down3 = DownBlock(base_filters*2, base_filters*4, stride=2)
        self.down4 = DownBlock(base_filters*4, base_filters*8, stride=2)
        self.down5 = DownBlock(base_filters*8, base_filters*16, stride=2)

        # Decoder (Up)
        self.up4 = UpBlock(base_filters*16, base_filters*8)
        self.loc4 = LocalizationBlock(base_filters*8 + base_filters*8, base_filters*8)

        self.up3 = UpBlock(base_filters*8, base_filters*4)
        self.loc3 = LocalizationBlock(base_filters*4 + base_filters*4, base_filters*4)

        self.seg3 = SegmentationLayer(base_filters*4, num_classes)

        self.up2 = UpBlock(base_filters*4, base_filters*2)
        self.loc2 = LocalizationBlock(base_filters*2 + base_filters*2, base_filters*2)
        self.seg2 = SegmentationLayer(base_filters*2, num_classes)

        self.up1 = UpBlock(base_filters*2, base_filters)
        self.loc1 = nn.Conv3d(base_filters + base_filters, base_filters*2, 3, padding=1)
        self.seg1 = SegmentationLayer(base_filters*2, num_classes)

        self.out_conv = nn.Conv3d(num_classes, num_classes, 1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        c5 = self.down5(c4)

        # Decoder
        u4 = self.up4(c5, c4)
        l4 = self.loc4(u4)
        u3 = self.up3(l4, c3)
        l3 = self.loc3(u3)

        seg3 = self.seg3(l3)

        u2 = self.up2(l3, c2)
        l2 = self.loc2(u2)
        seg2 = self.seg2(l2, seg3, upscale=True, add=True)

        u1 = self.up1(l2, c1)
        l1 = self.loc1(u1)
        seg1 = self.seg1(l1, seg2, upscale=False, add=True)

        out = self.out_conv(seg1)
        return out


# -----------------------------------
#  Test Model
# -----------------------------------
if __name__ == "__main__":
    model = ImprovedUNet3D(num_classes=6)
    x = torch.randn(1, 1, 64, 64, 32)  # Example input
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")