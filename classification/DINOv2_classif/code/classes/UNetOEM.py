# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:57:46 2024

@author: Florent.BRONDOLO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetOpenEarthMap(nn.Module):
    """
    UNetOpenEarthMap
    ----------
    This is a self-contained implementation of the UNet architecture (adapted from:
    https://discuss.pytorch.org/t/unet-implementation/426) following the SCHISM plug‑in guidelines.
    All building blocks are defined as callable functions (or inner classes) within this UNet class.
    
    REQUIRED_PARAMS:
      - num_classes (int): Number of segmentation classes.
    
    OPTIONAL_PARAMS:
      - channels (int): Number of input channels (default: 3).
      - n_block (int): Depth of the UNet (default: 5).
      - wf (int): Number of filters in the first layer is 2**wf (default: 6).
      - padding (bool): If True, apply padding such that input and output sizes match (default: True).
      - batch_norm (bool): Use BatchNorm after convolutional layers (default: False).
      - up_mode (str): 'upconv' (learned upsampling) or 'upsample' (bilinear) (default: "upconv").
    """
    REQUIRED_PARAMS = {
        'num_classes': int,
    }
    OPTIONAL_PARAMS = {
        'channels': int,
        'n_block': int,
        'wf': int,
        'padding': bool,
        'batch_norm': bool,
        'up_mode': str,
    }
    
    def __init__(self, num_classes, channels=3, n_block=5, wf=6, padding=True, batch_norm=False, up_mode="upconv"):
        super(UNetOpenEarthMap, self).__init__()
        assert up_mode in ("upconv", "upsample"), f"up_mode must be 'upconv' or 'upsample', got {up_mode}"
        self.n_block = n_block
        self.out_channels = num_classes
        self.up_mode = up_mode

        # Build the encoder (downsampling path)
        self.down_path = nn.ModuleList()
        current_in = channels
        for i in range(self.n_block):
            current_out = 2 ** (wf + i)
            self.down_path.append(self.unet_conv_block(current_in, current_out, padding, batch_norm))
            current_in = current_out

        # Build the decoder (upsampling path)
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.n_block - 1)):
            current_out = 2 ** (wf + i)
            self.up_path.append(self.unet_up_block(current_in, current_out, up_mode, padding, batch_norm))
            current_in = current_out

        # Final 1x1 convolution to produce the segmentation map
        self.last = nn.Conv2d(current_in, self.out_channels, kernel_size=1)
        self.name = "unet"

    def forward(self, x):
        blocks = []
        # Encoder: pass through each down block, saving features for skip connections.
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        # Decoder: use saved features in reverse order.
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        x = self.last(x)
        return x

    @staticmethod
    def unet_conv_block(in_channels, out_channels, padding, batch_norm):
        """
        Returns a sequential block of two convolutional layers (with ReLU and optional BatchNorm).
        """
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def unet_up_block(self, in_channels, out_channels, up_mode, padding, batch_norm):
        """
        Returns an upsampling block that first upsamples the input, concatenates it with the
        corresponding encoder feature (without cropping), and then applies a convolutional block.
        """
        # Define an inner module for the up block
        class UpBlock(nn.Module):
            def __init__(self, up_layer, conv_block):
                super(UpBlock, self).__init__()
                self.up = up_layer
                self.conv_block = conv_block

            def forward(self, x, bridge):
                up_out = self.up(x)
                out = torch.cat([up_out, bridge], dim=1)
                out = self.conv_block(out)
                return out

        # Choose upsampling method
        if up_mode == "upconv":
            up_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            up_layer = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            raise ValueError(f"Unsupported up_mode: {up_mode}")
        
        # The conv block receives concatenated features: (up_out + skip connection) = 2 * out_channels
        conv_block = self.unet_conv_block(out_channels * 2, out_channels, padding, batch_norm)
        return UpBlock(up_layer, conv_block)
