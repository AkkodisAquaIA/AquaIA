# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:57:46 2024

@author: Florent.BRONDOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetVanilla(nn.Module):
    REQUIRED_PARAMS = {
        'num_classes': int,
    }

    OPTIONAL_PARAMS = {
        'n_block': int,
        'channels': int,
        'k_size': int,  
        'activation': str 
    }

    def __init__(self, 
                 n_block=4, 
                 channels=8, 
                 num_classes=3, 
                 k_size=3, 
                 activation='relu'
                 ):
        super(UnetVanilla, self).__init__()
        self.n_block = n_block
        self.channels = channels
        self.num_classes = num_classes
        self.k_size = k_size
        self.activation = activation.lower()

        self.encoder_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        for i in range(n_block):
            self.encoder_blocks.append(self.encoder_block(channels * (2 ** i)))
            self.max_pools.append(nn.MaxPool2d(kernel_size=2, ceil_mode=True))

        # Define up sampling path
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_block, -1, -1):
            if i > 0:
                self.decoder_blocks.append(self.decoder_block(channels * (2 ** i)))
                self.up_convs.append(nn.ConvTranspose2d(
                    channels * (2 ** i), channels * (2 ** i) // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1  # Allow a small correction
                ))

        # Define output layer
        self.output_conv = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

        self.bridge = self.simple_conv(self.channels * (2 ** n_block) // 2, self.channels * (2 ** n_block))
        self.start = self.simple_conv(3, self.channels)

    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def simple_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, padding=self.k_size // 2),
            self._get_activation()
        )

    def encoder_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=self.k_size, padding=self.k_size // 2),
            nn.BatchNorm2d(in_channels * 2),
            self._get_activation()
        )

    def decoder_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=self.k_size, padding=self.k_size // 2),
            nn.BatchNorm2d(in_channels // 2),
            self._get_activation()
        )

    def forward(self, x):
        encodings = []
        x = self.start(x)
        encodings.append(x)

        for i in range(self.n_block):
            x = self.max_pools[i](x)
            if i < self.n_block - 1:
                x = self.encoder_blocks[i](x)
                encodings.append(x)

        x = self.bridge(x)

        for i in range(self.n_block):
            x = self.up_convs[i](x)
            enc = F.interpolate(encodings[-(i + 1)], size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
            x = torch.cat([x, enc], dim=1)
            x = self.decoder_blocks[i](x)

        x = self.output_conv(x)

        return x
