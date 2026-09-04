# Z: https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/detection/models/position_encoding.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
Z: The position of each grid is encoded by two coordinates, x_embed/y_embed.
The total positional encoding has 2*num_pos_feats=d_model dimensions, where x/y each occupy num_pos_feats dimensions.
Inside each axis, it is composed of num_pos_feats/2 sin(coord/dim_t) of different frequencies
and num_pos_feats/2 cos(coord/dim_t) alternating.
"""

import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Z: For each element in feature map, generate a PE of 1 x 1 x (2 * num_pos_feats)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # Z: num_pos_feats = number of positional features (sin or cos) for each axis (x/y)
        # Z: 1 feature is a value of sin or cos. num_pos_feats = d_model // 2 (x/y)
        self.num_pos_feats = num_pos_feats
        # Z: temperature controls the scale of different frequencies of sin or cos
        self.temperature = temperature
        # Z: normalize applies to coords, not to sin or cos outputs
        self.normalize = normalize
        # Z: scale maps normalized coords to a range before applying sin or cos
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, patch_x, device):
        """
        x: (B, C, H, H)
        returns: (B, 2*num_pos_feats, H, H)
        """
        # Z: patch_x is the grid size of ViT after patching
        H, W = patch_x, patch_x
        assert H == W, "Input must be square"

        # Coordinate grids
        # Z: obtain the center coords of each patch in the feature map
        y_embed = torch.arange(H, device=device).float() + 0.5
        x_embed = torch.arange(W, device=device).float() + 0.5

        # Z: normalize the coords to [0, scale] to prevent image size issues
        if self.normalize:
            y_embed = y_embed / H * self.scale
            x_embed = x_embed / W * self.scale

        # Shape: (H, W)
        # Z: expand via broadcast the coords vectors to create a 2D grid of shape (H, W)
        # Z: cause feature map is 2D, each element needs a x coord and a y coord
        y_embed = y_embed[:, None].expand(H, W)
        x_embed = x_embed[None, :].expand(H, W)

        # Frequencies
        # Z: Ex: tensor([0., 1., 2., 3., 4., 5., 6., 7.])
        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        # Z: Ex: tensor([1., 1., 10., 10., 100., 100., 1000., 1000.])
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Shape: (H, W, num_pos_feats)
        # Z: for each position (x, y) generate a vector of num_pos_feats values
        # Z: by dividing the each coords by a different dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = x_embed[:, :, None] / dim_t

        # sin / cos
        # Z: apply sin to even indices and cos to odd indices of the pos_y and pos_x vectors
        # Z: done along the dimension of num_pos_feats, final shape (H, W, num_pos_feats)
        # Z: flatten(-2) = flatten the last two dimensions (sin and cos) into one dimension
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

        # Concatenate and reshape
        # Z: concatenate by the last dimension
        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, 2*num_pos_feats)
        # Z: flatten to feed into transformer
        pos = pos.flatten(end_dim=1)  # (H*W, 2*num_pos_feats)
        return pos


def build_position_encoding(pe_type, h_dim=256):
    N_steps = h_dim // 2
    if pe_type == "sine":  # also called v2
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif pe_type == "learned":  # also called v3
        # Z: !Warnign! leared PE is not implemented
        position_embedding = PositionEmbeddingLearned(N_steps)  # noqa: F821
    elif pe_type == "sine_unnorm":  # also called v4
        position_embedding = PositionEmbeddingSine(N_steps, normalize=False)
    else:
        raise ValueError(f"not supported {pe_type}")
    # position_embedding = nn.ModuleList([position_embedding for _ in range(args.num_feature_levels)])
    return position_embedding
