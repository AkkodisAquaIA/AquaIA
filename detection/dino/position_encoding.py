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
"""

import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
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
        H, W = patch_x, patch_x
        assert H == W, "Input must be square"

        # Coordinate grids
        y_embed = torch.arange(H, device=device).float() + 0.5
        x_embed = torch.arange(W, device=device).float() + 0.5

        if self.normalize:
            y_embed = y_embed / H * self.scale
            x_embed = x_embed / W * self.scale

        # Shape: (H, W)
        y_embed = y_embed[:, None].expand(H, W)
        x_embed = x_embed[None, :].expand(H, W)

        # Frequencies
        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Shape: (H, W, num_pos_feats)
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = x_embed[:, :, None] / dim_t

        # sin / cos
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

        # Concatenate and reshape
        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, 2*num_pos_feats)
        pos = pos.flatten(end_dim=1)  # (H*W, 2*num_pos_feats)
        return pos


def build_position_encoding(pe_type, h_dim=256):
    N_steps = h_dim // 2
    if pe_type == "sine":  # also called v2
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif pe_type == "learned":  # also called v3
        position_embedding = PositionEmbeddingLearned(N_steps)  # noqa: F821
    elif pe_type == "sine_unnorm":  # also called v4
        position_embedding = PositionEmbeddingSine(N_steps, normalize=False)
    else:
        raise ValueError(f"not supported {pe_type}")
    # position_embedding = nn.ModuleList([position_embedding for _ in range(args.num_feature_levels)])
    return position_embedding
