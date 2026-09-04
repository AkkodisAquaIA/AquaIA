# Z: https://github.com/facebookresearch/detr/blob/main/models/transformer.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

Samuel Beaussant : Taken from DETR official repo. Modified and simplified for the current project:
    * Removed Type Hints for less verbosity (mostly tensors anyway)
    * Only supports post normalization
        Z: aka no pre-norm = x + Sublayer(Norm(x))
    * Removed support for masks and padding masks for simplicity (operate on square images)
        Z: remove masks (attention mask for local attention) and padding masks (for padding small images to square)
    * Added batch first, need_weight=False and fp16 for flash attention support
        Z: FlashAttention speeds up attention, requires batch first and no need for attention weights
    * Removed useless copies and permute for better efficiency
    * Default dropout is 0 (plain detr removed it completely)
"""

import copy
import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.0, activation="relu", return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Z: initialize parameters, no for biases."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        # Z: query_embed = object queries
        # Z: pos_embed = positional encoding

        bs = src.shape[0]
        # Z: [num_queries, d_model] -> [batch_size, num_queries, d_model]
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # no copy
        # Z: tgt = all zeros with same shape as query_embed
        tgt = torch.zeros_like(query_embed)
        # Z: memory = output of encoder
        memory = self.encoder(src, pos=pos_embed)
        # Z: hs = hidden state = output of decoder, query_pos = object queries
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        return hs


class TransformerEncoder(nn.Module):
    """Z: stacks encoder layer N times, each with its own parameters."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)

        # Z: [batch_size, seq_len, d_model]
        # Z: seq_len number of spatial locations
        # Z: after flattening the backbone feature map such as H * W
        return output


class TransformerDecoder(nn.Module):
    """Z: stacks decoder layer N times, each with its own parameters.
    Optionally returns intermediate outputs from each layer."""

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            # Z: store output of each layer
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            # Z: [num_layers, batch_size, num_queries, d_model]
            return torch.stack(intermediate)

        # Z: [1, batch_size, num_queries, d_model]
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    # Z: d_model = embedding dimension
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Z: 2 norms have 2 param sets, 2 dropouts for clarity
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos=None):
        # Z: src = featrure map from previous layer
        # Z: pos = positional encoding
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos=None):
        return self.forward_post(src, pos=pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, pos=None, query_pos=None):
        # Z: tgt = target sequence, starts as zero and is updated at each layer
        # Z: memory = output of encoder
        # Z: pos = positional encoding
        # Z: query_pos = object queries, like anchors in feature space
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, pos=None, query_pos=None):
        return self.forward_post(tgt, memory, pos, query_pos)


def _get_clones(module, N):
    """Z: clone a module N times, each indenpendent with its own parameters"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
