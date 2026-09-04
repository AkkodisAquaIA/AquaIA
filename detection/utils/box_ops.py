# Z: https://github.com/impiga/Plain-DETR/blob/main/util/box_ops.py
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
Utilities for bounding box manipulation and GIoU.
"""

import numpy as np
import torch
from torchvision.ops.boxes import box_area

# Z: In image referential, x increases to the right, y increases downwards.


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    """Z: default input xyxy, boxes1 can be multiple so as boxes2"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Z: left top coords of each box-pair intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # Z: right bottom coords of each box-pair intersection
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # Z: rb - lt = [inter_width, inter_height], clamp to 0 to prevent negatives
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # Z: calculate intersection = w*h
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Z: calculate union, [N,M]
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # Z: check that each box is valid: x1 >= x0 and y1 >= y0
    mask = (boxes1[:, 2:] >= boxes1[:, :2]).all(dim=1)
    # Z: if some bbox not valid
    if not mask.all():
        print("invalid boxes(x0y0x1y1)\n", flush=True)
        # Z: ~mask = opposite of mask
        print(boxes1[~mask], "\n", flush=True)
        print("invalid boxes(cxcywh)\n", flush=True)
        print(box_xyxy_to_cxcywh(boxes1[~mask]), "\n", flush=True)
        print("\n", flush=True)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # Z: left top coords of enclosing boxes, [N,M,2]
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # Z: right bottom coords of enclosing boxes, [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # Z: w and h of enclosing boxes, [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    # Z: GIoU = IoU - (C - union) / C
    return iou - (area - union) / area


def masks_to_boxes(masks):
    # Z: not used, masks' support have been eliminated
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def delta2bbox(proposals, deltas, max_shape=None, wh_ratio_clip=16 / 1000, clip_border=True, add_ctr_clamp=False, ctr_clamp=32):
    """Z: Not used. From dx dy dw dh via gx=pw*dx+px, gy=ph*dy+py, gw=pw*exp(dw), gh=ph*exp(dh) to xyxy.
    proposals = candidate boxes px py pw ph
    deltas = predicted offsets dx dy dw dh
    max_shape = max image size, crop final bbox within the image boundaries
    wh_ratio_clip = prevents predicted bbox from extreme values when exp(dw or dh)
    clip_border = whether to clip the final bbox within the image boundaries
    add_ctr_clamp = whether to clip dx dy dw dh to prevent extreme values
    ctr_clamp = when add_ctr_clamp=True, clip dx dy
    """
    # Z: get dx dy dw dh
    dxy = deltas[..., :2]
    dwh = deltas[..., 2:]

    # Compute width/height of each roi
    # Z: get px py pw ph
    pxy = proposals[..., :2]
    pwh = proposals[..., 2:]

    # Z: get pw*dx and ph*dy -> convert dx dy to same scale as px py
    # Z: PyTorch's * performs element-wise multiplication on tensors of the same shape by default
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    # Z: get gx gy gw gh
    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    # Z: convert cxcywh to xyxy
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    # Z: [x1, y1, x2, y2]
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        # Z: ... means all dimensions except the last one
        # Z: 0::2 means start at 0 and step by 2 (x coords) should <=w >=0
        # Z: 1::2 means start at 1 and step by 2 (y coords) should <=h >=0
        bboxes[..., 0::2].clamp_(min=0).clamp_(max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0).clamp_(max=max_shape[0])
    return bboxes


def bbox2delta(proposals, gt, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    """Z: Encode cxcywh GT boxes as normalized [dx, dy, dw, dh] offsets relative to cxcywh proposal boxes."""
    # hack for matcher
    if proposals.size() != gt.size():
        proposals = proposals[:, None]
        gt = gt[None]

    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph = proposals.unbind(-1)
    gx, gy, gw, gh = gt.unbind(-1)

    # Z: 0.1 to prevent division by 0 and log(0)
    # Z: gx=pw*dx+px, gy=ph*dy+py, gw=pw*exp(dw), gh=ph*exp(dh)
    dx = (gx - px) / (pw + 0.1)
    dy = (gy - py) / (ph + 0.1)
    dw = torch.log(gw / (pw + 0.1))
    dh = torch.log(gh / (ph + 0.1))
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    # Z: transform means and stds to same device and shape as deltas
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas
