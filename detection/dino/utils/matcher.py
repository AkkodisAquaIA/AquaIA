# Z: https://github.com/impiga/Plain-DETR/blob/main/models/matcher.py
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
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from detection.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, bbox2delta


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    # Z: Original DETR: class cost uses a linear approximation of probability, 
    # Z: considering only the score of the predicted Query on the target class. 
    # Z: Current DETR: class cost uses Focal Loss, considering the cost difference 
    # Z: between the predicted Query being matched as a target versus being treated as background. 
    # Z: L1 and GIoU costs calculate the distance between predicted boxes and all GT boxes.

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_bbox_type: str = "l1",
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_bbox_type: This decides how to calculate box loss.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_bbox_type = cost_bbox_type
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]  # (B, N, Nc)

            # We flatten to compute the cost matrices in a batch
            # Z: get pred prob and bbox coords via fusion of dim 0 and 1
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # B*N, Nc
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            # Z: get GT labels (number) and bbox coords
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            # Z: FL for neg samples = -(1-alpha)*p^gamma*log(1-p), 1e-8 avoids log(0)
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            # Z: FL for pos samples = -alpha*(1-p)^gamma*log(p), 1e-8 avoids log(0)
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # Z: How much does the cost increase if this prediction is assigned
            # Z: to this target (positive) versus being unassigned (as a negative)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            if self.cost_bbox_type == "l1":
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # Z: !Warning! reparam not fonctional
            elif self.cost_bbox_type == "reparam":
                # Z: get pred dx dy dw dh
                out_delta = outputs["pred_deltas"].flatten(0, 1)
                # Z: get base bbox coords px py pw ph
                out_bbox_old = outputs["pred_boxes_old"].flatten(0, 1)
                # Z: get GT dx dy dw dh
                tgt_delta = bbox2delta(out_bbox_old, tgt_bbox)
                # Z: compute L1 distance between pred dx dy dw dh and GT dx dy dw dh
                # Z: add one dimension to out_delta
                cost_bbox = torch.cdist(out_delta[:, None], tgt_delta, p=1).squeeze(1)
            else:
                raise NotImplementedError

            # Compute the giou cost betwen boxes
            # Z: negative -> 2 bboxes match -> GIoU high -> cost low
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # Z: reshape C to [batch_size, num_queries, sum of GT boxes in batch]
            C = C.view(bs, num_queries, -1).cpu()
            # print(C.shape)

            # Z: count how many GT boxses there are for each image in the current batch, ex. [2,3]
            sizes = [len(v["boxes"]) for v in targets]
            # print(sizes)
            # print(C.split(sizes, dim=-1)[0].shape)
            # print(len(C.split(sizes, -1)))
            # Z: split C by last dim according to sizes, c of shape [batch_size, num_queries, num_target_boxes_i] for each image in the batch
            # Z: c[i] get the cost matrix for the i-th image in the batch, of shape [num_queries, num_target_boxes_i]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # Z: Returns a list of (index_i, index_j) tuples for each image in the batch
            # Z: index_i: indices of predictions, index_j: indices of corresponding GTs
            # Z: [(tensor([10, 25]), tensor([0, 1])),      <-- Image 0: 2 targets matched (10->0, 25->1)
            # Z:   (tensor([5, 88, 12]), tensor([0, 1, 2])) <-- Image 1: 3 targets matched (5->0, 88->1, 12->2)
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]
