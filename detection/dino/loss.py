import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ..utils import box_ops
from .utils.misc import accuracy, get_world_size


# Z: https://github.com/impiga/Plain-DETR/blob/main/models/segmentation.py#L224
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
                Z: supposed shape [batch_size, num_queries, num_classes]
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                Z: expected to be one hot encoded of shape [batch_size, num_queries, num_classes]
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # Z: get probability when predicted as positive
    prob = inputs.sigmoid()
    # Z: compute the loss for positives and negatives
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # Z: if targets is 1, p_t = prob, else p_t = 1 - prob
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # Z: FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        # [batch_size, num_queries, num_classes]
        loss = alpha_t * loss

    # Z: average over query dimension to shape [batch_size, num_classes]
    # Z: then sum up to a scalar then divide by nb GT boxes of batch
    return loss.mean(1).sum() / num_boxes


# Z: https://github.com/impiga/Plain-DETR/blob/main/models/detr.py#L384
class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, focal_alpha=0.25, reparam=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories.
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            focal_alpha: alpha in Focal Loss
            loss_bbox_type: how to perform loss_bbox
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.loss_bbox_type = "l1" if (not reparam) else "reparam"
        # Z: the losses to be applied
        self.losses = ["labels", "boxes", "cardinality"]

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        # Z: src_logits [batch_size, num_queries, num_classes]
        src_logits = outputs["pred_logits"]

        # Z: get batch_idx and src_idx for matched predictions
        # batch_idx : which image in the batch each matched prediction belongs to.
        # src_idx : query index of each matched prediction within its image.
        idx = self._get_src_permutation_idx(indices)  # gets (batch_idx, i)

        # Z: t -> targets dict, indices -> (src, tgt), J -> tgt
        # Z: get a list of GT class labels
        # Z: then flatten to a 1D tensor of shape [num_matched_boxes]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # flatten

        # background class is the last one (num_classes)
        # Z: initialize a tensor of shape [batch_size, num_queries]
        # Z: filled with num_classes (temporary index for no-object class)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )  # (B, N)

        # Z:fill matched positions with their corresponding GT class labels
        # Z: shape [batch_size, num_queries]
        target_classes[idx] = target_classes_o

        # Z: initialize a one-hot tensor with 0 and an extra column
        # Z: shape [batch_size, num_queries, num_classes + 1]
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        # Z: On the dimension 2 of target_classes_onehot,
        # Z: fill with 1 according to the coords in target_classes.unsqueeze(-1)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # Supervision for the no-object tokens is all zeros
        # Z: remove the extra dummy column so target_classes_onehot matches src_logits shape
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=3.0,
            )
            # Z: sigmoid_focal_loss() returns num_queries averaged loss
            # Z: so multiply by num_queries to get the sum, avoid tiny loss value
            * src_logits.shape[1]
        )
        # Z: register loss value
        losses = {"loss_ce": loss_ce}

        # TODO this should probably be a separate loss, not hacked in this one here
        # Z: compute top1 accuracy then subtract from 100 to get error rate
        losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        # Z: count the number of target boxes for each batch element
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.sigmoid().max(-1).values >= 0.5).sum(1)
        # Z: compute the L1 loss between nb predicted boxes and nb GT boxes
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        # Z: outputs["pred_boxes"] of shape [batch_size, num_queries, 4]
        # Z: get matched predicted boxes, shape [num_matched_boxes, 4]
        src_boxes = outputs["pred_boxes"][idx]
        # Z: get GT boxes
        # Z: then concatenate to a single tensor of shape [num_matched_boxes, 4]
        # Z: t -> targets dict, indices -> (src, tgt), i -> tgt
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if self.loss_bbox_type == "l1":
            # print(src_boxes.shape)
            # print(target_boxes.shape)
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        elif self.loss_bbox_type == "reparam":
            src_deltas = outputs["pred_deltas"][idx]
            src_boxes_old = outputs["pred_boxes_old"][idx]
            target_deltas = box_ops.bbox2delta(src_boxes_old, target_boxes)
            loss_bbox = F.l1_loss(src_deltas, target_deltas, reduction="none")
        else:
            raise NotImplementedError

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # Z: only need matched src/target so diagonal
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        """Z: returns 
        batch_idx : which image in the batch each matched prediction belongs to.
        src_idx : query index of each matched prediction within its image."""
        # permute predictions following indices
        # print(indices)
        # Z: indices is a list of tuples (src, tgt) for each batch element after Hungarian matching
        # Z: src and tgt are the indices of the matched predictions and targets respectively
        # Z: ex indices = [( tensor([1, 4]), tensor([0, 1]) ),
        # Z:               ( tensor([0, 3, 6]), tensor([1, 0, 2]) )]
        # Z: image 0 -> prediction 1,4 match respectively target 0,1
        # Z: image 1 -> prediction 0,3,6 match respectively target 1,0,2

        # Z: create a tensor of shape src.shape filled with i then concatenate for all batch elements
        # Z: batch_idx = tensor([0, 0, 1, 1, 1])
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # Z: src_idx = tensor([1, 4, 0, 3, 6])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Z: returns 
        batch_idx : which image in the batch each matched target belongs to.
        tgt_idx : target index of each matched target within its image."""
        # permute targets following indices
        # Z: batch_idx = tensor([0, 0, 1, 1, 1])
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        # Z: tgt_idx = tensor([0, 1, 1, 0, 2])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        # Z: raise an error for unknown loss types
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # Z: call and return the loss function matching the requested loss name
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Z: gather only the last decoder layer outputs
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # Z: total number of GT boxes in the batch
        num_boxes = sum(len(t["labels"]) for t in targets)
        # Z: convert to tensor and move to the same device as outputs
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # Z: average number of GT boxes per node, for normalization purposes
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        # Z: !Warning! no log parameter in loss_labels
                        kwargs["log"] = False
                    # Z: add suffix _i to the loss name to avoid overwriting the main loss
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            # Z: !Warning! actually don't have enc_outputs in the current implementation
            enc_outputs = outputs["enc_outputs"]
            # Z: deepcopy because later change labels
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                # Z: all GT objects are assigned to class 0, one single foreground class
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    # Z: !Warning! no log parameter in loss_labels
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
