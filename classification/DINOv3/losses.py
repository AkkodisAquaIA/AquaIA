import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# LOSS
# =========================

class FocalLoss(nn.Module):
    """
    Focal Loss pour classification multi-classes.

    Args:
        alpha (None, float, list, tuple, Tensor):
            - None: pas de pondération de classes
            - float/int: facteur global appliqué à toutes les classes
            - list/tuple/Tensor de shape (C,): poids par classe
        gamma (float): paramètre de focalisation (>= 0)
        reduction (str): 'mean' | 'sum' | 'none'
        ignore_index (int): label ignoré, comme dans CrossEntropyLoss
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", ignore_index=-100):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.detach().clone().float())
        else:
            raise TypeError("alpha must be None, float, int, list, tuple, or Tensor")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C)
        targets: (B,)
        """
        if logits.ndim != 2:
            raise ValueError(f"logits doit être de shape (B, C), reçu {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets doit être de shape (B,), reçu {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError("Batch size de logits et targets incompatible")

        # cross entropy par échantillon, sans réduction
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        ) 

        # masque ignore_index
        valid_mask = (targets != self.ignore_index)

        if valid_mask.sum() == 0:
            return logits.new_zeros(())

        ce_loss_valid = ce_loss[valid_mask]
        targets_valid = targets[valid_mask]

        # pt = probabilité de la vraie classe
        pt = torch.exp(-ce_loss_valid)  # plus stable que softmax+gather

        # facteur alpha
        if self.alpha is None:
            alpha_t = 1.0
        else:
            if self.alpha.ndim == 0:
                alpha_t = self.alpha.to(logits.device)
            elif self.alpha.ndim == 1:
                if self.alpha.numel() != logits.size(1):
                    raise ValueError(
                        f"alpha a {self.alpha.numel()} classes, mais logits a {logits.size(1)} classes"
                    )
                alpha_t = self.alpha.to(logits.device)[targets_valid]
            else:
                raise ValueError("alpha doit être scalaire ou 1D")

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce_loss_valid

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            out = logits.new_zeros(targets.shape, dtype=loss.dtype)
            out[valid_mask] = loss
            return out
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")