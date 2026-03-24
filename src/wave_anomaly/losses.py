from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (values * weights).sum() / denom


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    weights = mask.to(probs.dtype)
    intersection = torch.sum(probs * targets * weights)
    denom = torch.sum(probs * weights) + torch.sum(targets * weights)
    return 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight.to(logits.device),
        )
        return masked_mean(bce, mask) + dice_loss_from_logits(logits, targets, mask)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return masked_mean(loss, mask)


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_type = str(config["train"]["loss_type"]).lower()
    if loss_type == "bce_dice":
        return WeightedBCEDiceLoss(pos_weight=float(config["train"]["pos_weight"]))
    if loss_type == "focal":
        return FocalLoss(
            alpha=float(config["train"].get("focal_alpha", 0.75)),
            gamma=float(config["train"].get("focal_gamma", 2.0)),
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")
