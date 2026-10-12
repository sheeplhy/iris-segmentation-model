from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _masked_bce_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    focal_gamma: float | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if focal_gamma is not None:
        probability = torch.sigmoid(logits)
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        bce = bce * (1.0 - pt).pow(focal_gamma)
    bce = bce.mean(dim=(1, 2, 3))
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    weights = valid.float() * sample_weight.float()
    return ((bce + dice) * weights).sum() / weights.sum().clamp_min(1.0)


def multitask_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["logits"]
    boundary_logits = output["boundary_logits"]
    components: dict[str, torch.Tensor] = {
        "iris": _masked_bce_dice(logits[:, 0:1], target[:, 0:1], valid[:, 0], sample_weight),
        "eye_region": _masked_bce_dice(logits[:, 1:2], target[:, 1:2], valid[:, 1], sample_weight),
        "pupil": _masked_bce_dice(logits[:, 2:3], target[:, 2:3], valid[:, 2], sample_weight),
        "occlusion_boundary": _masked_bce_dice(
            boundary_logits,
            target[:, 3:4],
            valid[:, 3],
            sample_weight,
            focal_gamma=2.0,
        ),
    }
    hierarchy_weights = valid[:, 0].float() * valid[:, 1].float() * sample_weight.float()
    iris_probability = torch.sigmoid(logits[:, 0])
    eye_probability = torch.sigmoid(logits[:, 1])
    hierarchy_per_sample = F.relu(iris_probability - eye_probability).mean(dim=(1, 2))
    components["hierarchy"] = (
        (hierarchy_per_sample * hierarchy_weights).sum() /
        hierarchy_weights.sum().clamp_min(1.0)
    )
    resolved_weights = dict(loss_weights)
    if "eye_region" not in resolved_weights and "sclera" in resolved_weights:
        resolved_weights["eye_region"] = resolved_weights["sclera"]
    total = sum(components[name] * float(resolved_weights.get(name, 0.0)) for name in components)
    return total, {name: float(value.detach()) for name, value in components.items()}
