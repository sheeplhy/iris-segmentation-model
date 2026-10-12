from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class FusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels),
            ConvNormAct(channels, channels),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((value, skip), dim=1))


class DPTLiteDecoder(nn.Module):
    """Fuse four ViT stages into a high-resolution dense representation."""

    def __init__(self, encoder_dim: int = 384, channels: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(encoder_dim, channels, 1) for _ in range(4)
        ])
        self.fusions = nn.ModuleList([FusionBlock(channels) for _ in range(3)])
        self.refine = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Dropout2d(dropout),
            ConvNormAct(channels, channels),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected four encoder features, got {len(features)}")
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        grid_h, grid_w = projected[0].shape[-2:]
        scales = (4.0, 2.0, 1.0, 0.5)
        pyramid: list[torch.Tensor] = []
        for value, scale in zip(projected, scales):
            size = (max(1, round(grid_h * scale)), max(1, round(grid_w * scale)))
            pyramid.append(F.interpolate(value, size=size, mode="bilinear", align_corners=False))
        value = pyramid[-1]
        for fusion, skip in zip(self.fusions, reversed(pyramid[:-1])):
            value = fusion(value, skip)
        value = self.refine(value)
        return F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)

