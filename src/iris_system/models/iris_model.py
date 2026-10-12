from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from .decoder import DPTLiteDecoder


ARCHITECTURE_VERSION = "dinov2s_dptlite_detail_dualview_v1"
BODY_CLASSES = ("iris", "eye_region", "pupil")


class IrisSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        decoder_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        encoder_dim = int(getattr(self.encoder, "embed_dim", 384))
        self.decoder = DPTLiteDecoder(encoder_dim, decoder_channels, dropout)
        detail_channels = max(32, decoder_channels // 2)
        self.detail_stem = nn.Sequential(
            nn.Conv2d(3, detail_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels // 2),
            nn.GELU(),
            nn.Conv2d(detail_channels // 2, detail_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
        )
        self.detail_fusion = nn.Sequential(
            nn.Conv2d(decoder_channels + detail_channels, decoder_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
        )
        self.body_head = nn.Conv2d(decoder_channels, len(BODY_CLASSES), 1)
        self.boundary_head = nn.Conv2d(decoder_channels, 1, 1)
        self.feature_layers = (2, 5, 8, 11)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = image.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError(f"Input size must be divisible by 14, got {height}x{width}")
        features = self.encoder.get_intermediate_layers(
            image,
            n=self.feature_layers,
            reshape=True,
            norm=True,
        )
        detail = self.detail_stem(image)
        semantic = self.decoder(list(features), detail.shape[-2:])
        dense = self.detail_fusion(torch.cat((semantic, detail), dim=1))
        logits = self.body_head(dense)
        boundary_logits = self.boundary_head(dense)
        return {
            "logits": torch.nn.functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            ),
            "boundary_logits": torch.nn.functional.interpolate(
                boundary_logits, size=(height, width), mode="bilinear", align_corners=False
            ),
        }


def build_model(config: dict[str, Any] | None = None, *, pretrained: bool | None = None) -> IrisSegmentationModel:
    values = (config or {}).get("model", config or {})
    use_pretrained = values.get("pretrained", True) if pretrained is None else pretrained
    return IrisSegmentationModel(
        encoder_name=values.get("encoder", "vit_small_patch14_dinov2.lvd142m"),
        pretrained=bool(use_pretrained),
        decoder_channels=int(values.get("decoder_channels", 128)),
        dropout=float(values.get("dropout", 0.1)),
    )
