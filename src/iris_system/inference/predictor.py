from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from iris_system.data.transforms import DINO_MEAN, letterbox_array, normalize_image, undo_letterbox
from iris_system.training.checkpoint import load_model_checkpoint


@dataclass(slots=True)
class PredictionResult:
    probability: np.ndarray
    mask: np.ndarray
    disagreement: np.ndarray
    eye_region_probability: np.ndarray
    eye_region_mask: np.ndarray
    eye_region_disagreement: np.ndarray
    sclera_probability: np.ndarray
    sclera_mask: np.ndarray
    sclera_disagreement: np.ndarray


def _positions(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    values = list(range(0, max(length - tile + 1, 1), step))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _blend_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height) if height > 1 else np.ones(1)
    wx = np.hanning(width) if width > 1 else np.ones(1)
    return np.clip(np.outer(wy, wx), 0.05, None).astype(np.float32)


class IrisPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        model_config: dict[str, Any],
        *,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        values = model_config["model"]
        self.input_size = (int(values["input_height"]), int(values["input_width"]))
        inference = model_config.get("inference", {})
        self.default_threshold = float(inference.get("threshold", 0.5))
        self.eye_region_threshold = float(inference.get("eye_region_threshold", self.default_threshold))
        self.default_overlap = float(inference.get("overlap", 0.33))
        self.default_scales = tuple(float(x) for x in inference.get("scales", [0.75, 1.0, 1.25]))
        self.global_weight = float(inference.get("global_weight", 0.25))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device | None = None,
        use_ema: bool = True,
    ) -> tuple["IrisPredictor", dict[str, Any]]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta, _ = load_model_checkpoint(checkpoint, device=device, use_ema=use_ema)
        return cls(model, meta["model_config"], device=device), meta

    @torch.inference_mode()
    def _fixed_input(self, rgb: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(normalize_image(rgb).transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        output = self.model(tensor)
        return (
            torch.sigmoid(output["logits"][0, :2])
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )

    def _global(self, rgb: np.ndarray) -> np.ndarray:
        padded, meta = letterbox_array(
            rgb,
            self.input_size,
            interpolation=cv2.INTER_LINEAR,
            fill=tuple(int(x * 255) for x in DINO_MEAN),
        )
        probability = self._fixed_input(padded)
        return undo_letterbox(probability, meta, interpolation=cv2.INTER_LINEAR)

    def _tiled_scale(self, rgb: np.ndarray, scale: float, overlap: float) -> np.ndarray:
        original_h, original_w = rgb.shape[:2]
        if scale != 1.0:
            scaled = cv2.resize(
                rgb,
                (max(1, round(original_w * scale)), max(1, round(original_h * scale))),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
        else:
            scaled = rgb
        scaled_h, scaled_w = scaled.shape[:2]
        tile_h, tile_w = self.input_size
        pad_h, pad_w = max(0, tile_h - scaled_h), max(0, tile_w - scaled_w)
        if pad_h or pad_w:
            scaled = cv2.copyMakeBorder(
                scaled,
                0,
                pad_h,
                0,
                pad_w,
                borderType=cv2.BORDER_REFLECT_101,
            )
        canvas_h, canvas_w = scaled.shape[:2]
        accumulator = np.zeros((canvas_h, canvas_w, 2), dtype=np.float32)
        normalizer = np.zeros_like(accumulator)
        window = _blend_window(tile_h, tile_w)[..., None]
        for top in _positions(canvas_h, tile_h, overlap):
            for left in _positions(canvas_w, tile_w, overlap):
                tile = scaled[top:top + tile_h, left:left + tile_w]
                probability = self._fixed_input(tile)
                accumulator[top:top + tile_h, left:left + tile_w] += probability * window
                normalizer[top:top + tile_h, left:left + tile_w] += window
        probability = accumulator / np.maximum(normalizer, 1e-6)
        probability = probability[:scaled_h, :scaled_w]
        if probability.shape != (original_h, original_w):
            probability = cv2.resize(probability, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return probability

    def _core(self, rgb: np.ndarray, preset: str) -> np.ndarray:
        global_probability = self._global(rgb)
        if preset == "fast":
            return global_probability
        scales = (1.0,) if preset == "standard" else self.default_scales
        tiled = [self._tiled_scale(rgb, scale, self.default_overlap) for scale in scales]
        local_probability = np.mean(tiled, axis=0)
        return (
            self.global_weight * global_probability +
            (1.0 - self.global_weight) * local_probability
        ).astype(np.float32)

    def predict(
        self,
        rgb: np.ndarray,
        *,
        preset: str = "high",
        threshold: float | None = None,
    ) -> PredictionResult:
        if preset not in {"fast", "standard", "high"}:
            raise ValueError(f"Unknown inference preset: {preset}")
        probability = self._core(rgb, preset)
        if preset == "high":
            # Use an identical ensemble on both sides so disagreement measures
            # flip instability instead of a difference in inference settings.
            flipped_probability = self._core(np.ascontiguousarray(rgb[:, ::-1]), preset="high")[:, ::-1]
            disagreement = np.abs(probability - flipped_probability).astype(np.float32)
            probability = 0.5 * (probability + flipped_probability)
        else:
            disagreement = np.zeros_like(probability)
        value = self.default_threshold if threshold is None else threshold
        iris_probability = probability[..., 0].astype(np.float32)
        raw_eye_probability = probability[..., 1].astype(np.float32)
        # Project the two independent heads into the valid anatomical set while
        # preserving the primary iris prediction.
        eye_region_probability = np.maximum(raw_eye_probability, iris_probability)
        iris_binary = iris_probability >= value
        eye_binary = eye_region_probability >= self.eye_region_threshold
        iris_mask = iris_binary.astype(np.uint8) * 255
        eye_region_mask = eye_binary.astype(np.uint8) * 255
        sclera_probability = np.clip(eye_region_probability - iris_probability, 0.0, 1.0).astype(np.float32)
        sclera_mask = (eye_binary & ~iris_binary).astype(np.uint8) * 255
        sclera_disagreement = np.maximum(disagreement[..., 0], disagreement[..., 1]).astype(np.float32)
        return PredictionResult(
            probability=iris_probability,
            mask=iris_mask,
            disagreement=disagreement[..., 0].astype(np.float32),
            eye_region_probability=eye_region_probability,
            eye_region_mask=eye_region_mask,
            eye_region_disagreement=disagreement[..., 1].astype(np.float32),
            sclera_probability=sclera_probability,
            sclera_mask=sclera_mask,
            sclera_disagreement=sclera_disagreement,
        )
