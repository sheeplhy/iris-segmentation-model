from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from iris_system.data.records import SampleRecord
from iris_system.models.iris_model import ARCHITECTURE_VERSION, build_model


def manifest_fingerprint(records: list[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(record.image_sha256.encode("ascii"))
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    epoch: int,
    mode: str,
    records: list[SampleRecord],
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    parent_checkpoint: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    training_samples = {
        record.sample_id: {
            "sha256": record.image_sha256,
            "perceptual_hash": record.perceptual_hash,
            "dataset": record.dataset,
            "domain": record.domain,
        }
        for record in records
    }
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": {
            "architecture_version": ARCHITECTURE_VERSION,
            "model_config": model_config,
            "epoch": epoch,
            "mode": mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "manifest_fingerprint": manifest_fingerprint(records),
            "training_samples": training_samples,
            "metrics": metrics,
            "parent_checkpoint": parent_checkpoint,
        },
    }
    torch.save(payload, path)


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {})
    if meta.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Incompatible checkpoint architecture: {meta.get('architecture_version')} != {ARCHITECTURE_VERSION}"
        )
    model = build_model(meta.get("model_config", {}), pretrained=False).to(device)
    state = checkpoint.get("ema_state_dict") if use_ema and checkpoint.get("ema_state_dict") else checkpoint["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta, checkpoint


def append_registry(registry_path: str | Path, entry: dict[str, Any]) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = []
    else:
        registry = []
    registry.append(entry)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

