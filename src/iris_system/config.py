from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = merge_dict(output[key], value)
        else:
            output[key] = value
    return output


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config() -> dict[str, Any]:
    root = project_root()
    return merge_dict(
        load_yaml(root / "configs" / "model.yaml"),
        load_yaml(root / "configs" / "training.yaml"),
    )

