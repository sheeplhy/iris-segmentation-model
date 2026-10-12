from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import cv2
import timm
import torch

from iris_system.config import default_config, project_root


def main() -> None:
    config = default_config()
    height = int(config["model"]["input_height"])
    width = int(config["model"]["input_width"])
    report = {
        "project": str(project_root()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "opencv": cv2.__version__,
        "timm": timm.__version__,
        "input_size": [height, width],
        "input_divisible_by_14": height % 14 == 0 and width % 14 == 0,
        "writable_directories": {},
    }
    for name in ("models", "runs", "evaluations", "manifests", "auto_annotations"):
        directory = project_root() / name
        directory.mkdir(parents=True, exist_ok=True)
        report["writable_directories"][name] = directory.is_dir()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("需要 Python 3.11")
    if not torch.cuda.is_available():
        raise SystemExit("没有检测到可用 CUDA GPU")
    if not report["input_divisible_by_14"]:
        raise SystemExit("模型输入尺寸必须能被 14 整除")


if __name__ == "__main__":
    main()
