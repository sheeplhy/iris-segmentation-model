from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a minimal EMA inference checkpoint")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    state = source.get("ema_state_dict") or source["state_dict"]
    if args.fp16:
        state = {
            name: value.half() if value.is_floating_point() else value
            for name, value in state.items()
        }
    source_meta = source.get("meta", {})
    payload = {
        "state_dict": state,
        "ema_state_dict": None,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "meta": {
            "architecture_version": source_meta.get("architecture_version"),
            "model_config": source_meta.get("model_config", {}),
            "epoch": source_meta.get("epoch"),
            "mode": "inference_export",
            "created_at": source_meta.get("created_at"),
            "metrics": source_meta.get("metrics", {}),
            "source_manifest_fingerprint": source_meta.get("manifest_fingerprint"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"inference checkpoint -> {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()

