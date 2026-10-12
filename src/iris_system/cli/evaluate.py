from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from iris_system.config import project_root
from iris_system.data.manifest import compile_dataset, load_manifest
from iris_system.evaluation import evaluate_records
from iris_system.inference import IrisPredictor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an iris model on a selected dataset or benchmark")
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root")
    source.add_argument("--benchmark", help="Locked JSONL manifest")
    parser.add_argument("--output")
    parser.add_argument("--preset", choices=["fast", "standard", "high"], default="high")
    parser.add_argument("--target", choices=["iris", "eye_region", "sclera"], default="iris")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    records = load_manifest(args.benchmark) if args.benchmark else compile_dataset(args.root)
    predictor, meta = IrisPredictor.from_checkpoint(args.model)
    output = args.output or str(project_root() / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    summary = evaluate_records(
        predictor,
        records,
        output_dir=output,
        checkpoint_meta=meta,
        preset=args.preset,
        threshold=args.threshold,
        target=args.target,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report -> {output}")


if __name__ == "__main__":
    main()
