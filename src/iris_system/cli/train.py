from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from iris_system.config import default_config, load_yaml, merge_dict, project_root
from iris_system.data.manifest import compile_dataset, load_manifest, save_manifest
from iris_system.data.validation import validate_records
from iris_system.training.checkpoint import append_registry
from iris_system.training.trainer import train_model


def _records(roots: list[str] | None, manifest: str | None):
    if manifest:
        return load_manifest(manifest)
    output = []
    for root in roots or []:
        output.extend(compile_dataset(root))
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or incrementally update the iris model")
    parser.add_argument("--root", action="append", help="Cumulative training dataset root")
    parser.add_argument("--manifest")
    parser.add_argument("--val-root", action="append")
    parser.add_argument("--val-manifest")
    parser.add_argument(
        "--skip-deep-validation",
        action="store_true",
        help="Skip rebuilding every target when using a manifest that was already validated",
    )
    parser.add_argument("--mode", choices=["incremental", "resume", "rebuild"], default="incremental")
    parser.add_argument("--base-model")
    parser.add_argument("--resume-model")
    parser.add_argument("--config")
    parser.add_argument("--run-name")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--encoder-lr-factor", type=float)
    parser.add_argument("--incremental-lr-factor", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--freeze-encoder-epochs", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--loss-iris", type=float)
    parser.add_argument("--loss-eye-region", type=float)
    parser.add_argument("--loss-sclera", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--loss-pupil", type=float)
    parser.add_argument("--loss-boundary", type=float)
    parser.add_argument("--loss-hierarchy", type=float)
    args = parser.parse_args(argv)
    if not args.root and not args.manifest:
        parser.error("--root or --manifest is required")
    config = default_config()
    if args.config:
        config = merge_dict(config, load_yaml(args.config))
    training_overrides = {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "encoder_lr_factor": args.encoder_lr_factor,
        "incremental_lr_factor": args.incremental_lr_factor,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "amp": args.amp,
    }
    config.setdefault("training", {}).update(
        {key: value for key, value in training_overrides.items() if value is not None}
    )
    loss_overrides = {
        "iris": args.loss_iris,
        "eye_region": args.loss_eye_region if args.loss_eye_region is not None else args.loss_sclera,
        "pupil": args.loss_pupil,
        "occlusion_boundary": args.loss_boundary,
        "hierarchy": args.loss_hierarchy,
    }
    config.setdefault("loss", {}).update(
        {key: value for key, value in loss_overrides.items() if value is not None}
    )

    training = config["training"]
    positive_integer_keys = (
        "max_epochs", "batch_size", "gradient_accumulation", "early_stopping_patience",
    )
    for key in positive_integer_keys:
        if int(training[key]) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be greater than zero")
    for key in ("warmup_epochs", "freeze_encoder_epochs", "num_workers"):
        if int(training[key]) < 0:
            parser.error(f"--{key.replace('_', '-')} cannot be negative")
    for key in ("learning_rate", "encoder_lr_factor", "incremental_lr_factor"):
        if float(training[key]) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be greater than zero")
    if float(training["weight_decay"]) < 0:
        parser.error("--weight-decay cannot be negative")
    if any(float(value) < 0 for value in config["loss"].values()):
        parser.error("loss weights cannot be negative")
    train_records = _records(args.root, args.manifest)
    val_records = _records(args.val_root, args.val_manifest)
    # Validate each split independently. The same records are allowed to occur
    # in both splits for convergence-only experiments, while duplicates inside
    # either split remain a data error.
    deep_validation = not args.skip_deep_validation
    issues = validate_records(train_records, deep=deep_validation) + validate_records(
        val_records, deep=deep_validation
    )
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise SystemExit("\n".join(f"{item.sample_id}: {item.message}" for item in errors))
    train_ids = {record.sample_id for record in train_records}
    train_hashes = {record.image_sha256 for record in train_records if record.image_sha256}
    leakage = [
        record.sample_id for record in val_records
        if record.sample_id in train_ids or (record.image_sha256 and record.image_sha256 in train_hashes)
    ]
    if leakage:
        preview = ", ".join(leakage[:10])
        print(
            f"WARNING: development set overlaps training data ({len(leakage)} samples): {preview}",
            flush=True,
        )
        print(
            "WARNING: validation metrics can be used to monitor convergence, but not to estimate generalization.",
            flush=True,
        )
    effective_batch = int(training["batch_size"]) * int(training["gradient_accumulation"])
    batches_per_epoch = (len(train_records) + int(training["batch_size"]) - 1) // int(training["batch_size"])
    updates_per_epoch = (
        batches_per_epoch + int(training["gradient_accumulation"]) - 1
    ) // int(training["gradient_accumulation"])
    decoder_lr = float(training["learning_rate"])
    if args.mode == "incremental":
        decoder_lr *= float(training["incremental_lr_factor"])
    encoder_lr = decoder_lr * float(training["encoder_lr_factor"])
    print(
        f"train_samples={len(train_records)} val_samples={len(val_records)} "
        f"effective_batch={effective_batch} optimizer_updates_per_epoch={updates_per_epoch}",
        flush=True,
    )
    print(
        f"mode={args.mode} max_epochs={training['max_epochs']} "
        f"decoder_lr={decoder_lr:.3g} encoder_lr={encoder_lr:.3g} "
        f"patience={training['early_stopping_patience']}",
        flush=True,
    )
    if not val_records:
        print(
            "validation disabled: early stopping is disabled and the last epoch checkpoint will be exported",
            flush=True,
        )
    root = project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    current = models_dir / "current.pth"
    base_model = args.base_model or (str(current) if current.exists() else None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_{args.mode}"
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(train_records, run_dir / "train_manifest.jsonl")
    if val_records:
        save_manifest(val_records, run_dir / "val_manifest.jsonl")
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    best = train_model(
        train_records=train_records,
        val_records=val_records,
        config=config,
        run_dir=run_dir,
        mode=args.mode,
        base_checkpoint=base_model,
        resume_checkpoint=args.resume_model,
    )
    version_path = models_dir / f"{run_name}.pth"
    version_path.write_bytes(best.read_bytes())
    if args.promote:
        current.write_bytes(best.read_bytes())
    append_registry(models_dir / "registry.json", {
        "name": run_name,
        "checkpoint": str(version_path.resolve()),
        "parent": base_model,
        "mode": args.mode,
        "promoted": bool(args.promote),
        "checkpoint_selection": "validation_best" if val_records else "last_epoch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"model version -> {version_path}")
    if args.promote:
        print(f"current model -> {current}")


if __name__ == "__main__":
    main()
