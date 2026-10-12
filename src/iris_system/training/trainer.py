from __future__ import annotations

import csv
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from iris_system.data.dataset import IrisDataset
from iris_system.data.records import SampleRecord
from iris_system.data.sampler import domain_balanced_sampler
from iris_system.models import build_model

from .checkpoint import load_model_checkpoint, save_checkpoint
from .ema import ModelEMA
from .losses import multitask_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _lr_lambda(epoch: int, max_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return max((epoch + 1) / max(warmup_epochs, 1), 1e-3)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _write_training_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row}, key=lambda key: (key != "epoch", key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    dice_sums = {"iris": 0.0, "eye_region": 0.0}
    dice_counts = {"iris": 0, "eye_region": 0}
    total = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        output = model(image)
        loss, _ = multitask_loss(output, target, valid, sample_weight, loss_weights)
        for name, channel in (("iris", 0), ("eye_region", 1)):
            selected = valid[:, channel] > 0.5
            if not selected.any():
                continue
            prediction = torch.sigmoid(output["logits"][selected, channel]) >= 0.5
            truth = target[selected, channel] >= 0.5
            intersection = (prediction & truth).sum(dim=(1, 2)).float()
            denominator = prediction.sum(dim=(1, 2)) + truth.sum(dim=(1, 2))
            dice = torch.where(
                denominator > 0,
                2.0 * intersection / denominator.clamp_min(1),
                torch.ones_like(intersection),
            )
            dice_sums[name] += float(dice.sum())
            dice_counts[name] += int(selected.sum())
        batch_size = image.shape[0]
        total_loss += float(loss) * batch_size
        total += batch_size
    metrics = {"val_loss": total_loss / max(total, 1)}
    task_scores = []
    for name in ("iris", "eye_region"):
        if dice_counts[name]:
            score = dice_sums[name] / dice_counts[name]
            metrics[f"val_{name}_dice"] = score
            task_scores.append(score)
    metrics["val_dice"] = float(np.mean(task_scores)) if task_scores else 0.0
    return metrics


def train_model(
    *,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    config: dict[str, Any],
    run_dir: str | Path,
    mode: str = "incremental",
    base_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
    device: torch.device | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    values = config["training"]
    model_values = config["model"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(values.get("seed", 66)))
    input_size = (int(model_values["input_height"]), int(model_values["input_width"]))
    if input_size[0] % 14 or input_size[1] % 14:
        raise ValueError(f"DINOv2 input size must be divisible by 14: {input_size}")

    resume_payload = None
    parent = base_checkpoint
    if mode == "resume":
        if not resume_checkpoint:
            raise ValueError("resume mode requires resume_checkpoint")
        model, _, resume_payload = load_model_checkpoint(resume_checkpoint, device=device, use_ema=False)
        parent = resume_checkpoint
    elif mode == "incremental":
        if not base_checkpoint:
            raise ValueError("incremental mode requires a base checkpoint")
        model, _, _ = load_model_checkpoint(base_checkpoint, device=device, use_ema=False)
    elif mode == "rebuild":
        model = build_model(config, pretrained=bool(model_values.get("pretrained", True))).to(device)
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    quality_weights = values.get("quality_weights", {})
    train_dataset = IrisDataset(
        train_records,
        input_size=input_size,
        train=True,
        quality_weights=quality_weights,
        view_config=values,
    )
    val_dataset = IrisDataset(val_records, input_size=input_size, train=False, quality_weights=quality_weights)
    sampler = domain_balanced_sampler(train_records)
    workers = int(values.get("num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(values.get("batch_size", 2)),
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
        persistent_workers=workers > 0,
    ) if val_records else None

    base_lr = float(values.get("learning_rate", 1e-4))
    if mode == "incremental":
        base_lr *= float(values.get("incremental_lr_factor", 0.33))
    encoder_factor = float(values.get("encoder_lr_factor", 0.1))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": base_lr * encoder_factor},
            {
                "params": list(model.decoder.parameters()) + list(model.body_head.parameters()) + list(model.boundary_head.parameters()),
                "lr": base_lr,
            },
        ],
        weight_decay=float(values.get("weight_decay", 1e-4)),
    )
    max_epochs = int(values.get("max_epochs", 120))
    warmup_epochs = int(values.get("warmup_epochs", 5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_lambda(epoch, max_epochs, warmup_epochs),
    )
    amp_enabled = bool(values.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model)
    start_epoch = 0
    if resume_payload is not None:
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])
        if resume_payload.get("scheduler"):
            scheduler.load_state_dict(resume_payload["scheduler"])
        if resume_payload.get("scaler"):
            scaler.load_state_dict(resume_payload["scaler"])
        if resume_payload.get("ema_state_dict"):
            ema.load_state_dict(resume_payload["ema_state_dict"])
        start_epoch = int(resume_payload.get("meta", {}).get("epoch", 0))

    loss_weights = config.get("loss", {})
    accumulation = int(values.get("gradient_accumulation", 1))
    freeze_epochs = int(values.get("freeze_encoder_epochs", 0)) if mode == "rebuild" else 0
    patience = int(values.get("early_stopping_patience", 20))
    best_score = -float("inf")
    stale_epochs = 0
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    rows: list[dict[str, float | int]] = []

    for epoch in range(start_epoch + 1, max_epochs + 1):
        epoch_started = time.perf_counter()
        progress_interval = max(1, len(train_loader) // 10)
        model.set_encoder_trainable(epoch > freeze_epochs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            context = torch.amp.autocast("cuda", enabled=True) if amp_enabled else nullcontext()
            with context:
                output = model(image)
                loss, _ = multitask_loss(output, target, valid, sample_weight, loss_weights)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            epoch_loss += float(loss.detach()) * image.shape[0]
            seen += image.shape[0]
            if step % progress_interval == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_started
                rate = step / max(elapsed, 1e-6)
                eta_minutes = (len(train_loader) - step) / max(rate, 1e-6) / 60.0
                print(
                    f"epoch={epoch}/{max_epochs} step={step}/{len(train_loader)} "
                    f"progress={100.0 * step / len(train_loader):.0f}% "
                    f"eta={eta_minutes:.1f}min",
                    flush=True,
                )
        scheduler.step()
        metrics = {"train_loss": epoch_loss / max(seen, 1)}
        has_validation = val_loader is not None
        if has_validation:
            # Checkpoint selection now measures the same EMA parameters that
            # IrisPredictor loads by default.
            with ema.average_parameters(model):
                metrics.update(validate_epoch(model, val_loader, device, loss_weights))
            score = metrics["val_dice"]
        rows.append({"epoch": epoch, **metrics})
        _write_training_log(run_dir / "train_log.csv", rows)
        save_checkpoint(
            last_path,
            model=model,
            model_config=config,
            epoch=epoch,
            mode=mode,
            records=train_records,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_state=ema.state_dict(),
            parent_checkpoint=parent,
        )
        if has_validation and score > best_score:
            best_score = score
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model=model,
                model_config=config,
                epoch=epoch,
                mode=mode,
                records=train_records,
                metrics=metrics,
                ema_state=ema.state_dict(),
                parent_checkpoint=parent,
            )
        elif has_validation:
            stale_epochs += 1
        validation_text = "no_validation"
        if "val_dice" in metrics:
            validation_text = f"val_dice={metrics['val_dice']:.5f}"
            for name in ("iris", "eye_region"):
                key = f"val_{name}_dice"
                if key in metrics:
                    validation_text += f" {key}={metrics[key]:.5f}"
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.5f} {validation_text}", flush=True)
        if has_validation and stale_epochs >= patience:
            print(f"early stopping after {stale_epochs} epochs without improvement", flush=True)
            break

    _write_training_log(run_dir / "train_log.csv", rows)
    # Without a development set there is no principled checkpoint-selection
    # score. In that mode the requested result is exactly the final epoch.
    return best_path if val_loader is not None else last_path
