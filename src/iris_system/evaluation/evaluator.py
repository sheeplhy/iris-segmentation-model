from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iris_system.data.records import SampleRecord
from iris_system.data.targets import build_targets
from iris_system.inference import IrisPredictor

from .metrics import segmentation_metrics


def semantic_mask(
    iris_mask: np.ndarray | None = None,
    sclera_mask: np.ndarray | None = None,
) -> np.ndarray:
    reference = iris_mask if iris_mask is not None else sclera_mask
    if reference is None:
        raise ValueError("at least one mask is required")
    output = np.zeros((*reference.shape, 3), dtype=np.uint8)
    if sclera_mask is not None:
        output[sclera_mask > 0] = (255, 0, 0)
    if iris_mask is not None:
        # Iris wins any overlap in the visualization. The raw predictions stay
        # independent and are not changed by this display-only rule.
        output[iris_mask > 0] = (0, 255, 0)
    return output


def green_mask(mask: np.ndarray) -> np.ndarray:
    return semantic_mask(iris_mask=mask)


def comparison_image(
    rgb: np.ndarray,
    iris_mask: np.ndarray | None = None,
    sclera_mask: np.ndarray | None = None,
) -> np.ndarray:
    colored = semantic_mask(iris_mask=iris_mask, sclera_mask=sclera_mask)
    overlay = rgb.copy()
    selected = np.any(colored > 0, axis=-1)
    overlay[selected] = (0.5 * overlay[selected] + 0.5 * colored[selected]).astype(np.uint8)
    gap = np.zeros((rgb.shape[0], 8, 3), dtype=np.uint8)
    return np.concatenate((rgb, gap, colored, gap, overlay), axis=1)


def _mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "iou", "dice", "precision", "recall", "boundary_f1", "boundary_iou",
        "hd95", "area_error", "false_positive_ratio",
    ]
    if not rows:
        return {name: float("nan") for name in metric_names}
    return {name: float(np.mean([row[name] for row in rows])) for name in metric_names}


def contamination_report(records: list[SampleRecord], checkpoint_meta: dict[str, Any]) -> list[dict[str, str]]:
    trained = checkpoint_meta.get("training_samples", {})
    by_hash = {value.get("sha256"): key for key, value in trained.items() if value.get("sha256")}
    trained_perceptual = [
        (key, value.get("perceptual_hash"))
        for key, value in trained.items()
        if value.get("perceptual_hash")
    ]
    output = []
    for record in records:
        if record.sample_id in trained:
            output.append({"sample_id": record.sample_id, "reason": "sample_id was used for training"})
        elif record.image_sha256 and record.image_sha256 in by_hash:
            output.append({"sample_id": record.sample_id, "reason": f"identical to training sample {by_hash[record.image_sha256]}"})
        elif record.perceptual_hash:
            value = int(record.perceptual_hash, 16)
            nearest = min(
                ((value ^ int(candidate, 16)).bit_count(), sample_id)
                for sample_id, candidate in trained_perceptual
            ) if trained_perceptual else None
            # A strict threshold catches resized/re-encoded copies while avoiding
            # treating merely similar eye photographs as leakage.
            if nearest is not None and nearest[0] <= 4:
                output.append({
                    "sample_id": record.sample_id,
                    "reason": f"near-duplicate of training sample {nearest[1]} (pHash distance {nearest[0]}/256)",
                })
    return output


def evaluate_records(
    predictor: IrisPredictor,
    records: list[SampleRecord],
    *,
    output_dir: str | Path,
    checkpoint_meta: dict[str, Any],
    preset: str = "high",
    threshold: float | None = None,
    target: str = "iris",
) -> dict[str, Any]:
    if target not in {"iris", "eye_region", "sclera"}:
        raise ValueError(f"Unknown evaluation target: {target}")
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        bgr = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unable to read {record.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        targets, valid = build_targets(record)
        target_is_valid = (
            valid["iris"] and valid["eye_region"]
            if target == "sclera" else valid[target]
        )
        if not target_is_valid:
            continue
        result = predictor.predict(rgb, preset=preset, threshold=threshold)
        if target == "iris":
            predicted_mask = result.mask
            truth = targets["iris"] > 0.5
        elif target == "eye_region":
            predicted_mask = result.eye_region_mask
            truth = targets["eye_region"] > 0.5
        else:
            predicted_mask = result.sclera_mask
            truth = (targets["eye_region"] > 0.5) & ~(targets["iris"] > 0.5)
        metrics = segmentation_metrics(predicted_mask > 0, truth)
        row = {
            "sample_id": record.sample_id,
            "dataset": record.dataset,
            "domain": record.domain,
            "quality": record.quality,
            "boundary_status": record.boundary_status,
            "unavailable": record.unavailable,
            **metrics,
        }
        rows.append(row)
        directory = predictions_dir / record.dataset
        directory.mkdir(parents=True, exist_ok=True)
        comparison = comparison_image(
            rgb,
            iris_mask=predicted_mask if target == "iris" else None,
            sclera_mask=predicted_mask if target != "iris" else None,
        )
        cv2.imwrite(
            str(directory / f"{Path(record.image_path).stem}.png"),
            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["domain"]].append(row)
    per_domain = {domain: {"n": len(items), **_mean(items)} for domain, items in groups.items()}
    macro_names = next(iter(per_domain.values())).keys() if per_domain else []
    domain_macro = {
        name: float(np.mean([values[name] for values in per_domain.values()]))
        for name in macro_names if name != "n"
    }
    contamination = contamination_report(records, checkpoint_meta)
    summary = {
        "n": len(rows),
        "target": target,
        "overall": _mean(rows),
        "per_domain": per_domain,
        "domain_macro": domain_macro,
        "contaminated": bool(contamination),
        "contamination": contamination,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
