from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from iris_system.config import project_root
from iris_system.data.schemas import IMG_EXTENSIONS
from iris_system.evaluation.evaluator import comparison_image, semantic_mask
from iris_system.inference import IrisPredictor


def _boundary_band(mask: np.ndarray, width: int = 6) -> np.ndarray:
    foreground = (mask > 0).astype(np.uint8)
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    return cv2.dilate(foreground, kernel) != cv2.erode(foreground, kernel)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Automatically annotate iris images")
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", choices=["fast", "standard", "high"], default="high")
    parser.add_argument("--target", choices=["iris", "sclera", "both"], default="iris")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--doubt", type=float, default=0.1, help="Fraction (0-1] or absolute count")
    args = parser.parse_args(argv)
    image_dir = Path(args.images)
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.casefold() in IMG_EXTENSIONS)
    if not paths:
        raise SystemExit("No images found")
    output = project_root() / "auto_annotations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ("Origin", "Masks", "Comparison", "Doubt/DoubtableOrigins", "Doubt/DoubtableComparisons"):
        (output / name).mkdir(parents=True, exist_ok=True)
    predictor, _ = IrisPredictor.from_checkpoint(args.model)
    rows = []
    for index, path in enumerate(paths, start=1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = predictor.predict(rgb, preset=args.preset, threshold=args.threshold)
        if args.target == "iris":
            probabilities = [result.probability]
            disagreements = [result.disagreement]
            iris_mask, sclera_mask = result.mask, None
        elif args.target == "sclera":
            probabilities = [result.sclera_probability]
            disagreements = [result.sclera_disagreement]
            iris_mask, sclera_mask = None, result.sclera_mask
        else:
            probabilities = [result.probability, result.sclera_probability]
            disagreements = [result.disagreement, result.sclera_disagreement]
            iris_mask, sclera_mask = result.mask, result.sclera_mask
        entropy_maps = [
            -(value * np.log(value + 1e-7) +
              (1 - value) * np.log(1 - value + 1e-7)) / np.log(2)
            for value in probabilities
        ]
        entropy = np.mean(entropy_maps, axis=0)
        union_mask = np.zeros_like(result.mask)
        if iris_mask is not None:
            union_mask = np.maximum(union_mask, iris_mask)
        if sclera_mask is not None:
            union_mask = np.maximum(union_mask, sclera_mask)
        band = _boundary_band(union_mask)
        band_uncertainty = float(entropy[band].mean()) if band.any() else float(entropy.mean())
        disagreement_map = np.mean(disagreements, axis=0)
        disagreement = float(disagreement_map[band].mean()) if band.any() else float(disagreement_map.mean())
        score = band_uncertainty + disagreement
        rows.append({
            "name": path.name,
            "score": score,
            "boundary_uncertainty": band_uncertainty,
            "tta_disagreement": disagreement,
            "iris_foreground_ratio": float((result.mask > 0).mean()),
            "eye_region_foreground_ratio": float((result.eye_region_mask > 0).mean()),
            "sclera_foreground_ratio": float((result.sclera_mask > 0).mean()),
        })
        cv2.imwrite(str(output / "Origin" / path.name), bgr)
        colored = semantic_mask(iris_mask=iris_mask, sclera_mask=sclera_mask)
        comparison = comparison_image(rgb, iris_mask=iris_mask, sclera_mask=sclera_mask)
        cv2.imwrite(
            str(output / "Masks" / f"{path.stem}.png"),
            cv2.cvtColor(colored, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(output / "Comparison" / f"{path.stem}.png"),
            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
        )
        print(f"{index}/{len(paths)} {path.name}", flush=True)
    rows.sort(key=lambda row: row["score"], reverse=True)
    count = max(1, int(len(rows) * args.doubt)) if 0 < args.doubt <= 1 else min(len(rows), int(args.doubt))
    with (output / "Doubt" / "doubt_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_name = {path.name: path for path in paths}
    for row in rows[:count]:
        path = by_name[row["name"]]
        (output / "Doubt" / "DoubtableOrigins" / path.name).write_bytes(path.read_bytes())
        comparison = output / "Comparison" / f"{path.stem}.png"
        (output / "Doubt" / "DoubtableComparisons" / comparison.name).write_bytes(comparison.read_bytes())
    print(f"output -> {output}")


if __name__ == "__main__":
    main()
