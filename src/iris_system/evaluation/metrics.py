from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    pred_area, truth_area = int(pred.sum()), int(truth.sum())
    both_empty = pred_area == 0 and truth_area == 0
    iou = 1.0 if both_empty else intersection / max(union, 1)
    dice = 1.0 if both_empty else 2.0 * intersection / max(pred_area + truth_area, 1)
    precision = 1.0 if both_empty else intersection / max(pred_area, 1)
    recall = 1.0 if both_empty else intersection / max(truth_area, 1)

    pred_edge = pred & ~binary_erosion(pred)
    truth_edge = truth & ~binary_erosion(truth)
    diagonal = float(np.hypot(*truth.shape))
    tolerance = max(2, round(diagonal * 0.0025))
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    pred_dilated = cv2.dilate(pred_edge.astype(np.uint8), kernel).astype(bool)
    truth_dilated = cv2.dilate(truth_edge.astype(np.uint8), kernel).astype(bool)
    edge_precision = (pred_edge & truth_dilated).sum() / max(pred_edge.sum(), 1)
    edge_recall = (truth_edge & pred_dilated).sum() / max(truth_edge.sum(), 1)
    boundary_f1 = 1.0 if both_empty else 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-8)
    boundary_iou = 1.0 if both_empty else (
        (pred_edge & truth_dilated).sum() + (truth_edge & pred_dilated).sum()
    ) / max(pred_edge.sum() + truth_edge.sum(), 1)

    if both_empty:
        hd95 = 0.0
    elif not pred_edge.any() or not truth_edge.any():
        hd95 = diagonal
    else:
        distance_to_truth = distance_transform_edt(~truth_edge)[pred_edge]
        distance_to_pred = distance_transform_edt(~pred_edge)[truth_edge]
        hd95 = float(np.percentile(np.concatenate((distance_to_truth, distance_to_pred)), 95))
    area_error = abs(pred_area - truth_area) / max(truth_area, 1)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "boundary_iou": float(boundary_iou),
        "hd95": float(hd95),
        "area_error": float(area_error),
        "false_positive_ratio": float(pred_area / pred.size) if truth_area == 0 else 0.0,
    }

