"""Post-processing helpers for Severstal mask predictions.

This module converts probability masks into binary masks using class-specific
thresholds and connected-component area filtering.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def normalize_class_values(value: float | int | Sequence[float | int], num_classes: int) -> list[float]:
    """Return one value per class.

    Args:
        value: Scalar value or sequence with one value per class.
        num_classes: Number of prediction classes.

    Raises:
        ValueError: If sequence length does not match ``num_classes``.
    """
    if isinstance(value, (int, float)):
        return [float(value)] * num_classes

    values = [float(v) for v in value]
    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} values, got {len(values)}")
    return values


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` from a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    if min_area <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 1

    return cleaned


def postprocess_prediction(
    prob_masks: np.ndarray,
    thresholds: float | Sequence[float],
    min_pixels: int | Sequence[int],
) -> np.ndarray:
    """Convert probability masks to cleaned binary masks.

    Args:
        prob_masks: Array with shape ``(C, H, W)``, values normally in ``[0, 1]``.
        thresholds: Scalar threshold or one threshold per class.
        min_pixels: Scalar min component area or one value per class.

    Returns:
        ``uint8`` array with shape ``(C, H, W)`` and values 0/1.
    """
    if prob_masks.ndim != 3:
        raise ValueError(f"Expected shape (C,H,W), got {prob_masks.shape}")

    num_classes = prob_masks.shape[0]
    thresholds_per_class = normalize_class_values(thresholds, num_classes)
    min_pixels_per_class = [int(v) for v in normalize_class_values(min_pixels, num_classes)]

    output = np.zeros_like(prob_masks, dtype=np.uint8)
    for class_idx in range(num_classes):
        binary = prob_masks[class_idx] > thresholds_per_class[class_idx]
        output[class_idx] = remove_small_components(binary, min_pixels_per_class[class_idx])

    return output


def load_postprocess_config(path: str | Path | None = None) -> tuple[list[float], list[int]]:
    """Load tuned postprocess config, or fall back to train_config defaults."""
    from train_config import PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS

    default_thresholds = list(PIXEL_THRESHOLDS)
    default_min_pixels = [int(v) for v in MIN_DEFECT_PIXELS_PER_CLASS]

    if path is None:
        return default_thresholds, default_min_pixels

    path = Path(path)
    if not path.exists():
        return default_thresholds, default_min_pixels

    data = json.loads(path.read_text())
    thresholds = data.get("pixel_thresholds", default_thresholds)
    min_pixels = data.get("min_defect_pixels_per_class", default_min_pixels)
    return [float(v) for v in thresholds], [int(v) for v in min_pixels]


if __name__ == "__main__":
    dummy = np.zeros((4, 16, 16), dtype=np.float32)
    dummy[1, 2:4, 2:4] = 0.9
    dummy[2, 4:12, 4:12] = 0.9
    result = postprocess_prediction(dummy, [0.5, 0.5, 0.5, 0.5], [10, 2, 20, 10])
    assert result.shape == dummy.shape
    assert result.dtype == np.uint8
    assert int(result[1].sum()) == 4
    assert int(result[2].sum()) == 64
    print("postprocess ok")
