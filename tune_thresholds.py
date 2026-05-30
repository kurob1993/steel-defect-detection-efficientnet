"""Tune class-specific threshold and min-area values on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeverstalDataset, prepare_data
from model import load_trained_model
from postprocess import postprocess_prediction
from train_config import (
    BEST_MODEL_PATH,
    MIN_DEFECT_PIXELS_PER_CLASS,
    NUM_CLASSES,
    PIXEL_THRESHOLDS,
    TTA_FLIPS,
    USE_TTA,
)
from tta import tta_predict


def parse_float_grid(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(float(v.strip())) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Severstal postprocess thresholds on validation split")
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--output", type=str, default="models/segmentation/best_postprocess.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--threshold-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.35,0.40,0.45,0.50,0.55,0.60,0.65"),
    )
    parser.add_argument(
        "--min-area-grid",
        type=parse_int_grid,
        default=parse_int_grid("0,25,50,80,128,200,300,600,1000"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA while collecting validation predictions")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def dice_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2 * intersection + eps) / (denom + eps))


@torch.no_grad()
def collect_validation_predictions(
    model_path: str,
    device: str,
    batch_size: int,
    num_workers: int,
    use_tta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _, val_ids, df = prepare_data()
    val_dataset = SeverstalDataset(val_ids, df, mode="val", use_ram_cache=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = load_trained_model(model_path, device=device)
    all_probs: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    for images, masks, _ in tqdm(val_loader, desc="Collect val predictions"):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        if use_tta and USE_TTA:
            result = tta_predict(model, images, flips=TTA_FLIPS)
        else:
            result = model.predict(images)

        all_probs.append(result["mask"].detach().cpu().numpy().astype(np.float32))
        all_masks.append(masks.detach().cpu().numpy().astype(np.uint8))

    return np.concatenate(all_probs, axis=0), np.concatenate(all_masks, axis=0)


def tune_class(
    probs: np.ndarray,
    targets: np.ndarray,
    class_idx: int,
    threshold_grid: list[float],
    min_area_grid: list[int],
) -> tuple[float, float, int]:
    best_score = -1.0
    best_threshold = float(PIXEL_THRESHOLDS[class_idx])
    best_min_area = int(MIN_DEFECT_PIXELS_PER_CLASS[class_idx])

    for threshold in threshold_grid:
        for min_area in min_area_grid:
            scores = []
            for prob, target in zip(probs, targets):
                binary = postprocess_prediction(
                    prob[[class_idx]],
                    thresholds=[threshold],
                    min_pixels=[min_area],
                )[0]
                scores.append(dice_np(binary, target[class_idx]))

            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = float(threshold)
                best_min_area = int(min_area)

    return best_score, best_threshold, best_min_area


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Threshold grid: {args.threshold_grid}")
    print(f"Min-area grid: {args.min_area_grid}")

    probs, targets = collect_validation_predictions(
        model_path=str(model_path),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tta=not args.no_tta,
    )
    print(f"Validation tensors: probs={probs.shape}, targets={targets.shape}")

    best_thresholds: list[float] = []
    best_min_pixels: list[int] = []
    class_scores: list[float] = []

    for class_idx in range(NUM_CLASSES):
        score, threshold, min_area = tune_class(
            probs=probs,
            targets=targets,
            class_idx=class_idx,
            threshold_grid=args.threshold_grid,
            min_area_grid=args.min_area_grid,
        )
        best_thresholds.append(threshold)
        best_min_pixels.append(min_area)
        class_scores.append(score)
        print(
            f"Class {class_idx + 1}: dice={score:.5f}, "
            f"threshold={threshold:.2f}, min_area={min_area}"
        )

    output = {
        "pixel_thresholds": best_thresholds,
        "min_defect_pixels_per_class": best_min_pixels,
        "class_scores": class_scores,
        "mean_score": float(np.mean(class_scores)),
        "model": str(model_path),
        "use_tta": bool(not args.no_tta and USE_TTA),
        "threshold_grid": args.threshold_grid,
        "min_area_grid": args.min_area_grid,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Mean Dice: {output['mean_score']:.5f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
