# Severstal Threshold + OOF Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first improvement stage for pushing Kaggle Severstal score toward 0.90 by adding class-specific threshold/min-area postprocessing and validation/OOF tuning.

**Architecture:** Keep the current EfficientNet-B3 FPN two-headed model unchanged. Add a reusable postprocessing layer used by validation, inference, and submission generation. Add an OOF/validation tuning script that searches per-class pixel thresholds and minimum component areas, then writes the best config to JSON for Kaggle inference.

**Tech Stack:** Python, PyTorch, OpenCV, NumPy, Pandas, Albumentations, segmentation-models-pytorch, Kaggle notebook workflow.

---

## Baseline

Current observed Kaggle score:

- Public: `0.89166`
- Private: `0.87125`

Current project setup:

- Model: `FPN + EfficientNet-B3`
- Input size: `256x800`
- Current global pixel threshold: `0.55`
- Current global min defect pixels: `128`
- TTA: horizontal flip
- Split: single random split

Main weakness to fix first:

- Global threshold is too coarse for four defect types.
- Private/public gap suggests validation/postprocess is not robust enough.
- Thin defects can be removed by `MIN_DEFECT_PIXELS = 128`.

---

## Target Output

After this plan, the repo should support:

1. Class-specific thresholds, for example:

```python
PIXEL_THRESHOLDS = [0.50, 0.40, 0.55, 0.45]
```

2. Class-specific min component areas, for example:

```python
MIN_DEFECT_PIXELS_PER_CLASS = [300, 50, 600, 80]
```

3. A reusable postprocessing function:

```python
postprocess_prediction(prob_masks, thresholds, min_pixels)
```

4. A tuning script:

```bash
python tune_thresholds.py --model models/segmentation/fpn_efficientnet_b3_best.pth --output models/segmentation/best_postprocess.json
```

5. Kaggle inference can load `best_postprocess.json` and use tuned values.

---

## Task 1: Add Backward-Compatible Postprocess Config

**Files:**

- Modify: `train_config.py`

**Step 1: Add class-specific config while keeping old scalar config**

Add below the existing inference settings:

```python
# ── Inference / Postprocessing ─────────────────────────────
# Backward-compatible scalar defaults.
PIXEL_THRESHOLD = 0.55
MIN_DEFECT_PIXELS = 128

# Class-specific defaults. Order: class 1, 2, 3, 4.
# These are starting points only; tune_thresholds.py should optimize them.
PIXEL_THRESHOLDS = [0.50, 0.40, 0.55, 0.45]
MIN_DEFECT_PIXELS_PER_CLASS = [300, 50, 600, 80]
```

If `PIXEL_THRESHOLD` and `MIN_DEFECT_PIXELS` already exist, replace that block instead of duplicating it.

**Step 2: Add comments for class behavior**

Add comments explaining:

- Class 2 and 4 are often thin, so lower min area.
- Class 3 is frequent and noisy, so higher min area is acceptable.
- Final values must come from validation/OOF tuning.

**Step 3: Smoke test import**

Run:

```bash
python - <<'PY'
from train_config import PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS
assert len(PIXEL_THRESHOLDS) == 4
assert len(MIN_DEFECT_PIXELS_PER_CLASS) == 4
print('config ok')
PY
```

Expected:

```text
config ok
```

**Step 4: Commit**

```bash
git add train_config.py
git commit -m "feat: add class-specific postprocess config"
```

---

## Task 2: Create Reusable Postprocessing Module

**Files:**

- Create: `postprocess.py`

**Step 1: Implement parameter normalization**

Create `postprocess.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def normalize_class_values(value: float | int | Sequence[float | int], num_classes: int) -> list[float]:
    """Return one value per class.

    Accepts either a scalar or a sequence with length == num_classes.
    """
    if isinstance(value, (int, float)):
        return [float(value)] * num_classes

    values = [float(v) for v in value]
    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} values, got {len(values)}")
    return values
```

**Step 2: Implement small component removal**

Add:

```python
def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area from a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    if min_area <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 1

    return cleaned
```

**Step 3: Implement prediction postprocess**

Add:

```python
def postprocess_prediction(
    prob_masks: np.ndarray,
    thresholds: float | Sequence[float],
    min_pixels: int | Sequence[int],
) -> np.ndarray:
    """Convert probability masks to cleaned binary masks.

    Args:
        prob_masks: Array with shape (C, H, W), values 0..1.
        thresholds: Scalar or one threshold per class.
        min_pixels: Scalar or one min component area per class.

    Returns:
        uint8 array with shape (C, H, W), values 0/1.
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
```

**Step 4: Add minimal CLI smoke test**

Add at bottom:

```python
if __name__ == "__main__":
    dummy = np.zeros((4, 16, 16), dtype=np.float32)
    dummy[1, 2:4, 2:4] = 0.9
    dummy[2, 4:12, 4:12] = 0.9
    result = postprocess_prediction(dummy, [0.5, 0.5, 0.5, 0.5], [10, 2, 20, 10])
    assert result.shape == dummy.shape
    assert result.dtype == np.uint8
    print("postprocess ok")
```

**Step 5: Run smoke test**

```bash
python postprocess.py
```

Expected:

```text
postprocess ok
```

**Step 6: Commit**

```bash
git add postprocess.py
git commit -m "feat: add reusable mask postprocessing"
```

---

## Task 3: Use Postprocessing in Validation

**Files:**

- Modify: `train.py`

**Step 1: Import new config and helper**

Update imports near the existing config imports:

```python
from train_config import PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS
from postprocess import postprocess_prediction
```

Keep `PIXEL_THRESHOLD` import if other code still uses it.

**Step 2: Replace validation thresholding**

Current validation uses roughly:

```python
preds = (gated_mask > PIXEL_THRESHOLD).float()
```

Replace with batch-wise postprocessing:

```python
gated_np = gated_mask.detach().cpu().numpy()
preds_np = np.stack([
    postprocess_prediction(sample, PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS)
    for sample in gated_np
])
preds = torch.from_numpy(preds_np).to(device=device, dtype=torch.float32)
```

Ensure `numpy as np` is imported.

**Step 3: Run syntax check**

```bash
python -m py_compile train.py postprocess.py
```

Expected: no output and exit code 0.

**Step 4: Optional short validation run**

If a checkpoint exists locally:

```bash
python train.py --epochs 1 --batch-size 2 --num-workers 0
```

Expected:

- Training starts successfully.
- Validation completes without shape/type error.

**Step 5: Commit**

```bash
git add train.py
git commit -m "feat: validate with class-specific postprocessing"
```

---

## Task 4: Use Postprocessing in Inference/Submission

**Files:**

- Modify: `tta.py` if it thresholds internally.
- Modify: submission inference code in `steel-defect-detection-v1.ipynb` or extract the logic to a `.py` script if easier.
- Modify: any helper in `predict_test.py` that applies `PIXEL_THRESHOLD` or `MIN_DEFECT_PIXELS`.

**Step 1: Locate inference thresholding**

Run:

```bash
rg "PIXEL_THRESHOLD|MIN_DEFECT_PIXELS|threshold=|area_px|EncodedPixels" . --glob '!submission.csv'
```

Expected: find all places that convert probabilities to binary/RLE.

**Step 2: Replace scalar thresholding with `postprocess_prediction`**

Use:

```python
from train_config import PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS
from postprocess import postprocess_prediction

binary_masks = postprocess_prediction(
    prob_masks,
    thresholds=PIXEL_THRESHOLDS,
    min_pixels=MIN_DEFECT_PIXELS_PER_CLASS,
)
```

**Step 3: Preserve RLE format**

The RLE encoder must still receive one binary mask per class with original submission shape.

Expected output columns remain:

```text
ImageId_ClassId,EncodedPixels
```

**Step 4: Run submission format sanity check**

```bash
python - <<'PY'
import pandas as pd
s = pd.read_csv('submission.csv')
assert list(s.columns) == ['ImageId_ClassId', 'EncodedPixels']
print(len(s), 'rows ok')
PY
```

Expected: row count matches Kaggle sample submission if `submission.csv` was regenerated.

**Step 5: Commit**

```bash
git add predict_test.py tta.py steel-defect-detection-v1.ipynb
# include only files actually modified
git commit -m "feat: apply class-specific postprocess during inference"
```

---

## Task 5: Create Threshold Tuning Script

**Files:**

- Create: `tune_thresholds.py`

**Step 1: Implement CLI arguments**

The script should support:

```bash
python tune_thresholds.py \
  --model models/segmentation/fpn_efficientnet_b3_best.pth \
  --output models/segmentation/best_postprocess.json \
  --batch-size 4 \
  --num-workers 2
```

Arguments:

```python
--model
--output
--batch-size
--num-workers
--threshold-grid
--min-area-grid
--device
```

Default grids:

```python
threshold_grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
min_area_grid = [0, 25, 50, 80, 128, 200, 300, 600, 1000]
```

**Step 2: Load validation split**

Reuse:

```python
from dataset import prepare_data, SeverstalDataset
```

Then create validation dataset only:

```python
_, val_ids, df = prepare_data()
val_dataset = SeverstalDataset(val_ids, df, mode="val", use_ram_cache=False)
```

**Step 3: Run model prediction on validation set**

Use:

```python
from model import load_trained_model
from tta import tta_predict
```

Store predictions and masks in memory if possible:

```python
all_probs = []
all_masks = []
```

For each batch:

- Get `prob_masks` from `model.predict()` or `tta_predict()`.
- Append predicted probability masks as `float16` or `float32` CPU arrays.
- Append ground-truth masks as `uint8` CPU arrays.

**Step 4: Implement Dice computation for numpy**

Add:

```python
def dice_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2 * inter + eps) / (denom + eps))
```

**Step 5: Tune each class independently**

For class index `0..3`:

- Try all threshold/min-area combinations.
- Apply `postprocess_prediction` to that class.
- Compute average Dice over validation images for that class.
- Select best pair.

Pseudo-code:

```python
best_thresholds = []
best_min_pixels = []
class_scores = []

for class_idx in range(4):
    best = (-1, None, None)
    for threshold in threshold_grid:
        for min_area in min_area_grid:
            scores = []
            for prob, target in zip(all_probs, all_masks):
                binary = postprocess_prediction(
                    prob[[class_idx]],
                    thresholds=[threshold],
                    min_pixels=[min_area],
                )[0]
                scores.append(dice_np(binary, target[class_idx]))
            mean_score = float(np.mean(scores))
            if mean_score > best[0]:
                best = (mean_score, threshold, min_area)

    class_scores.append(best[0])
    best_thresholds.append(best[1])
    best_min_pixels.append(best[2])
```

**Step 6: Write JSON output**

Output structure:

```json
{
  "pixel_thresholds": [0.5, 0.4, 0.55, 0.45],
  "min_defect_pixels_per_class": [300, 50, 600, 80],
  "class_scores": [0.0, 0.0, 0.0, 0.0],
  "mean_score": 0.0,
  "model": "models/segmentation/fpn_efficientnet_b3_best.pth"
}
```

**Step 7: Run script on a small grid first**

```bash
python tune_thresholds.py \
  --model models/segmentation/fpn_efficientnet_b3_best.pth \
  --output models/segmentation/best_postprocess.json \
  --threshold-grid 0.45,0.50,0.55 \
  --min-area-grid 50,128,300 \
  --batch-size 2 \
  --num-workers 0
```

Expected:

- Script finishes.
- JSON file is created.
- Printed best values per class.

**Step 8: Commit**

```bash
git add tune_thresholds.py models/segmentation/best_postprocess.json
git commit -m "feat: add validation threshold tuning"
```

---

## Task 6: Load Tuned Postprocess Config During Inference

**Files:**

- Create or modify: `postprocess.py`
- Modify: Kaggle inference code / `steel-defect-detection-v1.ipynb`

**Step 1: Add JSON loader to `postprocess.py`**

Add:

```python
import json
from pathlib import Path


def load_postprocess_config(path: str | Path | None = None) -> tuple[list[float], list[int]]:
    """Load tuned postprocess config, or fall back to train_config defaults."""
    from train_config import PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS

    if path is None:
        return list(PIXEL_THRESHOLDS), list(MIN_DEFECT_PIXELS_PER_CLASS)

    path = Path(path)
    if not path.exists():
        return list(PIXEL_THRESHOLDS), list(MIN_DEFECT_PIXELS_PER_CLASS)

    data = json.loads(path.read_text())
    thresholds = data.get("pixel_thresholds", PIXEL_THRESHOLDS)
    min_pixels = data.get("min_defect_pixels_per_class", MIN_DEFECT_PIXELS_PER_CLASS)
    return list(thresholds), [int(v) for v in min_pixels]
```

**Step 2: In Kaggle notebook, locate config JSON**

Search in likely dataset paths:

```python
POSTPROCESS_CONFIG_CANDIDATES = list(Path('/kaggle/input').glob('**/best_postprocess.json'))
POSTPROCESS_CONFIG_PATH = POSTPROCESS_CONFIG_CANDIDATES[0] if POSTPROCESS_CONFIG_CANDIDATES else None
```

Then:

```python
thresholds, min_pixels = load_postprocess_config(POSTPROCESS_CONFIG_PATH)
print('thresholds:', thresholds)
print('min pixels:', min_pixels)
```

**Step 3: Use loaded values during binary mask creation**

```python
binary_masks = postprocess_prediction(prob_masks, thresholds, min_pixels)
```

**Step 4: Commit**

```bash
git add postprocess.py steel-defect-detection-v1.ipynb
git commit -m "feat: load tuned postprocess config for kaggle inference"
```

---

## Task 7: Create Experiment Notes Template

**Files:**

- Create: `docs/experiments/severstal-score-log.md`

**Step 1: Create experiment log**

```markdown
# Severstal Score Log

| Date | Model | Resolution | Folds | Thresholds | Min Pixels | Public | Private | Notes |
|---|---|---:|---:|---|---|---:|---:|---|
| 2026-05-30 | FPN EfficientNet-B3 | 256x800 | 1 | global 0.55 | global 128 | 0.89166 | 0.87125 | baseline |
```

**Step 2: Add next experiment row after submission**

After generating a new Kaggle submission, record:

- exact checkpoint
- thresholds
- min pixels
- public/private score
- whether postprocess improved private score

**Step 3: Commit**

```bash
git add docs/experiments/severstal-score-log.md
git commit -m "docs: add severstal experiment score log"
```

---

## Task 8: Validation Checklist Before Kaggle Submit

Run these before uploading a new submission:

```bash
python -m py_compile train_config.py postprocess.py tune_thresholds.py train.py predict_test.py
```

If local images/checkpoint exist:

```bash
python postprocess.py
python tune_thresholds.py \
  --model models/segmentation/fpn_efficientnet_b3_best.pth \
  --output models/segmentation/best_postprocess.json \
  --threshold-grid 0.45,0.50,0.55 \
  --min-area-grid 50,128,300 \
  --batch-size 2 \
  --num-workers 0
```

Kaggle notebook checks:

```python
print('submission rows:', len(submission))
print('empty masks:', (submission['EncodedPixels'] == '').sum())
print('non-empty masks:', (submission['EncodedPixels'] != '').sum())
assert list(submission.columns) == ['ImageId_ClassId', 'EncodedPixels']
```

Expected row count for Severstal test set:

```text
22024
```

---

## Expected Impact

Likely gain from this stage:

- Conservative: `+0.003` to `+0.008`
- Good case: `+0.010` to `+0.015`

This alone may not guarantee private `0.90`. It is the foundation before larger improvements:

1. 5-fold ensemble
2. Full-width or higher-resolution training
3. Mixed architecture ensemble
4. OOF-tuned postprocess from all folds

---

## Next Plan After This

If this stage works, create a second plan:

```text
5-fold OOF training + ensemble inference
```

That second stage is the most realistic path from private `0.87125` toward `0.90`.
