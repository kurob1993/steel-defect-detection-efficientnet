"""TTA (Test-Time Augmentation) — Horizontal Flip.

Pendekatan Juara 4: average predictions dari original + Hflip.
"""

import torch
import torch.nn.functional as F


def tta_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    flips: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Predict dengan TTA.

    Args:
        model: TwoHeadedModel
        image: (B, 1, H, W) grayscale tensor
        flips: list of flips to apply, e.g. ["horizontal"]

    Returns:
        Averaged predictions: mask, cls_prob, seg_prob
    """
    if flips is None:
        flips = ["horizontal"]

    all_masks = []
    all_cls_probs = []

    # Original
    pred = model.predict(image)
    all_masks.append(pred["mask"])
    all_cls_probs.append(pred["cls_prob"])

    # Horizontal flip
    if "horizontal" in flips:
        flipped = torch.flip(image, dims=[-1])  # flip width
        pred_flip = model.predict(flipped)
        # Flip back mask
        mask_flip_back = torch.flip(pred_flip["mask"], dims=[-1])
        all_masks.append(mask_flip_back)
        all_cls_probs.append(pred_flip["cls_prob"])

    # Vertical flip
    if "vertical" in flips:
        flipped = torch.flip(image, dims=[-2])  # flip height
        pred_flip = model.predict(flipped)
        mask_flip_back = torch.flip(pred_flip["mask"], dims=[-2])
        all_masks.append(mask_flip_back)
        all_cls_probs.append(pred_flip["cls_prob"])

    # Average
    avg_mask = torch.stack(all_masks).mean(dim=0)
    avg_cls = torch.stack(all_cls_probs).mean(dim=0)

    return {
        "mask": avg_mask,
        "cls_prob": avg_cls,
    }
