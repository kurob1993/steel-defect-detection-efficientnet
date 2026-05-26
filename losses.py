"""Loss functions untuk Two-Headed Network.

- Focal Loss (segmentation)
- BCE + Focal (segmentation)
- BCE (classification)
- Combined two-head loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_config import POS_WEIGHT, FOCAL_GAMMA, FOCAL_ALPHA, BCE_WEIGHT, FOCAL_WEIGHT


class FocalLoss(nn.Module):
    """Focal Loss untuk binary segmentation per class.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Membantu model fokus ke hard examples (defect kecil/sulit)
    dan mengurangi dominasi easy negatives (background).
    """

    def __init__(
        self,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: (B, 4, H, W) raw logits
            targets: (B, 4, H, W) binary targets
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # probability of correct class
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()


class SegmentationLoss(nn.Module):
    """Combined BCE + Focal loss untuk segmentation.

    Loss = bce_weight × BCE(pos_weight) + focal_weight × FocalLoss
    """

    def __init__(
        self,
        pos_weight: list[float] | None = None,
        bce_weight: float = BCE_WEIGHT,
        focal_weight: float = FOCAL_WEIGHT,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None

        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: (B, 4, H, W) raw logits
            targets: (B, 4, H, W) binary targets
        """
        # BCE dengan pos_weight (reshape untuk broadcast: (1,4,1,1))
        pw = self.pos_weight.view(1, -1, 1, 1).to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pw,
            reduction="mean",
        )

        # Focal loss
        focal = self.focal(logits, targets)

        return self.bce_weight * bce + self.focal_weight * focal


class ClassificationLoss(nn.Module):
    """BCE loss untuk classification head."""

    def __init__(self, pos_weight: list[float] | None = None):
        super().__init__()
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: (B, 4) raw logits
            targets: (B, 4) binary labels
        """
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pw,
            reduction="mean",
        )


class TwoHeadLoss(nn.Module):
    """Combined loss untuk two-headed network.

    total = seg_weight × SegmentationLoss + cls_weight × ClassificationLoss
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 0.5,
        pos_weight: list[float] | None = None,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss = SegmentationLoss(pos_weight=pos_weight)
        self.cls_loss = ClassificationLoss(pos_weight=pos_weight)

    def forward(
        self,
        seg_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            seg_logits: (B, 4, H, W) segmentation logits
            cls_logits: (B, 4) classification logits
            seg_targets: (B, 4, H, W) segmentation targets
            cls_targets: (B, 4) classification targets

        Returns:
            dict dengan total_loss, seg_loss, cls_loss
        """
        seg_loss = self.seg_loss(seg_logits, seg_targets)
        cls_loss = self.cls_loss(cls_logits, cls_targets)
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "cls_loss": cls_loss,
        }


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> dict[str, float]:
    """Hitung Dice score per class.

    Args:
        preds: (B, 4, H, W) binary predictions (after threshold)
        targets: (B, 4, H, W) binary targets
    """
    results = {}
    num_classes = preds.shape[1]

    for c in range(num_classes):
        pred_c = preds[:, c]
        target_c = targets[:, c]
        intersection = (pred_c * target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        results[f"dice_class_{c+1}"] = dice.item()

    # Mean dice
    dices = [results[f"dice_class_{c+1}"] for c in range(num_classes)]
    results["mean_dice"] = sum(dices) / len(dices)

    return results


if __name__ == "__main__":
    # Test losses
    seg_logits = torch.randn(2, 4, 128, 256)
    cls_logits = torch.randn(2, 4)
    seg_targets = torch.randint(0, 2, (2, 4, 128, 256)).float()
    cls_targets = torch.randint(0, 2, (2, 4)).float()

    loss_fn = TwoHeadLoss(pos_weight=[2.0, 2.0, 1.0, 1.5])
    result = loss_fn(seg_logits, cls_logits, seg_targets, cls_targets)

    print(f"Total loss: {result['total_loss'].item():.4f}")
    print(f"Seg loss:   {result['seg_loss'].item():.4f}")
    print(f"Cls loss:   {result['cls_loss'].item():.4f}")

    # Test dice
    preds = (seg_logits.sigmoid() > 0.5).float()
    dice = compute_dice(preds, seg_targets)
    print(f"Mean dice:  {dice['mean_dice']:.4f}")
    print("✅ Losses OK")
