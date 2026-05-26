"""Two-Headed Network: Segmentation + Classification.

Encoder: EfficientNet-B3 (pretrained ImageNet)
Decoder: FPN
Heads: Segmentation (mask) + Classification (probability)
Soft Gating: final_mask = mask.sigmoid() * cls_prob.sigmoid()

Pendekatan Juara 4 versi Production.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from train_config import ENCODER_NAME, ENCODER_WEIGHTS, NUM_CLASSES, IN_CHANNELS


class TwoHeadedModel(nn.Module):
    """Two-Headed Segmentation + Classification model.

    Arsitektur:
        Input (grayscale)
            ↓
        Encoder (EfficientNet-B3)
            ↓
        FPN Decoder
            ↓
        ┌── Segmentation Head → mask (4 ch)
        └── Classification Head → prob (4)
                ↓
        Soft Gating: mask × prob → final mask
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        encoder_weights: str = ENCODER_WEIGHTS,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        decoder_type: str = "fpn",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Channel adapter: grayscale → 3 channel untuk pretrained encoder
        self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Backbone segmentation model — digunakan utuh sebagai segmentation head
        DecoderClass = smp.FPN if decoder_type == "fpn" else smp.Unet
        self.seg_model = DecoderClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        # Classification head menggunakan encoder features
        encoder_out_channels = self.seg_model.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(encoder_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Initialize cls_head
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, 1, H, W) grayscale

        Returns:
            dict dengan:
                - seg_logits: (B, 4, H, W) segmentation logits
                - cls_logits: (B, 4) classification logits
        """
        # Expand grayscale ke 3 channel
        if x.shape[1] == 1:
            x = self.channel_adapter(x)

        # Encoder sekali saja → features dipakai untuk SEG + CLS
        features = self.seg_model.encoder(x)

        # Decoder dari encoder features → segmentation
        decoder_output = self.seg_model.decoder(features)
        seg_logits = self.seg_model.segmentation_head(decoder_output)

        # Classification head dari bottleneck features
        cls_features = features[-1]  # Bottleneck features
        cls_logits = self.cls_head(cls_features)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
        }

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference dengan soft gating.

        Returns:
            dict dengan:
                - mask: (B, 4, H, W) final mask (soft gated)
                - cls_prob: (B, 4) classification probability
                - seg_prob: (B, 4, H, W) raw segmentation probability
        """
        output = self.forward(x)

        seg_prob = output["seg_logits"].sigmoid()
        cls_prob = output["cls_logits"].sigmoid()

        # Soft gating: mask × classification probability
        mask = seg_prob * cls_prob[:, :, None, None]

        return {
            "mask": mask,
            "cls_prob": cls_prob,
            "seg_prob": seg_prob,
        }


def create_model(num_classes: int = NUM_CLASSES) -> TwoHeadedModel:
    """Factory function untuk membuat model."""
    return TwoHeadedModel(num_classes=num_classes)


def load_trained_model(
    model_path: str,
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> TwoHeadedModel:
    """Load model terlatih untuk inference."""
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint.get("model", {})))
        model.load_state_dict(state_dict)
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model
    model = create_model()
    dummy = torch.randn(2, 1, 256, 512)

    output = model(dummy)
    print(f"Input:       {dummy.shape}")
    print(f"Seg logits:  {output['seg_logits'].shape}")
    print(f"Cls logits:  {output['cls_logits'].shape}")

    pred = model.predict(dummy)
    print(f"Mask (gated): {pred['mask'].shape}")
    print(f"Cls prob:     {pred['cls_prob'].shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:  {num_params:,}")
    print("✅ Two-Headed Model OK")
