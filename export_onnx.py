"""Export trained Two-Headed model ke ONNX.

Output ONNX:
- mask: soft-gated mask, shape (B, 4, H, W)
- cls_prob: classification probability, shape (B, 4)

Usage:
    python export_onnx.py
    python export_onnx.py --model models/segmentation/fpn_efficientnet_b3_best.pth
    python export_onnx.py --output models/segmentation/fpn_efficientnet_b3.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import load_trained_model
from train_config import BEST_MODEL_PATH, INPUT_HEIGHT, INPUT_WIDTH


class OnnxWrapper(torch.nn.Module):
    """Wrap model.predict() agar output ONNX berupa tensor, bukan dict."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor):
        result = self.model.predict(x)
        return result["mask"], result["cls_prob"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument(
        "--output",
        type=str,
        default="models/segmentation/fpn_efficientnet_b3.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("onnx belum terinstall. Jalankan: pip install onnx") from exc

    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        raise SystemExit(f"Model tidak ditemukan: {model_path}")

    print(f"Loading model: {model_path}")
    model = load_trained_model(str(model_path), device=args.device)
    wrapper = OnnxWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["mask", "cls_prob"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "mask": {0: "batch", 2: "height", 3: "width"},
                "cls_prob": {0: "batch"},
            },
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX saved: {output_path}")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        session = ort.InferenceSession(str(output_path), providers=providers)
        outputs = session.run(None, {"input": dummy.detach().cpu().numpy()})
        print(f"✅ ONNX verified: mask={outputs[0].shape}, cls_prob={outputs[1].shape}")
        print(f"Providers: {providers}")
    except ImportError:
        print("ℹ️ onnxruntime tidak terinstall, skip runtime verification")


if __name__ == "__main__":
    main()
