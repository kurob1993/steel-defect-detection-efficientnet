"""Test trained model pada sample train dan batch bertahap test set.

Default behavior:
- 5 gambar defect dari data/train_images
- 5 gambar normal dari data/train_images
- maksimal 100 gambar BELUM dites dari data/test_images
- simpan report batch ke outputs/predict_test_reports/
- simpan progress agar run berikutnya lanjut ke gambar test yang belum dites

Usage:
    python predict_test.py
    python predict_test.py --image path/to/image.jpg
    python predict_test.py --test-batch-size 100
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from app.services.inference_pipeline import (
    DEFAULT_COLORS,
    analyze_prediction,
    preprocess_image as shared_preprocess_image,
)
from train_config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MIN_DEFECT_PIXELS,
    PIXEL_THRESHOLD,
    TTA_FLIPS,
    USE_TTA,
)
from model import load_trained_model
from tta import tta_predict


COLORS = DEFAULT_COLORS


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_once(model_path: str, device: str):
    print("Loading model...")
    return load_trained_model(model_path, device=device)


def preprocess_image(image_path: str, device: str) -> tuple[object, torch.Tensor]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    tensor_np = shared_preprocess_image(
        image,
        target_width=INPUT_WIDTH,
        target_height=INPUT_HEIGHT,
    )
    tensor = torch.from_numpy(tensor_np).to(device)
    return image, tensor


def predict_single(
    image_path: str,
    model_path: str,
    device: str = "auto",
    use_tta: bool = True,
    model=None,
    save_overlay: bool = False,
    overlay_dir: str = "outputs/predict_test_overlays",
) -> dict:
    device = resolve_device(device)
    if model is None:
        model = load_model_once(model_path, device=device)

    image, tensor = preprocess_image(image_path, device)

    with torch.no_grad():
        if use_tta and USE_TTA:
            result = tta_predict(model, tensor, flips=TTA_FLIPS)
        else:
            result = model.predict(tensor)

    mask = result["mask"].squeeze(0).detach().cpu().numpy()
    cls_prob = result["cls_prob"].squeeze(0).detach().cpu().numpy()

    analyzed = analyze_prediction(
        image=image,
        mask=mask,
        cls_prob=cls_prob,
        threshold=PIXEL_THRESHOLD,
        min_defect_pixels=MIN_DEFECT_PIXELS,
        class_names=CLASS_NAMES,
        colors=COLORS,
    )

    prediction = {
        "image": str(image_path),
        "cls_probabilities": analyzed["cls_probabilities"],
        "defects": analyzed["defects"],
        "has_defect": analyzed["has_defect"],
        "overlay_path": None,
    }

    if save_overlay:
        output_dir = Path(overlay_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_path).stem}_overlay.jpg"
        cv2.imwrite(str(output_path), analyzed["overlay"])
        prediction["overlay_path"] = str(output_path)

    return prediction


def print_single_result(result: dict, gt_label: str | None = None):
    print(f"{'─' * 50}")
    print(f"  File:    {result['image']}")
    if gt_label is not None:
        print(f"  GT:      {gt_label}")
    print(f"  Defect:  {'✅ YA' if result['has_defect'] else '❌ Tidak'}")
    for name, prob in result["cls_probabilities"].items():
        print(f"  Cls {name}: {prob:.3f}")
    for defect in result["defects"]:
        print(
            f"  └─ {defect['class_name']}: {defect['area_px']}px "
            f"({defect['area_ratio']:.4%}), cls={defect['cls_prob']:.3f}"
        )
    if result.get("overlay_path"):
        print(f"  Overlay: {result['overlay_path']}")
    print()


def pick_train_samples(train_dir: Path, train_csv: Path) -> tuple[list[Path], list[Path], pd.DataFrame]:
    df = pd.read_csv(train_csv)
    defect_names = sorted(df["ImageId"].dropna().unique().tolist())
    all_train_names = sorted(p.name for p in train_dir.glob("*.jpg"))
    defect_set = set(defect_names)
    normal_names = [name for name in all_train_names if name not in defect_set]
    defect_samples = [train_dir / name for name in defect_names[:5]]
    normal_samples = [train_dir / name for name in normal_names[:5]]
    return defect_samples, normal_samples, df


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"tested_files": [], "runs": []}


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_default_suite(
    model_path: str,
    device: str,
    use_tta: bool,
    report_dir: Path,
    test_batch_size: int,
):
    train_dir = Path("data/train_images")
    train_csv = Path("data/train.csv")
    test_dir = Path("data/test_images")
    progress_path = report_dir / "test_images_progress.json"

    defect_samples, normal_samples, df = pick_train_samples(train_dir, train_csv)
    train_samples = [(p, "defect") for p in defect_samples] + [(p, "normal") for p in normal_samples]

    all_test_files = sorted(test_dir.glob("*.jpg"))
    progress = load_progress(progress_path)
    tested_files = set(progress.get("tested_files", []))
    pending_test_files = [p for p in all_test_files if p.name not in tested_files]
    current_batch = pending_test_files[:test_batch_size]

    if not train_samples:
        print("❌ Tidak ada sample train.")
        return
    if not all_test_files:
        print("❌ Tidak ada sample test.")
        return

    print(f"📌 test_images total:   {len(all_test_files)}")
    print(f"📌 sudah dites:        {len(tested_files)}")
    print(f"📌 belum dites:        {len(pending_test_files)}")
    print(f"📌 batch sekarang:     {len(current_batch)}\n")

    model = load_model_once(model_path, device=device)

    print("🔍 Testing 5 defect + 5 normal dari train_images\n")
    train_results = []
    train_start = time.perf_counter()
    for sample, gt_label in train_samples:
        result = predict_single(
            str(sample),
            model_path,
            device=device,
            use_tta=use_tta,
            model=model,
            save_overlay=True,
            overlay_dir="outputs/predict_test_overlays/train_samples",
        )
        gt_classes = []
        if gt_label == "defect":
            rows = df[df["ImageId"] == sample.name]
            gt_classes = sorted(rows["ClassId"].dropna().astype(int).unique().tolist())
        train_results.append(
            {
                "file": sample.name,
                "gt_label": gt_label,
                "gt_classes": gt_classes,
                "prediction": result,
            }
        )
        pretty_gt = gt_label if gt_label == "normal" else f"defect {gt_classes}"
        print_single_result(result, gt_label=pretty_gt)
    train_elapsed = round(time.perf_counter() - train_start, 1)

    test_results = []
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    images_with_defect = 0
    images_without_defect = 0

    if current_batch:
        print(f"🔍 Testing {len(current_batch)} gambar baru dari data/test_images\n")
        test_start = time.perf_counter()
        for idx, sample in enumerate(current_batch, start=1):
            result = predict_single(
                str(sample),
                model_path,
                device=device,
                use_tta=use_tta,
                model=model,
                save_overlay=False,
            )
            if result["has_defect"]:
                result = predict_single(
                    str(sample),
                    model_path,
                    device=device,
                    use_tta=use_tta,
                    model=model,
                    save_overlay=True,
                    overlay_dir="outputs/predict_test_overlays/test_batches",
                )
            test_results.append({
                "file": sample.name,
                "prediction": result,
            })
            if result["has_defect"]:
                images_with_defect += 1
                for defect in result["defects"]:
                    class_counts[defect["class_name"]] += 1
            else:
                images_without_defect += 1

            if idx % 25 == 0 or idx == len(current_batch):
                print(f"  Progress: {idx}/{len(current_batch)}")
        test_elapsed = round(time.perf_counter() - test_start, 1)
    else:
        print("✅ Semua gambar di data/test_images sudah pernah dites.")
        test_elapsed = 0.0

    if current_batch:
        print("\n📊 Ringkasan batch test_images")
        print(f"  Batch size:          {len(test_results)}")
        print(f"  Terdeteksi defect:   {images_with_defect}")
        print(f"  Tidak terdeteksi:    {images_without_defect}")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

    batch_index = len(progress.get("runs", [])) + 1
    batch_report_path = report_dir / f"predict_test_batch_{batch_index:04d}.json"
    report = {
        "batch_index": batch_index,
        "model_path": str(model_path),
        "device": device,
        "use_tta": use_tta,
        "tta_flips": TTA_FLIPS if use_tta and USE_TTA else [],
        "threshold": PIXEL_THRESHOLD,
        "min_defect_pixels": MIN_DEFECT_PIXELS,
        "train_summary": {
            "total": len(train_results),
            "defect_samples": len(defect_samples),
            "normal_samples": len(normal_samples),
            "elapsed_sec": train_elapsed,
        },
        "test_summary": {
            "requested_batch_size": test_batch_size,
            "tested_in_this_run": len(test_results),
            "already_tested_before_run": len(tested_files),
            "remaining_before_run": len(pending_test_files),
            "remaining_after_run": len(pending_test_files) - len(test_results),
            "images_with_defect": images_with_defect,
            "images_without_defect": images_without_defect,
            "class_counts": class_counts,
            "elapsed_sec": test_elapsed,
        },
        "train_results": train_results,
        "test_results": test_results,
    }
    save_json(batch_report_path, report)

    if current_batch:
        progress["tested_files"].extend([item["file"] for item in test_results])
        progress["runs"].append(
            {
                "batch_index": batch_index,
                "report_file": str(batch_report_path),
                "tested_count": len(test_results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        save_json(progress_path, progress)

    print(f"\n💾 Batch report:  {batch_report_path}")
    print(f"💾 Progress file: {progress_path}")


def main():
    parser = argparse.ArgumentParser(description="Test trained Two-Headed model")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--report-dir", type=str, default="outputs/predict_test_reports")
    args = parser.parse_args()

    device = resolve_device(args.device)
    use_tta = not args.no_tta

    if args.image is None:
        run_default_suite(
            model_path=args.model,
            device=device,
            use_tta=use_tta,
            report_dir=Path(args.report_dir),
            test_batch_size=args.test_batch_size,
        )
    else:
        result = predict_single(
            args.image,
            args.model,
            device=device,
            use_tta=use_tta,
            save_overlay=True,
        )
        print_single_result(result)

        report_dir = Path(args.report_dir)
        single_report_path = report_dir / f"single_{Path(args.image).stem}.json"
        single_report = {
            "model_path": str(args.model),
            "device": device,
            "use_tta": use_tta,
            "tta_flips": TTA_FLIPS if use_tta and USE_TTA else [],
            "threshold": PIXEL_THRESHOLD,
            "min_defect_pixels": MIN_DEFECT_PIXELS,
            "result": result,
        }
        save_json(single_report_path, single_report)
        print(f"💾 Single report: {single_report_path}")


if __name__ == "__main__":
    main()
