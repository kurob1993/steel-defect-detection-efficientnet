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
    python predict_test.py --video path/to/video.mp4
    python predict_test.py --video path/to/video.mp4 --no-tta --frame-skip 5
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from train_config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MIN_DEFECT_PIXELS,
    MIN_DEFECT_PIXELS_PER_CLASS,
    PIXEL_THRESHOLD,
    PIXEL_THRESHOLDS,
    TTA_FLIPS,
    USE_TTA,
)
from model import load_trained_model
from postprocess import postprocess_prediction
from tta import tta_predict


COLORS = {
    0: (255, 80, 80),
    1: (80, 255, 80),
    2: (80, 160, 255),
    3: (255, 220, 80),
}


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

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = (gray - 0.485) / 0.229
    tensor = torch.from_numpy(gray[None, None, :, :]).to(device)
    return image, tensor


def analyze_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    cls_prob: np.ndarray,
    thresholds: list[float],
    min_pixels: list[int],
    class_names: dict[int, str],
    colors: dict[int, tuple[int, int, int]],
) -> dict:
    """Summarize model output and create an overlay for manual QA."""
    binary_masks = postprocess_prediction(mask, thresholds, min_pixels)
    h, w = image.shape[:2]
    overlay = image.copy()
    defects = []

    for class_idx in range(binary_masks.shape[0]):
        binary = cv2.resize(binary_masks[class_idx], (w, h), interpolation=cv2.INTER_NEAREST)
        area_px = int(binary.sum())
        if area_px <= 0:
            continue

        color = colors.get(class_idx, (0, 255, 255))
        color_layer = np.zeros_like(overlay)
        color_layer[binary > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)

        defects.append({
            "class_id": class_idx + 1,
            "class_name": class_names.get(class_idx, f"Class {class_idx + 1}"),
            "area_px": area_px,
            "area_ratio": area_px / float(h * w),
            "cls_prob": float(cls_prob[class_idx]),
        })

    return {
        "cls_probabilities": {
            class_names.get(i, f"Class {i + 1}"): float(cls_prob[i])
            for i in range(len(cls_prob))
        },
        "defects": defects,
        "has_defect": bool(defects),
        "overlay": overlay,
    }


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

    t_start = time.perf_counter()
    with torch.no_grad():
        if use_tta and USE_TTA:
            result = tta_predict(model, tensor, flips=TTA_FLIPS)
        else:
            result = model.predict(tensor)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    mask = result["mask"].squeeze(0).detach().cpu().numpy()
    cls_prob = result["cls_prob"].squeeze(0).detach().cpu().numpy()

    analyzed = analyze_prediction(
        image=image,
        mask=mask,
        cls_prob=cls_prob,
        thresholds=PIXEL_THRESHOLDS,
        min_pixels=MIN_DEFECT_PIXELS_PER_CLASS,
        class_names=CLASS_NAMES,
        colors=COLORS,
    )

    prediction = {
        "image": str(image_path),
        "cls_probabilities": analyzed["cls_probabilities"],
        "defects": analyzed["defects"],
        "has_defect": analyzed["has_defect"],
        "overlay_path": None,
        "elapsed_ms": round(elapsed_ms, 1),
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
    if result.get("elapsed_ms") is not None:
        print(f"  ⏱ Waktu predict: {result['elapsed_ms']} ms")
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
        "pixel_thresholds": PIXEL_THRESHOLDS,
        "min_defect_pixels_per_class": MIN_DEFECT_PIXELS_PER_CLASS,
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


def _draw_video_overlay(
    frame: np.ndarray,
    analyzed: dict,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Gambar overlay mask + label teks pada frame video."""
    out = analyzed["overlay"].copy()
    h, w = out.shape[:2]

    # ── Header: nomor frame & timestamp ──
    timestamp_sec = frame_idx / fps if fps > 0 else 0
    header = f"Frame {frame_idx}  |  {timestamp_sec:.2f}s"
    cv2.putText(out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Status defect ──
    status = "DEFECT DETECTED" if analyzed["has_defect"] else "Normal"
    color_status = (0, 60, 255) if analyzed["has_defect"] else (60, 200, 60)
    cv2.putText(out, status, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color_status, 2, cv2.LINE_AA)

    # ── Label per kelas yang terdeteksi ──
    y_offset = 82
    for defect in analyzed["defects"]:
        label = (
            f"{defect['class_name']}: "
            f"{defect['cls_prob']:.2f} | "
            f"{defect['area_px']}px ({defect['area_ratio']:.2%})"
        )
        cv2.putText(out, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 100), 1, cv2.LINE_AA)
        y_offset += 22

    return out


def predict_video(
    video_path: str,
    model_path: str,
    device: str = "auto",
    use_tta: bool = False,
    frame_skip: int = 1,
    output_dir: str = "outputs/predict_video",
    report_dir: str = "outputs/predict_test_reports",
    pixel_thresholds: list[float] | None = None,
    min_defect_pixels: list[int] | None = None,
) -> None:
    """Proses video frame per frame dan simpan video output dengan overlay defect.

    Args:
        video_path:         Path ke file video input (.mp4, .avi, dll).
        model_path:         Path ke checkpoint model.
        device:             'auto' | 'cuda' | 'cpu'.
        use_tta:            Aktifkan TTA (lambat untuk video, default False).
        frame_skip:         Proses 1 dari setiap N frame (1 = semua frame).
        output_dir:         Direktori simpan video output.
        report_dir:         Direktori simpan laporan JSON.
        pixel_thresholds:   Override threshold mask per class. Default lebih rendah
                            dari konfigurasi global agar lebih sensitif di video.
        min_defect_pixels:  Override min pixel per class. Default lebih rendah
                            dari konfigurasi global agar tidak terlalu ketat.
    """
    # ── Threshold video: default lebih sensitif dari gambar statis ──
    # Video bisa punya domain gap (lighting, resolusi, motion blur)
    # sehingga probabilitas model lebih rendah dari threshold training.
    _thresholds = pixel_thresholds if pixel_thresholds is not None else [t - 0.10 for t in PIXEL_THRESHOLDS]
    _min_pixels = min_defect_pixels if min_defect_pixels is not None else [max(1, m // 4) for m in MIN_DEFECT_PIXELS_PER_CLASS]

    device = resolve_device(device)
    model = load_model_once(model_path, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak dapat dibuka: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    output_video_path = out_path / f"{stem}_overlay.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (orig_w, orig_h))

    print(f"\n🎬 Video input    : {video_path}")
    print(f"   Resolusi      : {orig_w}x{orig_h}  |  FPS: {fps:.1f}  |  Total frame: {total_frames}")
    print(f"   Frame skip    : {frame_skip} (proses 1 dari setiap {frame_skip} frame)")
    print(f"   Thresholds    : {_thresholds}  (global: {PIXEL_THRESHOLDS})")
    print(f"   Min px/class  : {_min_pixels}  (global: {MIN_DEFECT_PIXELS_PER_CLASS})")
    print(f"   Output        : {output_video_path}\n")

    frame_reports = []
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    frames_with_defect = 0
    frames_without_defect = 0
    processed = 0
    last_analyzed: dict | None = None  # reuse hasil frame sebelumnya jika di-skip

    total_start = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # ── Preprocess frame (tanpa baca file, langsung dari array) ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gray = (gray - 0.485) / 0.229
            tensor = torch.from_numpy(gray[None, None, :, :]).to(device)

            t_start = time.perf_counter()
            with torch.no_grad():
                if use_tta and USE_TTA:
                    result = tta_predict(model, tensor, flips=TTA_FLIPS)
                else:
                    result = model.predict(tensor)
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

            mask = result["mask"].squeeze(0).detach().cpu().numpy()
            cls_prob = result["cls_prob"].squeeze(0).detach().cpu().numpy()

            last_analyzed = analyze_prediction(
                image=frame,
                mask=mask,
                cls_prob=cls_prob,
                thresholds=_thresholds,
                min_pixels=_min_pixels,
                class_names=CLASS_NAMES,
                colors=COLORS,
            )

            # ── Statistik ──
            if last_analyzed["has_defect"]:
                frames_with_defect += 1
                for defect in last_analyzed["defects"]:
                    class_counts[defect["class_name"]] += 1
            else:
                frames_without_defect += 1

            frame_reports.append({
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "elapsed_ms": elapsed_ms,
                "has_defect": last_analyzed["has_defect"],
                "cls_probabilities": last_analyzed["cls_probabilities"],
                "defects": last_analyzed["defects"],
            })
            processed += 1

            if processed % 50 == 0:
                elapsed_total = time.perf_counter() - total_start
                print(f"  Progress: frame {frame_idx}/{total_frames}  "
                      f"({processed} diproses, {elapsed_total:.1f}s)")

        # ── Tulis frame ke video output (selalu, skip atau tidak) ──
        if last_analyzed is not None:
            out_frame = _draw_video_overlay(frame, last_analyzed, frame_idx, fps)
        else:
            out_frame = frame
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    total_elapsed = round(time.perf_counter() - total_start, 1)

    # ── Ringkasan ──
    print(f"\n📊 Ringkasan Video")
    print(f"  Total frame          : {frame_idx}")
    print(f"  Frame diproses       : {processed}")
    print(f"  Frame dengan defect  : {frames_with_defect}")
    print(f"  Frame tanpa defect   : {frames_without_defect}")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} frame")
    print(f"  Total waktu          : {total_elapsed}s")
    print(f"\n💾 Video output : {output_video_path}")

    # ── Simpan laporan JSON ──
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    json_report_path = report_path / f"video_{stem}.json"
    save_json(json_report_path, {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "device": device,
        "use_tta": use_tta,
        "frame_skip": frame_skip,
        "fps": fps,
        "total_frames": frame_idx,
        "frames_processed": processed,
        "pixel_thresholds_used": _thresholds,
        "pixel_thresholds_global": PIXEL_THRESHOLDS,
        "min_defect_pixels_used": _min_pixels,
        "min_defect_pixels_global": MIN_DEFECT_PIXELS_PER_CLASS,
        "summary": {
            "frames_with_defect": frames_with_defect,
            "frames_without_defect": frames_without_defect,
            "class_counts": class_counts,
            "elapsed_sec": total_elapsed,
        },
        "frame_results": frame_reports,
    })
    print(f"💾 JSON report  : {json_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Test trained Two-Headed model")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None,
                        help="Path ke file video (.mp4, .avi, dll)")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Proses 1 dari setiap N frame (default: 1 = semua frame)")
    parser.add_argument(
        "--video-thresh", type=float, nargs=4, default=None,
        metavar=("C0", "C1", "C2", "C3"),
        help="Override pixel threshold per class untuk video "
             "(default: PIXEL_THRESHOLDS - 0.10). "
             "Contoh: --video-thresh 0.40 0.30 0.45 0.35"
    )
    parser.add_argument(
        "--video-min-px", type=int, nargs=4, default=None,
        metavar=("C0", "C1", "C2", "C3"),
        help="Override min pixel per class untuk video "
             "(default: MIN_DEFECT_PIXELS_PER_CLASS // 4). "
             "Contoh: --video-min-px 75 12 150 20"
    )
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--report-dir", type=str, default="outputs/predict_test_reports")
    args = parser.parse_args()

    device = resolve_device(args.device)
    use_tta = not args.no_tta

    if args.video is not None:
        predict_video(
            video_path=args.video,
            model_path=args.model,
            device=device,
            use_tta=use_tta,
            frame_skip=args.frame_skip,
            report_dir=args.report_dir,
            pixel_thresholds=args.video_thresh,
            min_defect_pixels=args.video_min_px,
        )
    elif args.image is not None:
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
            "pixel_thresholds": PIXEL_THRESHOLDS,
            "min_defect_pixels_per_class": MIN_DEFECT_PIXELS_PER_CLASS,
            "result": result,
        }
        save_json(single_report_path, single_report)
        print(f"💾 Single report: {single_report_path}")
    else:
        run_default_suite(
            model_path=args.model,
            device=device,
            use_tta=use_tta,
            report_dir=Path(args.report_dir),
            test_batch_size=args.test_batch_size,
        )


if __name__ == "__main__":
    main()
