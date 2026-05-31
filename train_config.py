"""Konfigurasi training — Pendekatan Juara 4 versi Production.

Arsitektur: Two-Headed Network
  Encoder: EfficientNet-B3
  Decoder: FPN
  Head: Segmentation + Classification (soft gating)
  Input: Grayscale 1 channel
  Loss: BCE + Focal
"""

import os
from pathlib import Path


def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None


def _detect_data_dir() -> Path:
    """Auto-detect dataset path for local or Kaggle.

    Priority:
    1. env DATA_DIR
    2. local ./data
    3. scan /kaggle/input/* for train.csv + train_images
    """
    env_path = os.getenv("DATA_DIR")
    if env_path:
        return Path(env_path)

    local_path = Path("data")
    if (local_path / "train.csv").exists() and (local_path / "train_images").exists():
        return local_path

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_csv in kaggle_input.glob("**/train.csv"):
            candidate = train_csv.parent
            if (candidate / "train_images").exists():
                return candidate

    return local_path


def _detect_model_save_dir() -> Path:
    """Auto-detect output directory for checkpoints."""
    env_path = os.getenv("MODEL_SAVE_DIR")
    if env_path:
        return Path(env_path)

    if _is_kaggle():
        return Path("/kaggle/working/models/segmentation")

    return Path("models/segmentation")


# ── Paths ──────────────────────────────────────────────
DATA_DIR = _detect_data_dir()
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"
MODEL_SAVE_DIR = _detect_model_save_dir()
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_SAVE_DIR / "fpn_efficientnet_b3_best.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "fpn_efficientnet_b3_last.pth"

# ── Image ──────────────────────────────────────────────
ORIGINAL_HEIGHT = 256
ORIGINAL_WIDTH = 1600
# Input: resize full image (256×1600) → (256×800)
# Model melihat seluruh gambar, tidak ada defect yang terpotong
INPUT_HEIGHT = 256
INPUT_WIDTH = 800
IN_CHANNELS = 1          # Grayscale
NUM_CLASSES = 4           # 4 defect classes (multi-label, tanpa background channel)

# ── Model ──────────────────────────────────────────────
ENCODER_NAME = "efficientnet-b3"
ENCODER_WEIGHTS = "imagenet"
DECODER_TYPE = "fpn"      # "fpn" | "unet"

# ── Two-Headed Network ─────────────────────────────────
SEG_LOSS_WEIGHT = 1.0     # Bobot loss segmentation
CLS_LOSS_WEIGHT = 0.5     # Bobot loss classification

# ── Training ───────────────────────────────────────────
BATCH_SIZE = 8             # T4 15GB + FP16 bisa batch 8 (256×800 grayscale)
ACCUMULATE_STEPS = 2       # Effective batch = 8 × 2 = 16
EPOCHS = 50
LEARNING_RATE = 3e-4       # Konservatif untuk EffB3 + FP16
WEIGHT_DECAY = 1e-4
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
VAL_SPLIT = 0.20           # 20% validation — lebih reliable untuk class imbalance
SEED = 42
USE_FP16 = True            # Mixed precision training
# RAM cache MATI untuk 12GB RAM (butuh ~8GB)
# Set 'true' kalau punya 16GB+ RAM untuk cache train set di /content/data
USE_RAM_CACHE = os.getenv("USE_RAM_CACHE", "false").lower() == "true"

# ── Loss ────────────────────────────────────────────────
# BCE pos_weight per class (menangani imbalance)
# Class 0 (Patch, 897): 2.0
# Class 1 (Crack, 247): 2.0
# Class 2 (Pitted, 5150): 1.0
# Class 3 (Scratch, 801): 1.5
POS_WEIGHT = [2.0, 2.0, 1.0, 1.5]

FOCAL_GAMMA = 2.0          # Focal loss focusing parameter
FOCAL_ALPHA = 0.5          # Lebih stabil untuk defect segmentation (bukan 0.25 dari object detection)
BCE_WEIGHT = 0.5           # Proporsi BCE dalam seg loss
FOCAL_WEIGHT = 0.5         # Proporsi Focal dalam seg loss

# ── Scheduler ──────────────────────────────────────────
SCHEDULER = "cosine"
PLATEAU_PATIENCE = 3

# ── Early stopping ─────────────────────────────────────
EARLY_STOP_PATIENCE = 7

# ── Inference / Post-processing ────────────────────────
# Backward-compatible scalar defaults.
PIXEL_THRESHOLD = 0.55     # Threshold binary mask
MIN_DEFECT_PIXELS = 128    # Minimum pixel per class mask
MIN_COMPONENT_SIZE = 150   # Minimum connected component size

# Class-specific defaults. Order: class 1, 2, 3, 4.
# Class 2 and 4 often contain thin defects, so their min-area filters start lower.
# Class 3 is frequent and noisier, so a stricter min-area filter is safer.
# Final values should be selected from validation/OOF with tune_thresholds.py.
PIXEL_THRESHOLDS = [0.50, 0.40, 0.55, 0.45]
MIN_DEFECT_PIXELS_PER_CLASS = [300, 50, 600, 80]

# ── TTA ────────────────────────────────────────────────
USE_TTA = True
TTA_FLIPS = ["horizontal"]  # Horizontal only — steel defect orientasi tertentu

# ── Augmentation ───────────────────────────────────────
USE_AUGMENTATION = True
# ❌ Tidak pakai: Rotate besar, ElasticTransform (tidak cocok untuk steel)
# ✅ Yang dipakai: HorizontalFlip, BrightnessContrast, GaussNoise, CLAHE

# ── Defect class names (index 0-3, sesuai output tensor [B, 4, H, W]) ──
CLASS_NAMES = {
    0: "Patch",           # Tambalan / bercak pada permukaan baja
    1: "Crack / Crazing", # Retakan / retak halus seperti jaring
    2: "Pitted Surface",  # Permukaan berlubang / bopeng akibat korosi
    3: "Scratch",         # Goresan pada permukaan baja
}
