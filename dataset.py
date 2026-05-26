"""Custom Dataset untuk Severstal Steel Defect Detection.

Resize full image (256×1600) → (256×800), grayscale, multi-class mask.
Model melihat seluruh gambar — tidak ada defect yang terpotong.
"""

import time
import numpy as np
import cv2
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train_config import (
    TRAIN_CSV,
    TRAIN_IMG_DIR,
    ORIGINAL_HEIGHT,
    ORIGINAL_WIDTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    IN_CHANNELS,
    NUM_CLASSES,
    USE_AUGMENTATION,
    USE_RAM_CACHE,
    VAL_SPLIT,
)


def rle_to_mask(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode RLE string ke binary mask."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if pd.isna(rle) or rle == "" or rle.strip() == "":
        return mask.reshape(shape, order="F")
    parts = list(map(int, str(rle).split()))
    for start, length in zip(parts[::2], parts[1::2]):
        mask[start - 1 : start - 1 + length] = 1
    return mask.reshape(shape, order="F")


def build_masks_for_image(
    image_id: str, df: pd.DataFrame, shape: tuple[int, int]
) -> np.ndarray:
    """Build 4-channel binary masks (one per class).

    Returns:
        np.ndarray shape (4, H, W) — satu channel per defect class
    """
    rows = df[df["ImageId"] == image_id]
    masks = np.zeros((NUM_CLASSES, shape[0], shape[1]), dtype=np.uint8)
    for _, row in rows.iterrows():
        class_id = int(row["ClassId"])
        if 1 <= class_id <= NUM_CLASSES:
            masks[class_id - 1] = rle_to_mask(row["EncodedPixels"], shape)
    return masks


def build_cls_labels(image_id: str, df: pd.DataFrame) -> np.ndarray:
    """Build classification labels: 1 jika class ada, 0 jika tidak.

    Returns:
        np.ndarray shape (4,) — binary per class
    """
    rows = df[df["ImageId"] == image_id]
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _, row in rows.iterrows():
        class_id = int(row["ClassId"])
        if 1 <= class_id <= NUM_CLASSES:
            labels[class_id - 1] = 1.0
    return labels


def get_transforms(mode: str = "train") -> A.Compose:
    """Get augmentation pipeline.

    Resize full image ke INPUT_HEIGHT × INPUT_WIDTH, lalu augmentasi.

    Steel-safe augmentations:
    ✅ HorizontalFlip, RandomBrightnessContrast, GaussNoise, CLAHE
    ❌ Rotate besar, ElasticTransform (tidak cocok untuk steel defect)
    """
    if mode == "train" and USE_AUGMENTATION:
        return A.Compose(
            [
                A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.4
                ),
                A.GaussNoise(p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                A.ToGray(p=1.0),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH, interpolation=cv2.INTER_LINEAR),
                A.ToGray(p=1.0),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ]
        )


class SeverstalDataset(Dataset):
    """Dataset untuk Severstal — Two-Headed Network.

    Full image resize (bukan random crop) — semua defect terlihat.
    Mendukung RAM cache untuk mempercepat training di Colab (I/O Google Drive lambat).
    """

    def __init__(
        self,
        image_ids: list[str],
        df: pd.DataFrame,
        mode: str = "train",
    ):
        self.image_ids = image_ids
        self.df = df
        self.mode = mode
        self.transforms = get_transforms(mode)
        self.original_shape = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)

        # Fast index: hindari df[df["ImageId"] == image_id] tiap __getitem__
        self._rle_index: dict[str, list[tuple[int, str]]] = {}
        self._cls_cache: dict[str, np.ndarray] = {}
        for row in self.df.itertuples(index=False):
            image_id = row.ImageId
            class_id = int(row.ClassId)
            rle = row.EncodedPixels
            if pd.isna(rle) or rle == "" or str(rle).strip() == "":
                continue
            self._rle_index.setdefault(image_id, []).append((class_id, rle))

        for image_id in self.image_ids:
            labels = np.zeros(NUM_CLASSES, dtype=np.float32)
            for class_id, _ in self._rle_index.get(image_id, []):
                if 1 <= class_id <= NUM_CLASSES:
                    labels[class_id - 1] = 1.0
            self._cls_cache[image_id] = labels

        # RAM cache: IMAGE ONLY. Jangan cache mask — mask cache >16GB dan bikin swap.
        self._image_cache: dict[str, np.ndarray] = {}
        if USE_RAM_CACHE and mode == "train":
            self._preload_cache()

    def _preload_cache(self):
        """Preload image saja ke RAM. Mask tidak di-cache agar tidak swap."""
        try:
            print(f"   📦 Caching {len(self.image_ids)} images ke RAM (image only)...")
            cache_start = time.time()
            success_count = 0
            total_bytes = 0
            for i, image_id in enumerate(self.image_ids):
                try:
                    image_path = TRAIN_IMG_DIR / image_id
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self._image_cache[image_id] = image
                        total_bytes += image.nbytes
                        success_count += 1
                except Exception as e:
                    print(f"   ⚠️  Skip {image_id}: {e}")

                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - cache_start
                    speed = (i + 1) / max(elapsed, 1e-6)
                    mem_gb = total_bytes / 1024**3
                    print(f"   {i+1}/{len(self.image_ids)} ({speed:.0f} img/s, {mem_gb:.1f}GB)")

            elapsed = time.time() - cache_start
            mem_gb = total_bytes / 1024**3
            print(f"   ✅ Image cache done! ({elapsed:.0f}s, {success_count}/{len(self.image_ids)} images, {mem_gb:.1f}GB)")
        except Exception as e:
            print(f"   ⚠️  Cache gagal: {e}")
            print(f"   📌 Melanjutkan tanpa cache")
            self._image_cache.clear()

    def _build_masks_fast(self, image_id: str) -> np.ndarray:
        """Build 4-channel mask dari prebuilt RLE index (cepat, tanpa pandas filter)."""
        masks = np.zeros((NUM_CLASSES, self.original_shape[0], self.original_shape[1]), dtype=np.uint8)
        for class_id, rle in self._rle_index.get(image_id, []):
            if 1 <= class_id <= NUM_CLASSES:
                masks[class_id - 1] = rle_to_mask(rle, self.original_shape)
        return masks

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple:
        image_id = self.image_ids[idx]

        if self._image_cache:
            image = self._image_cache[image_id]
        else:
            image_path = TRAIN_IMG_DIR / image_id
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self._build_masks_fast(image_id)
        cls_labels = self._cls_cache[image_id]

        # Apply transforms (resize, grayscale, normalize)
        masks_hwc = masks.transpose(1, 2, 0)  # (H, W, 4)

        transformed = self.transforms(image=image, mask=masks_hwc)
        image_tensor = transformed["image"]  # (3, H, W) float32 — ToGray = 3ch
        mask_tensor = transformed["mask"].permute(2, 0, 1).float()  # (4, H, W) float32

        return image_tensor, mask_tensor, torch.tensor(cls_labels)


def prepare_data() -> tuple[list[str], list[str], pd.DataFrame]:
    """Siapkan data: baca CSV, split train/val."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(TRAIN_CSV)
    all_image_names = {f.name for f in TRAIN_IMG_DIR.glob("*.jpg")}
    complete_ids = list(all_image_names)

    train_ids, val_ids = train_test_split(
        complete_ids,
        test_size=VAL_SPLIT,
        random_state=42,
        shuffle=True,
    )

    return train_ids, val_ids, df
