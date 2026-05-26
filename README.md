# Steel Defect Detection — EfficientNet-B3 + FPN

Solusi deteksi cacat baja berbasis deep learning untuk kompetisi [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) di Kaggle. Mengimplementasikan pendekatan Top-4 dengan arsitektur *Two-Headed Network* (segmentasi + klasifikasi).

## Arsitektur

```
Input (grayscale 256×800)
        ↓
Channel Adapter (1 → 3 ch)
        ↓
Encoder: EfficientNet-B3 (pretrained ImageNet)
        ↓
Decoder: FPN (Feature Pyramid Network)
        ↓
┌── Segmentation Head → mask logits (4 ch)
└── Classification Head → class prob (4)
            ↓
Soft Gating: mask × cls_prob → final mask
```

**Soft gating** membantu menekan false positive: jika classifier yakin tidak ada defect, mask segmentasi ikut ditekan.

## Kelas Defect

| Index | Nama | Karakteristik |
|-------|------|---------------|
| 0 | Patch | Area besar, mudah terdeteksi |
| 1 | Crack / Crazing | Retak tipis, jumlah sedikit |
| 2 | Pitted Surface | Permukaan berlubang, paling umum |
| 3 | Scratch | Goresan tipis memanjang |

## Struktur File

```
├── train_config.py     # Semua hyperparameter & konfigurasi path
├── dataset.py          # Dataset loader + RLE decoder + augmentasi
├── model.py            # TwoHeadedModel (EfficientNet-B3 + FPN)
├── losses.py           # Focal Loss + BCE + Combined two-head loss
├── tta.py              # Test-Time Augmentation (horizontal flip)
├── train.py            # Training script (single/multi-GPU, FP16, DDP)
├── predict_test.py     # Inferensi & visualisasi hasil prediksi
├── export_onnx.py      # Export model ke format ONNX
└── models/
    └── segmentation/   # Direktori simpan checkpoint
```

## Setup Dataset

Unduh dataset dari Kaggle dan letakkan di folder `data/`:

```
data/
├── train.csv
├── train_images/
└── test_images/
```

Atau set environment variable:
```bash
export DATA_DIR=/path/to/dataset
```

Untuk Kaggle Notebooks, path `/kaggle/input/` akan terdeteksi otomatis.

## Training

```bash
# Training default (50 epoch, batch 8, lr 3e-4)
python train.py

# Custom hyperparameter
python train.py --epochs 30 --batch-size 16 --lr 1e-4

# Lanjut dari checkpoint
python train.py --resume models/segmentation/fpn_efficientnet_b3_last.pth

# Single GPU (paksa)
python train.py --no-multi-gpu

# Multi-GPU dengan DDP (2 GPU)
torchrun --nproc_per_node=2 train.py
```

Checkpoint disimpan ke `models/segmentation/`:
- `fpn_efficientnet_b3_best.pth` — model dengan val Dice terbaik
- `fpn_efficientnet_b3_last.pth` — model epoch terakhir

## Konfigurasi Utama (`train_config.py`)

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `INPUT_HEIGHT` / `INPUT_WIDTH` | 256 / 800 | Resize dari 256×1600 |
| `BATCH_SIZE` | 8 | Cocok untuk T4 15GB + FP16 |
| `ACCUMULATE_STEPS` | 2 | Effective batch = 16 |
| `EPOCHS` | 50 | |
| `LEARNING_RATE` | 3e-4 | |
| `USE_FP16` | True | Mixed precision |
| `PIXEL_THRESHOLD` | 0.55 | Threshold binary mask |
| `MIN_DEFECT_PIXELS` | 128 | Filter mask terlalu kecil |
| `USE_TTA` | True | Horizontal flip TTA |

## Loss Function

```
L_total = L_seg + 0.5 × L_cls

L_seg = 0.5 × BCE(pos_weight) + 0.5 × FocalLoss(γ=2, α=0.5)
L_cls = BCE per class
```

`pos_weight` per kelas: `[2.0, 2.0, 1.0, 1.5]` — menangani class imbalance.

## Inferensi

```bash
# Prediksi pada sample train + batch test
python predict_test.py

# Prediksi satu gambar
python predict_test.py --image path/to/image.jpg

# Ubah ukuran batch test
python predict_test.py --test-batch-size 200
```

## Export ONNX

```bash
# Export dengan default path
python export_onnx.py

# Custom path
python export_onnx.py \
  --model models/segmentation/fpn_efficientnet_b3_best.pth \
  --output models/segmentation/fpn_efficientnet_b3.onnx \
  --opset 17
```

Output ONNX memiliki dua output:
- `mask`: soft-gated mask, shape `(B, 4, H, W)`
- `cls_prob`: classification probability, shape `(B, 4)`

## Dependensi Utama

```
torch >= 2.0
segmentation-models-pytorch
albumentations
opencv-python
pandas
numpy
tqdm
```

## Referensi

- [Severstal Steel Defect Detection — Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
