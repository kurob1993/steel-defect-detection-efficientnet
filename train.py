"""Training script untuk Two-Headed Network (Juara 4 versi Production).

Arsitektur:
  Encoder: EfficientNet-B3
  Decoder: FPN
  Head: Segmentation + Classification (soft gating)
  Input: Grayscale 256×800
  Loss: BCE + Focal (seg) + BCE (cls)

Multi-GPU:
  DataParallel — otomatis jika ada >1 GPU (cocok untuk Kaggle 2xT4)
  DDP — via torchrun untuk setup cluster

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 8 --lr 3e-4
    python train.py --resume models/segmentation/fpn_efficientnet_b3_last.pth
    python train.py --no-multi-gpu          # Force single GPU
    torchrun --nproc_per_node=2 train.py    # DDP mode
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.amp import autocast, GradScaler

from tqdm import tqdm

from train_config import *
from dataset import SeverstalDataset, prepare_data
from model import create_model
from losses import TwoHeadLoss, compute_dice


def parse_args():
    parser = argparse.ArgumentParser(description="Train Two-Headed Steel Defect Model (Juara 4)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Force single GPU")
    return parser.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Must be False for benchmark
    torch.backends.cudnn.benchmark = True       # Auto-tune untuk input size tetap (10-30% faster)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler | None,
    accumulate_steps: int,
) -> dict:
    """Train satu epoch."""
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks, cls_labels in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)

        # Forward (mixed precision)
        if scaler is not None:
            with autocast('cuda'):
                output = model(images)
                loss_dict = criterion(
                    output["seg_logits"], output["cls_logits"],
                    masks, cls_labels,
                )
                loss = loss_dict["total_loss"] / accumulate_steps
            scaler.scale(loss).backward()

            if (num_batches + 1) % accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            output = model(images)
            loss_dict = criterion(
                output["seg_logits"], output["cls_logits"],
                masks, cls_labels,
            )
            loss = loss_dict["total_loss"] / accumulate_steps
            loss.backward()

            if (num_batches + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Metrics
        total_loss += loss_dict["total_loss"].item()
        total_seg_loss += loss_dict["seg_loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()
        num_batches += 1

        pbar.set_postfix(
            loss=f"{loss_dict['total_loss'].item():.4f}",
            seg=f"{loss_dict['seg_loss'].item():.4f}",
            cls=f"{loss_dict['cls_loss'].item():.4f}",
        )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    scaler: GradScaler | None,
) -> dict:
    """Validasi dengan soft gating."""
    model.eval()
    total_loss = 0.0
    all_dice = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for images, masks, cls_labels in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)

        if scaler is not None:
            with autocast('cuda'):
                output = model(images)
                loss_dict = criterion(
                    output["seg_logits"], output["cls_logits"],
                    masks, cls_labels,
                )
        else:
            output = model(images)
            loss_dict = criterion(
                output["seg_logits"], output["cls_logits"],
                masks, cls_labels,
            )

        total_loss += loss_dict["total_loss"].item()

        # Soft gating dice
        seg_prob = output["seg_logits"].sigmoid()
        cls_prob = output["cls_logits"].sigmoid()
        gated_mask = seg_prob * cls_prob[:, :, None, None]
        preds = (gated_mask > PIXEL_THRESHOLD).float()
        batch_dice = compute_dice(preds, masks)
        for k, v in batch_dice.items():
            if k not in all_dice:
                all_dice[k] = 0.0
            all_dice[k] += v

        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss_dict['total_loss'].item():.4f}",
            dice=f"{batch_dice['mean_dice']:.4f}",
        )

    n = max(num_batches, 1)
    avg_dice = {k: v / n for k, v in all_dice.items()}
    return {"loss": total_loss / n, **avg_dice}


def main():
    args = parse_args()
    seed_everything(SEED)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = USE_FP16 and not args.no_fp16 and device == "cuda"
    scaler = GradScaler('cuda') if use_fp16 else None

    # ── Multi-GPU Detection ──────────────────────────────
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1 and not args.no_multi_gpu and device == "cuda"

    print(f"Device: {device}")
    if device == "cuda":
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    print(f"Multi-GPU: {'✅ DataParallel x' + str(num_gpus) if use_multi_gpu else '❌ Single GPU'}")
    print(f"FP16: {'✅ Ya' if use_fp16 else '❌ Tidak'}")

    # Data
    print("\n📦 Loading dataset...")
    train_ids, val_ids, df = prepare_data()
    print(f"   Train: {len(train_ids)} gambar")
    print(f"   Val:   {len(val_ids)} gambar")

    train_dataset = SeverstalDataset(train_ids, df, mode="train")
    val_dataset = SeverstalDataset(val_ids, df, mode="val")

    # DataLoader batch size = args.batch_size (FULL batch)
    # DataParallel akan otomatis split ke tiap GPU
    # JANGAN manual split — itu double-split!
    if use_multi_gpu:
        per_gpu = args.batch_size // num_gpus
        print(f"   DataLoader batch: {args.batch_size} → DataParallel split: {per_gpu}/GPU")
    else:
        print(f"   Batch size: {args.batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # FULL batch, DataParallel auto-split
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=8 if args.num_workers > 0 else None,  # Prefetch more batches untuk avoid GPU starvation
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # FULL batch
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=8 if args.num_workers > 0 else None,  # Prefetch more batches untuk avoid GPU starvation
    )

    # Model
    print("\n🔧 Building Two-Headed Network")
    print(f"   Encoder:  {ENCODER_NAME}")
    print(f"   Decoder:  {DECODER_TYPE.upper()}")
    print(f"   Input:    {IN_CHANNELS}ch × {INPUT_HEIGHT}×{INPUT_WIDTH}")
    model = create_model()

    # ── Multi-GPU: DataParallel ──────────────────────────
    if use_multi_gpu:
        model = DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"   Multi-GPU: DataParallel across {num_gpus} GPUs")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Params:   {num_params:,}")

    # Loss
    criterion = TwoHeadLoss(
        seg_weight=SEG_LOSS_WEIGHT,
        cls_weight=CLS_LOSS_WEIGHT,
        pos_weight=POS_WEIGHT,
    )

    # Optimizer: RAdam
    try:
        from torch.optim import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("   Optimizer: RAdam ✅")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("   Optimizer: AdamW (RAdam tidak tersedia)")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume
    start_epoch = 0
    best_val_dice = 0.0
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"\n📂 Resuming dari {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Handle DataParallel wrapped state_dict
            if use_multi_gpu and not any(k.startswith("module.") for k in state_dict):
                state_dict = {"module." + k: v for k, v in state_dict.items()}
            elif not use_multi_gpu and any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Training loop
    print(f"\n🚀 Training: {args.epochs} epochs, batch={args.batch_size}, accumulate={ACCUMULATE_STEPS}")
    print(f"   Effective batch: {args.batch_size * ACCUMULATE_STEPS}")
    if use_multi_gpu:
        print(f"   DataParallel: {args.batch_size} → {args.batch_size // num_gpus}/GPU × {num_gpus} GPUs")
    print(f"   Loss: {BCE_WEIGHT}×BCE + {FOCAL_WEIGHT}×Focal (seg) + {CLS_LOSS_WEIGHT}×BCE (cls)")
    print("=" * 70)

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ACCUMULATE_STEPS
        )
        val_metrics = validate(model, val_loader, criterion, device, scaler)

        epoch_time = time.time() - epoch_start
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mean_dice": val_metrics["mean_dice"],
            "lr": current_lr,
            "time": round(epoch_time, 1),
        })

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Dice: {val_metrics['mean_dice']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.0f}s"
        )

        # Save best — unwrap DataParallel for clean state_dict
        model_state = model.module.state_dict() if isinstance(model, (DataParallel, DistributedDataParallel)) else model.state_dict()

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "num_classes": NUM_CLASSES,
                    "in_channels": IN_CHANNELS,
                    "input_size": (INPUT_HEIGHT, INPUT_WIDTH),
                    "config": {
                        "encoder": ENCODER_NAME,
                        "decoder": DECODER_TYPE,
                        "pixel_threshold": PIXEL_THRESHOLD,
                        "min_defect_pixels": MIN_DEFECT_PIXELS,
                        "use_soft_gating": True,
                    },
                },
                BEST_MODEL_PATH,
            )
            print(f"   ✅ Best! Dice={best_val_dice:.4f}")
        else:
            patience_counter += 1

        # Save last
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
            },
            LAST_MODEL_PATH,
        )

        # Save history
        history_path = MODEL_SAVE_DIR / "training_history.json"
        history_path.write_text(json.dumps(history, indent=2))

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping: tidak ada improvement {EARLY_STOP_PATIENCE} epochs")
            break

    print("\n" + "=" * 70)
    print("🏁 Training selesai!")
    print(f"   Best Dice:  {best_val_dice:.4f}")
    print(f"   Model:      {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
