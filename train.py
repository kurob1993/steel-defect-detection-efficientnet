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
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.amp import autocast, GradScaler

from tqdm import tqdm

from train_config import *
from dataset import SeverstalDataset, prepare_data
from model import create_model
from losses import TwoHeadLoss, compute_dice
from postprocess import postprocess_prediction


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
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Must be False for benchmark
    torch.backends.cudnn.benchmark = True       # Auto-tune untuk input size tetap (10-30% faster)


def get_distributed_context() -> dict:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": is_distributed,
        "is_main": rank == 0,
    }


def setup_distributed(device_arg: str | None, no_multi_gpu: bool) -> tuple[dict, str]:
    ctx = get_distributed_context()
    cuda_available = torch.cuda.is_available()

    if ctx["is_distributed"] and not no_multi_gpu and cuda_available:
        torch.cuda.set_device(ctx["local_rank"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = f"cuda:{ctx['local_rank']}"
        return ctx, device

    device = device_arg or ("cuda" if cuda_available else "cpu")
    return ctx, device


def cleanup_distributed(ctx: dict):
    if ctx["is_distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(value: float, device: str, ctx: dict) -> float:
    if not ctx["is_distributed"]:
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def broadcast_int(value: int, device: str, ctx: dict) -> int:
    if not ctx["is_distributed"]:
        return value
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=0)
    return int(tensor.item())


def reduce_metric_dict(metric_sums: dict[str, float], count: int, device: str, ctx: dict) -> dict:
    if ctx["is_distributed"]:
        ordered_keys = list(metric_sums.keys())
        tensor = torch.tensor([metric_sums[k] for k in ordered_keys] + [float(count)], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        count = int(tensor[-1].item())
        metric_sums = {k: tensor[i].item() for i, k in enumerate(ordered_keys)}

    denom = max(count, 1)
    return {k: v / denom for k, v in metric_sums.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler | None,
    accumulate_steps: int,
    show_progress: bool,
) -> dict:
    """Train satu epoch."""
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False, disable=not show_progress)
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
        "loss": total_loss,
        "seg_loss": total_seg_loss,
        "cls_loss": total_cls_loss,
        "num_batches": n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    scaler: GradScaler | None,
    show_progress: bool,
) -> dict:
    """Validasi dengan soft gating."""
    model.eval()
    total_loss = 0.0
    all_dice = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validation", leave=False, disable=not show_progress)
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
        gated_np = gated_mask.detach().cpu().numpy()
        preds_np = np.stack([
            postprocess_prediction(sample, PIXEL_THRESHOLDS, MIN_DEFECT_PIXELS_PER_CLASS)
            for sample in gated_np
        ])
        preds = torch.from_numpy(preds_np).to(device=device, dtype=torch.float32)
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
    return {"loss": total_loss, **all_dice, "num_batches": n}


def main():
    args = parse_args()
    seed_everything(SEED)

    ctx, device = setup_distributed(args.device, args.no_multi_gpu)
    is_cuda = str(device).startswith("cuda")
    use_fp16 = USE_FP16 and not args.no_fp16 and is_cuda
    scaler = GradScaler('cuda') if use_fp16 else None

    # ── Multi-GPU Detection ──────────────────────────────
    num_visible_gpus = torch.cuda.device_count()
    world_size = ctx["world_size"] if ctx["is_distributed"] else 1
    use_ddp = ctx["is_distributed"] and not args.no_multi_gpu and is_cuda
    use_data_parallel = world_size == 1 and num_visible_gpus > 1 and not args.no_multi_gpu and device == "cuda"
    use_multi_gpu = use_ddp or use_data_parallel

    if ctx["is_main"]:
        print(f"Device: {device}")
        if is_cuda:
            for i in range(num_visible_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
        if use_ddp:
            print(f"Multi-GPU: ✅ DDP x{world_size}")
        elif use_data_parallel:
            print(f"Multi-GPU: ✅ DataParallel x{num_visible_gpus}")
        else:
            print("Multi-GPU: ❌ Single GPU")
        print(f"FP16: {'✅ Ya' if use_fp16 else '❌ Tidak'}")

    # Data
    if ctx["is_main"]:
        print("\n📦 Loading dataset...")
    train_ids, val_ids, df = prepare_data()
    if ctx["is_main"]:
        print(f"   Train: {len(train_ids)} gambar")
        print(f"   Val:   {len(val_ids)} gambar")

    use_ram_cache = USE_RAM_CACHE and not use_ddp
    if ctx["is_main"] and USE_RAM_CACHE and use_ddp:
        print("   ⚠️  RAM cache dimatikan di DDP agar tidak menduplikasi ~11GB cache per proses")

    train_dataset = SeverstalDataset(train_ids, df, mode="train", use_ram_cache=use_ram_cache)
    val_dataset = SeverstalDataset(val_ids, df, mode="val", use_ram_cache=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=ctx["rank"], shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=ctx["rank"], shuffle=False) if use_ddp else None

    dataloader_workers = args.num_workers
    if use_ddp:
        dataloader_workers = max(1, args.num_workers // world_size)

    batch_size = args.batch_size
    if use_ddp:
        if args.batch_size < world_size:
            raise ValueError(f"Global batch-size {args.batch_size} harus >= jumlah GPU {world_size}")
        batch_size = max(1, args.batch_size // world_size)
        if ctx["is_main"] and args.batch_size % world_size != 0:
            print(f"   ⚠️  Global batch {args.batch_size} tidak habis dibagi {world_size}; memakai {batch_size} per GPU ({batch_size * world_size} global)")

    # DataLoader batch size = args.batch_size (FULL batch)
    # DataParallel akan otomatis split ke tiap GPU
    # JANGAN manual split — itu double-split!
    if ctx["is_main"]:
        if use_ddp:
            print(f"   DDP batch: global {args.batch_size} → {batch_size}/GPU × {world_size} proses")
            print(f"   DataLoader workers: total {args.num_workers} → {dataloader_workers}/proses")
        elif use_data_parallel:
            per_gpu = args.batch_size // num_visible_gpus
            print(f"   DataLoader batch: {args.batch_size} → DataParallel split: {per_gpu}/GPU")
        else:
            print(f"   Batch size: {args.batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=dataloader_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(dataloader_workers > 0),
        prefetch_factor=4 if dataloader_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=dataloader_workers,
        pin_memory=True,
        persistent_workers=(dataloader_workers > 0),
        prefetch_factor=4 if dataloader_workers > 0 else None,
    )

    # Model
    if ctx["is_main"]:
        print("\n🔧 Building Two-Headed Network")
        print(f"   Encoder:  {ENCODER_NAME}")
        print(f"   Decoder:  {DECODER_TYPE.upper()}")
        print(f"   Input:    {IN_CHANNELS}ch × {INPUT_HEIGHT}×{INPUT_WIDTH}")
    model = create_model()

    model = model.to(device)
    if use_ddp:
        model = DistributedDataParallel(model, device_ids=[ctx["local_rank"]], output_device=ctx["local_rank"])
    elif use_data_parallel:
        model = DataParallel(model, device_ids=list(range(num_visible_gpus)))
        if ctx["is_main"]:
            print(f"   Multi-GPU: DataParallel across {num_visible_gpus} GPUs")

    num_params = sum(p.numel() for p in model.parameters())
    if ctx["is_main"]:
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
        if ctx["is_main"]:
            print("   Optimizer: RAdam ✅")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if ctx["is_main"]:
            print("   Optimizer: AdamW (RAdam tidak tersedia)")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume
    start_epoch = 0
    best_val_dice = 0.0
    history = []

    if args.resume and Path(args.resume).exists():
        if ctx["is_main"]:
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
    global_batch = batch_size * world_size if use_ddp else args.batch_size
    if ctx["is_main"]:
        print(f"\n🚀 Training: {args.epochs} epochs, batch={global_batch}, accumulate={ACCUMULATE_STEPS}")
        print(f"   Effective batch: {global_batch * ACCUMULATE_STEPS}")
        if use_ddp:
            print(f"   DDP: {batch_size}/GPU × {world_size} GPUs")
        elif use_data_parallel:
            print(f"   DataParallel: {args.batch_size} → {args.batch_size // num_visible_gpus}/GPU × {num_visible_gpus} GPUs")
        print(f"   Loss: {BCE_WEIGHT}×BCE + {FOCAL_WEIGHT}×Focal (seg) + {CLS_LOSS_WEIGHT}×BCE (cls)")
        print("=" * 70)

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ACCUMULATE_STEPS, ctx["is_main"]
        )
        val_metrics = validate(model, val_loader, criterion, device, scaler, ctx["is_main"])

        train_metrics = reduce_metric_dict(
            {
                "loss": train_metrics["loss"],
                "seg_loss": train_metrics["seg_loss"],
                "cls_loss": train_metrics["cls_loss"],
            },
            train_metrics["num_batches"],
            device,
            ctx,
        )
        val_metrics = reduce_metric_dict(
            {k: v for k, v in val_metrics.items() if k != "num_batches"},
            val_metrics["num_batches"],
            device,
            ctx,
        )

        epoch_time = time.time() - epoch_start
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if ctx["is_main"]:
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

        improved = val_metrics["mean_dice"] > best_val_dice

        if ctx["is_main"] and improved:
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
                        "pixel_thresholds": PIXEL_THRESHOLDS,
                        "min_defect_pixels_per_class": MIN_DEFECT_PIXELS_PER_CLASS,
                        "use_soft_gating": True,
                    },
                },
                BEST_MODEL_PATH,
            )
            print(f"   ✅ Best! Dice={best_val_dice:.4f}")
        elif ctx["is_main"]:
            patience_counter += 1

        # Save last
        if ctx["is_main"]:
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
        if ctx["is_main"]:
            history_path = MODEL_SAVE_DIR / "training_history.json"
            history_path.write_text(json.dumps(history, indent=2))

        # Early stopping (sync dari rank-0 ke semua rank, bukan all-reduce)
        patience_counter = broadcast_int(patience_counter, device, ctx)
        if (epoch + 1) >= args.min_epochs and patience_counter >= args.early_stop_patience:
            if ctx["is_main"]:
                print(f"\n⏹️  Early stopping: tidak ada improvement {args.early_stop_patience} epochs")
            break

    if ctx["is_main"]:
        print("\n" + "=" * 70)
        print("🏁 Training selesai!")
        print(f"   Best Dice:  {best_val_dice:.4f}")
        print(f"   Model:      {BEST_MODEL_PATH}")

    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
