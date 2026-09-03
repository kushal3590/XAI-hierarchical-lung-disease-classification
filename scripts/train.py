"""
Training script for the hierarchical multi-scale lung disease classifier.

Two-phase training strategy (appropriate for a 6GB VRAM laptop GPU):
  Phase 1 (warmup): backbone frozen, train only the multi-scale heads.
                     Fast, stabilizes the classifier before touching
                     pretrained weights.
  Phase 2 (fine-tune): unfreeze the last few EfficientNet blocks, train
                        everything with a lower LR. This is where most of
                        the accuracy gain over a naive flat baseline comes from.

Run from project root:
    python scripts/train.py
"""
import sys
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from metrics import accuracy_score, f1_score, recall_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders, FLAT_LABELS
from model import MultiScaleEfficientNet, combine_predictions

# ---------------------------------------------------------------------------
# Config — tune these based on your GPU / time budget
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"

BATCH_SIZE = 16          # fits comfortably in ~6GB VRAM with AMP at 224x224
NUM_WORKERS = 4           # lower to 0 if you hit DataLoader worker errors on Windows

EPOCHS_PHASE1 = 5        # backbone frozen, heads only
EPOCHS_PHASE2 = 15       # backbone partially unfrozen, full fine-tune
UNFREEZE_LAST_N_BLOCKS = 3

LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4

STAGE2_IGNORE_INDEX = -1

# ---------------------------------------------------------------------------
# Reproducibility (Phase 1 NFR1) — fixed seed, but now overridable via CLI
# so multiple seeds can be run for a seed-sensitivity check without
# overwriting each other's checkpoints/logs.
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Trades a little throughput for exact run-to-run reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(stage1_logits, stage2_logits, batch, criterion1, criterion2):
    loss1 = criterion1(stage1_logits, batch["stage1"])
    loss2 = criterion2(stage2_logits, batch["stage2"])
    return loss1 + loss2, loss1.item(), loss2.item()


def evaluate(model, loader, device, criterion1, criterion2):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["image"] = images

            stage1_logits, stage2_logits = model(images)
            loss, _, _ = compute_loss(stage1_logits, stage2_logits, batch, criterion1, criterion2)
            total_loss += loss.item()
            n_batches += 1

            flat_pred = combine_predictions(stage1_logits, stage2_logits)
            all_preds.extend(flat_pred.cpu().tolist())
            all_true.extend(batch["flat"].cpu().tolist())

    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    # TB is FLAT_LABELS index 2 — track its recall specifically since it's
    # the rare, clinically critical class (missed TB = worst-case outcome)
    tb_idx = FLAT_LABELS.index("Tuberculosis")
    tb_recall = recall_score(all_true, all_preds, labels=[tb_idx], average="micro", zero_division=0)

    avg_loss = total_loss / max(n_batches, 1)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "tb_recall": tb_recall,
        "y_true": all_true,
        "y_pred": all_preds,
    }


def run_epoch(model, loader, optimizer, scaler, device, criterion1, criterion2, epoch, phase, writer):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["image"] = images

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                stage1_logits, stage2_logits = model(images)
                loss, l1, l2 = compute_loss(stage1_logits, stage2_logits, batch, criterion1, criterion2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            stage1_logits, stage2_logits = model(images)
            loss, l1, l2 = compute_loss(stage1_logits, stage2_logits, batch, criterion1, criterion2)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [phase{phase} epoch {epoch}] batch {i}/{len(loader)} "
                  f"loss={loss.item():.4f} (stage1={l1:.4f} stage2={l2:.4f}) "
                  f"[{elapsed:.0f}s elapsed]")

    avg_loss = running_loss / len(loader)
    writer.add_scalar(f"train/loss_phase{phase}", avg_loss, epoch)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default 42). Use a different value "
                             "per run for a seed-sensitivity check.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional label appended to output folder names "
                             "(e.g. --tag masked -> outputs/checkpoints_masked/), "
                             "so experimental runs don't overwrite your existing "
                             "seed-42/123 results.")
    parser.add_argument("--mask-shortcut-regions", action="store_true",
                        help="Apply shortcut-region masking augmentation to the "
                             "TRAINING set only (randomly blanks the mediastinal "
                             "midline strip and top corners) — directly targets the "
                             "attention-shortcut pattern found in the Grad-CAM "
                             "analysis. val/test are never masked.")
    parser.add_argument("--stage1-pos-weight", type=float, default=1.0,
                        help="Class weight for the 'Abnormal' class in Stage 1's "
                             "loss (default 1.0 = unweighted). Values > 1.0 push "
                             "the model to reduce Pneumonia/Abnormal false "
                             "negatives, at some cost to Normal precision.")
    args = parser.parse_args()
    seed = args.seed

    set_seed(seed)  # NFR1: must run before model/optimizer/dataloader creation
    print(f"Random seed fixed at {seed} for reproducibility (NFR1)")
    if args.mask_shortcut_regions:
        print("Shortcut-region masking augmentation: ENABLED (train set only)")
    if args.stage1_pos_weight != 1.0:
        print(f"Stage 1 Abnormal class weight: {args.stage1_pos_weight}")

    # Seed-suffixed (and optionally tag-suffixed) output paths so experimental
    # runs don't overwrite your existing seed-42/123 results.
    seed_suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    suffix = seed_suffix + tag_suffix
    checkpoint_dir = ROOT / "outputs" / f"checkpoints{suffix}"
    log_dir = ROOT / "outputs" / f"logs{suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=seed,
        mask_shortcut_train=args.mask_shortcut_regions,
    )

    print("\nBuilding model...")
    model = MultiScaleEfficientNet(pretrained=True, freeze_backbone=True).to(device)

    if args.stage1_pos_weight != 1.0:
        stage1_weights = torch.tensor([1.0, args.stage1_pos_weight], device=device)
        criterion1 = nn.CrossEntropyLoss(weight=stage1_weights)
    else:
        criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(ignore_index=STAGE2_IGNORE_INDEX)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(log_dir))

    best_macro_f1 = -1.0

    # ---------------- Phase 1: heads only ----------------
    print(f"\n=== Phase 1: training heads only ({EPOCHS_PHASE1} epochs) ===")
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR_PHASE1
    )
    for epoch in range(1, EPOCHS_PHASE1 + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device,
                                criterion1, criterion2, epoch, phase=1, writer=writer)
        val_metrics = evaluate(model, val_loader, device, criterion1, criterion2)
        print(f"[Phase1 Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} val_tb_recall={val_metrics['tb_recall']:.4f}")
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("val/tb_recall", val_metrics["tb_recall"], epoch)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (macro_f1={best_macro_f1:.4f})")

    # ---------------- Phase 2: fine-tune backbone ----------------
    print(f"\n=== Phase 2: fine-tuning last {UNFREEZE_LAST_N_BLOCKS} backbone blocks "
          f"({EPOCHS_PHASE2} epochs) ===")
    model.set_backbone_trainable(True, unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR_PHASE2
    )
    total_epochs_so_far = EPOCHS_PHASE1
    for epoch in range(1, EPOCHS_PHASE2 + 1):
        global_epoch = total_epochs_so_far + epoch
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device,
                                criterion1, criterion2, global_epoch, phase=2, writer=writer)
        val_metrics = evaluate(model, val_loader, device, criterion1, criterion2)
        print(f"[Phase2 Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} val_tb_recall={val_metrics['tb_recall']:.4f}")
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], global_epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], global_epoch)
        writer.add_scalar("val/tb_recall", val_metrics["tb_recall"], global_epoch)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (macro_f1={best_macro_f1:.4f})")

    # ---------------- Final test-set evaluation ----------------
    print("\n=== Loading best checkpoint and evaluating on TEST set ===")
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate(model, test_loader, device, criterion1, criterion2)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test TB recall: {test_metrics['tb_recall']:.4f}")
    print("\nFull classification report:")
    print(classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        target_names=FLAT_LABELS, zero_division=0,
    ))

    writer.close()
    print(f"\nDone. Best model saved to {checkpoint_path}")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
