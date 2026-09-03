"""
Trains the FLAT baseline classifier — same multi-scale EfficientNet-B0
backbone, same data, same two-phase schedule as scripts/train.py, but a
single 4-way head instead of the two hierarchical stages.

This isolates the hierarchical-vs-flat variable for a fair comparison,
needed for your report's Comparative Analysis / Expected Contributions
section (quantitatively demonstrating the hierarchical approach's benefit
on YOUR data, not just citing other papers' numbers on different datasets).

Run from project root:
    python scripts/train_baseline.py
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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

from dataset import get_dataloaders, FLAT_LABELS
from model import FlatMultiScaleEfficientNet

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"

# Same hyperparameters as train.py — fair comparison requires matching schedule
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 15
UNFREEZE_LAST_N_BLOCKS = 3
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4

# ---------------------------------------------------------------------------
# Reproducibility (Phase 1 NFR1) — overridable via CLI. IMPORTANT: always run
# this with the SAME --seed value as train.py for a given comparison — the
# McNemar's test assumes any prediction difference reflects the architecture
# change, not incidental differences in sample draw order or augmentation.
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["flat"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    tb_idx = FLAT_LABELS.index("Tuberculosis")
    tb_recall = recall_score(all_true, all_preds, labels=[tb_idx], average="micro", zero_division=0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "tb_recall": tb_recall,
        "y_true": all_true,
        "y_pred": all_preds,
    }


def run_epoch(model, loader, optimizer, scaler, device, criterion, epoch, phase, writer):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["flat"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [baseline phase{phase} epoch {epoch}] batch {i}/{len(loader)} "
                  f"loss={loss.item():.4f} [{elapsed:.0f}s elapsed]")

    avg_loss = running_loss / len(loader)
    writer.add_scalar(f"train/loss_phase{phase}", avg_loss, epoch)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default 42). MUST match the seed "
                             "used for train.py in the same comparison.")
    args = parser.parse_args()
    seed = args.seed

    set_seed(seed)  # NFR1: must run before model/optimizer/dataloader creation
    print(f"Random seed fixed at {seed} for reproducibility (NFR1) — "
          f"make sure this matches the hierarchical model's seed")

    suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    checkpoint_dir = ROOT / "outputs" / f"checkpoints{suffix}"
    log_dir = ROOT / "outputs" / f"logs_baseline{suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "baseline_flat_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nBuilding dataloaders (same weighted sampler as the hierarchical model)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=seed
    )

    print("\nBuilding FLAT baseline model...")
    model = FlatMultiScaleEfficientNet(pretrained=True, freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(log_dir))

    best_macro_f1 = -1.0

    print(f"\n=== Phase 1: training head only ({EPOCHS_PHASE1} epochs) ===")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_PHASE1)
    for epoch in range(1, EPOCHS_PHASE1 + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, criterion, epoch, 1, writer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        print(f"[Phase1 Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} val_tb_recall={val_metrics['tb_recall']:.4f}")
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("val/tb_recall", val_metrics["tb_recall"], epoch)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best baseline model saved (macro_f1={best_macro_f1:.4f})")

    print(f"\n=== Phase 2: fine-tuning last {UNFREEZE_LAST_N_BLOCKS} backbone blocks "
          f"({EPOCHS_PHASE2} epochs) ===")
    model.set_backbone_trainable(True, unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_PHASE2)
    for epoch in range(1, EPOCHS_PHASE2 + 1):
        global_epoch = EPOCHS_PHASE1 + epoch
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, criterion, global_epoch, 2, writer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        print(f"[Phase2 Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} val_tb_recall={val_metrics['tb_recall']:.4f}")
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], global_epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], global_epoch)
        writer.add_scalar("val/tb_recall", val_metrics["tb_recall"], global_epoch)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best baseline model saved (macro_f1={best_macro_f1:.4f})")

    print("\n=== Loading best baseline checkpoint and evaluating on TEST set ===")
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate(model, test_loader, device, criterion)
    print(f"Baseline test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Baseline test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Baseline test TB recall: {test_metrics['tb_recall']:.4f}")
    print("\nBaseline classification report:")
    report = classification_report(
        test_metrics["y_true"], test_metrics["y_pred"], target_names=FLAT_LABELS, zero_division=0,
    )
    print(report)

    eval_dir = ROOT / "outputs" / f"evaluation{suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "baseline_classification_report.txt", "w") as f:
        f.write(report)

    writer.close()
    print(f"\nDone. Best baseline model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
