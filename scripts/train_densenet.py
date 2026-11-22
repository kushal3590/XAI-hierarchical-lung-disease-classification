"""
Trains the DenseNet-121 comparison model with the SAME recipe as scripts/train.py
(guide's Review-1 request: compare against a second backbone).

Nothing here modifies or overwrites the existing project. All outputs go to a
dedicated folder:

    outputs/densenet/<run_name>/best_model.pt
    outputs/densenet/<run_name>/logs/            (TensorBoard)
    outputs/densenet/<run_name>/run_info.json    (args + headline metrics)

    run_name = <variant>[_seed<N> if seed != 42][_<tag> if --tag given]

and the script refuses to overwrite an existing DenseNet checkpoint unless you
pass --overwrite.

Identical to train.py: manifests / splits, preprocessing, augmentation,
WeightedRandomSampler, batch size 16, mixed precision, AdamW, Phase 1 (5 epochs,
backbone frozen, LR 1e-3) then Phase 2 (15 epochs, last 3 backbone children
unfrozen, LR 1e-4), best checkpoint chosen on validation macro-F1, same seeding.
The only thing that changes is the backbone (EfficientNet-B0 -> DenseNet-121).

Two comparisons worth running (same seed for both sides):

  1) Original recipe vs your original EfficientNet result (seed 42, 90.3%):
       python scripts/train_densenet.py
  2) Improved recipe vs your final EfficientNet model (93.2%):
       python scripts/train_densenet.py --tag masked --mask-shortcut-regions --stage1-pos-weight 1.3

Optional: --variant flat trains the single 4-way-head version (does hierarchy
help on a different backbone too?).

Run from project root.
"""
import sys
import time
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

# numpy-only metrics shim (src/metrics.py) — avoids the sklearn/tensorboard
# native-DLL conflict that crashed train.py on this Windows machine.
from metrics import accuracy_score, f1_score, recall_score, classification_report

from dataset import get_dataloaders, FLAT_LABELS
from model import combine_predictions
from model_densenet import MultiScaleDenseNet, FlatMultiScaleDenseNet

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_ROOT = ROOT / "outputs" / "densenet"

# Same hyperparameters as scripts/train.py
BATCH_SIZE = 16
NUM_WORKERS = 4          # lower to 0 if you hit DataLoader worker errors on Windows
EPOCHS_PHASE1 = 3
EPOCHS_PHASE2 = 9
UNFREEZE_LAST_N_BLOCKS = 3
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
STAGE2_IGNORE_INDEX = -1
DEFAULT_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(variant, outputs, batch, crit):
    """Hierarchical: Stage 1 CE + Stage 2 CE (ignore_index for Normal rows).
    Flat: single 4-way CE. Returns (loss, part1, part2) for logging."""
    if variant == "hierarchical":
        s1_logits, s2_logits = outputs
        l1 = crit["stage1"](s1_logits, batch["stage1"])
        l2 = crit["stage2"](s2_logits, batch["stage2"])
        return l1 + l2, l1.item(), l2.item()
    loss = crit["flat"](outputs, batch["flat"])
    return loss, loss.item(), 0.0


def predict_labels(variant, outputs):
    """Final 4-class label. Hierarchical uses the same hard sequential rule as
    train.py (Stage 1 decides Normal/Abnormal, then Stage 2); flat uses argmax."""
    if variant == "hierarchical":
        s1_logits, s2_logits = outputs
        return combine_predictions(s1_logits, s2_logits)
    return outputs.argmax(dim=1)


def to_device(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["image"] = images
    return batch


def evaluate(model, loader, device, variant, crit):
    model.eval()
    all_preds, all_true = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(batch["image"])
            loss, _, _ = compute_loss(variant, outputs, batch, crit)
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(predict_labels(variant, outputs).cpu().tolist())
            all_true.extend(batch["flat"].cpu().tolist())

    tb_idx = FLAT_LABELS.index("Tuberculosis")
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(all_true, all_preds),
        "macro_f1": f1_score(all_true, all_preds, average="macro", zero_division=0),
        "tb_recall": recall_score(all_true, all_preds, labels=[tb_idx], average="micro", zero_division=0),
        "y_true": all_true,
        "y_pred": all_preds,
    }


def run_epoch(model, loader, optimizer, scaler, device, variant, crit, epoch, phase, writer):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch["image"])
                loss, l1, l2 = compute_loss(variant, outputs, batch, crit)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch["image"])
            loss, l1, l2 = compute_loss(variant, outputs, batch, crit)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            parts = f"(stage1={l1:.4f} stage2={l2:.4f}) " if variant == "hierarchical" else ""
            print(f"  [densenet {variant} phase{phase} epoch {epoch}] batch {i}/{len(loader)} "
                  f"loss={loss.item():.4f} {parts}[{time.time() - t0:.0f}s elapsed]")

    avg_loss = running_loss / len(loader)
    writer.add_scalar(f"train/loss_phase{phase}", avg_loss, epoch)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["hierarchical", "flat"], default="hierarchical",
                        help="hierarchical = same 2-stage framework on a DenseNet-121 backbone "
                             "(default); flat = single 4-way head.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag", type=str, default="",
                        help="Label appended to the run folder name (e.g. masked).")
    parser.add_argument("--mask-shortcut-regions", action="store_true",
                        help="Shortcut-region masking augmentation (train split only) — same "
                             "flag as train.py.")
    parser.add_argument("--stage1-pos-weight", type=float, default=1.0,
                        help="Weight of the 'Abnormal' class in Stage 1's loss (hierarchical "
                             "only) — same flag as train.py.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing DenseNet checkpoint of the same run name.")
    args = parser.parse_args()

    variant, seed = args.variant, args.seed
    if variant == "flat" and args.stage1_pos_weight != 1.0:
        print("Note: --stage1-pos-weight only applies to the hierarchical variant; ignoring it.")

    run_name = variant
    if seed != DEFAULT_SEED:
        run_name += f"_seed{seed}"
    if args.tag:
        run_name += f"_{args.tag}"
    run_dir = OUTPUT_ROOT / run_name
    ckpt_path = run_dir / "best_model.pt"

    if ckpt_path.exists() and not args.overwrite:
        print(f"A DenseNet checkpoint already exists at:\n  {ckpt_path}\n"
              f"Refusing to overwrite. Use a different --tag / --seed, or pass --overwrite.")
        sys.exit(1)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    print(f"DenseNet-121 | variant={variant} | seed={seed} | run folder: {run_dir}")
    if args.mask_shortcut_regions:
        print("Shortcut-region masking augmentation: ENABLED (train set only)")
    if variant == "hierarchical" and args.stage1_pos_weight != 1.0:
        print(f"Stage 1 Abnormal class weight: {args.stage1_pos_weight}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nBuilding dataloaders (same manifests/splits as the EfficientNet runs)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=seed,
        mask_shortcut_train=args.mask_shortcut_regions,
    )

    print("\nBuilding model...")
    if variant == "hierarchical":
        model = MultiScaleDenseNet(pretrained=True, freeze_backbone=True).to(device)
        if args.stage1_pos_weight != 1.0:
            w = torch.tensor([1.0, args.stage1_pos_weight], device=device)
            crit = {"stage1": nn.CrossEntropyLoss(weight=w)}
        else:
            crit = {"stage1": nn.CrossEntropyLoss()}
        crit["stage2"] = nn.CrossEntropyLoss(ignore_index=STAGE2_IGNORE_INDEX)
    else:
        model = FlatMultiScaleDenseNet(pretrained=True, freeze_backbone=True).to(device)
        crit = {"flat": nn.CrossEntropyLoss()}

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(run_dir / "logs"))
    best = {"macro_f1": -1.0, "epoch": 0}

    def fit(phase, n_epochs, epoch_offset, optimizer):
        for epoch in range(1, n_epochs + 1):
            global_epoch = epoch_offset + epoch
            train_loss = run_epoch(model, train_loader, optimizer, scaler, device, variant, crit,
                                   global_epoch, phase, writer)
            val = evaluate(model, val_loader, device, variant, crit)
            print(f"[Phase{phase} Epoch {epoch}] train_loss={train_loss:.4f} "
                  f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
                  f"val_macro_f1={val['macro_f1']:.4f} val_tb_recall={val['tb_recall']:.4f}")
            writer.add_scalar("val/accuracy", val["accuracy"], global_epoch)
            writer.add_scalar("val/macro_f1", val["macro_f1"], global_epoch)
            writer.add_scalar("val/tb_recall", val["tb_recall"], global_epoch)

            if val["macro_f1"] > best["macro_f1"]:
                best["macro_f1"], best["epoch"] = val["macro_f1"], global_epoch
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> New best model saved (macro_f1={best['macro_f1']:.4f})")

    # ---------------- Phase 1: heads only ----------------
    print(f"\n=== Phase 1: training heads only ({EPOCHS_PHASE1} epochs) ===")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_PHASE1)
    fit(1, EPOCHS_PHASE1, 0, optimizer)

    # ---------------- Phase 2: fine-tune last blocks ----------------
    print(f"\n=== Phase 2: fine-tuning last {UNFREEZE_LAST_N_BLOCKS} backbone children "
          f"({EPOCHS_PHASE2} epochs) ===")
    model.set_backbone_trainable(True, unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_PHASE2)
    fit(2, EPOCHS_PHASE2, EPOCHS_PHASE1, optimizer)

    # ---------------- Final test-set evaluation (default decision rule) ----------------
    print("\n=== Loading best checkpoint and evaluating on TEST set ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = evaluate(model, test_loader, device, variant, crit)
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test macro F1: {test['macro_f1']:.4f}")
    print(f"Test TB recall: {test['tb_recall']:.4f}")
    print("\nFull classification report:")
    print(classification_report(test["y_true"], test["y_pred"], target_names=FLAT_LABELS, zero_division=0))

    info = {
        "run_name": run_name,
        "args": vars(args),
        "best_val_macro_f1": best["macro_f1"],
        "best_epoch": best["epoch"],
        "test_accuracy_default_rule": test["accuracy"],
        "test_macro_f1_default_rule": test["macro_f1"],
        "test_tb_recall_default_rule": test["tb_recall"],
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    writer.close()
    print(f"\nDone. Best model: {ckpt_path}")
    if variant == "hierarchical":
        print(f"Next: python scripts/evaluate_densenet.py --tune"
              + (f" --seed {seed}" if seed != DEFAULT_SEED else "")
              + (f" --tag {args.tag}" if args.tag else ""))
    else:
        print("Next: python scripts/evaluate_densenet.py --variant flat"
              + (f" --seed {seed}" if seed != DEFAULT_SEED else "")
              + (f" --tag {args.tag}" if args.tag else ""))


if __name__ == "__main__":
    main()
