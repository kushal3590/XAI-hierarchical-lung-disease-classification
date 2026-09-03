"""
Tunes Stage 1's Normal/Abnormal decision threshold on the VALIDATION set
(never the test set — that would leak test information into a "trained"
choice) to find a better operating point than the default 0.5 argmax cutoff.

Rationale: per-class AUC is already 0.98+ across all 4 classes on this
model, meaning it separates Abnormal-vs-Normal well in principle — the 91%
overall accuracy is being held down mainly by Pneumonia recall (81%),
where the default 0.5 cutoff is too conservative. Lowering the threshold
trades some Normal precision for higher Pneumonia/overall recall. This
script finds a good trade-off empirically, with NO retraining required.

Run from project root (after evaluate.py / a trained checkpoint exists):
    python scripts/tune_stage1_threshold.py
    python scripts/tune_stage1_threshold.py --seed 123
    python scripts/tune_stage1_threshold.py --metric accuracy   # optimize for a different metric
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

from dataset import get_dataloaders, FLAT_LABELS
from model import MultiScaleEfficientNet, combine_predictions_thresholded

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
DEFAULT_SEED = 42

BATCH_SIZE = 16
THRESHOLDS = np.arange(0.15, 0.55, 0.025)  # sweep below the default 0.5


def collect_stage_logits(model, loader, device):
    """Runs the model once and caches raw logits — so we can re-score every
    threshold WITHOUT re-running the network each time."""
    all_stage1_logits, all_stage2_logits, all_true = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            s1_logits, s2_logits = model(images)
            all_stage1_logits.append(s1_logits.cpu())
            all_stage2_logits.append(s2_logits.cpu())
            all_true.extend(batch["flat"].tolist())
    return (torch.cat(all_stage1_logits), torch.cat(all_stage2_logits), np.array(all_true))


def sweep_thresholds(stage1_logits, stage2_logits, y_true, thresholds):
    pneu_idx = FLAT_LABELS.index("Pneumonia")
    rows = []
    for t in thresholds:
        y_pred = combine_predictions_thresholded(stage1_logits, stage2_logits, abnormal_threshold=t).numpy()
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pneu_recall = recall_score(y_true, y_pred, labels=[pneu_idx], average="micro", zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(FLAT_LABELS))))
        rows.append({
            "threshold": round(float(t), 3),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "pneumonia_recall": pneu_recall,
            **{f"recall_{label}": per_class_recall[i] for i, label in enumerate(FLAT_LABELS)},
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag", type=str, default="",
                         help="Must match the --tag used for train.py, if any "
                              "(e.g. --tag masked -> reads outputs/checkpoints_masked/).")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1", "pneumonia_recall"],
                         default="macro_f1",
                         help="Which metric to maximize when picking the threshold on the val set.")
    args = parser.parse_args()

    seed_suffix = "" if args.seed == DEFAULT_SEED else f"_seed{args.seed}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    suffix = seed_suffix + tag_suffix
    checkpoint_path = ROOT / "outputs" / f"checkpoints{suffix}" / "best_model.pt"
    eval_dir = ROOT / "outputs" / f"evaluation{suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed={args.seed} | checkpoint: {checkpoint_path}")

    model = MultiScaleEfficientNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, val_loader, test_loader = get_dataloaders(MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=0)

    print("\nRunning model on VALIDATION set (thresholds tuned here, not on test)...")
    val_s1, val_s2, val_true = collect_stage_logits(model, val_loader, device)
    val_results = sweep_thresholds(val_s1, val_s2, val_true, THRESHOLDS)

    print("\nValidation-set sweep:")
    print(val_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_row = val_results.loc[val_results[args.metric].idxmax()]
    best_threshold = best_row["threshold"]
    default_row = val_results.iloc[(val_results["threshold"] - 0.5).abs().idxmin()]

    print(f"\nBest threshold on VAL by {args.metric}: {best_threshold} "
          f"({args.metric}={best_row[args.metric]:.4f})")
    print(f"Default threshold (0.5) on VAL for comparison: "
          f"{args.metric}={default_row[args.metric]:.4f}")

    val_results.to_csv(eval_dir / "stage1_threshold_sweep_val.csv", index=False)
    print(f"Saved full val sweep to {eval_dir / 'stage1_threshold_sweep_val.csv'}")

    # ---------------- Final honest evaluation on TEST set at the chosen threshold ----------------
    print(f"\nEvaluating chosen threshold ({best_threshold}) on the TEST set "
          f"(never touched during threshold selection)...")
    test_s1, test_s2, test_true = collect_stage_logits(model, test_loader, device)

    y_pred_default = combine_predictions_thresholded(test_s1, test_s2, abnormal_threshold=0.5).numpy()
    y_pred_tuned = combine_predictions_thresholded(test_s1, test_s2, abnormal_threshold=best_threshold).numpy()

    print("\n--- TEST set, default threshold (0.5) ---")
    print(f"Accuracy: {accuracy_score(test_true, y_pred_default):.4f}")
    print(classification_report(test_true, y_pred_default, target_names=FLAT_LABELS, zero_division=0))

    print(f"\n--- TEST set, TUNED threshold ({best_threshold}) ---")
    print(f"Accuracy: {accuracy_score(test_true, y_pred_tuned):.4f}")
    print(classification_report(test_true, y_pred_tuned, target_names=FLAT_LABELS, zero_division=0))

    print(f"\nRecommended: apply --stage1-threshold {best_threshold} in evaluate.py "
          f"to reproduce this as your final reported number.")


if __name__ == "__main__":
    main()
