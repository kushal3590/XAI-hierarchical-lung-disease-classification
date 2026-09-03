"""
Full evaluation report: confusion matrix + per-class ROC/AUC curves.
Matches Section 7 (Evaluation Metrics) / Figure 4 of the Phase 1 report.

Run from project root (after training has produced outputs/checkpoints/best_model.pt):
    python scripts/evaluate.py
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from dataset import get_dataloaders, FLAT_LABELS
from model import MultiScaleEfficientNet, combine_predictions, combine_predictions_thresholded

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
DEFAULT_SEED = 42

BATCH_SIZE = 16


def get_combined_probs(stage1_logits, stage2_logits):
    """Combine the two heads into a proper 4-class probability distribution
    (each row sums to 1), needed for per-class ROC/AUC — not just the argmax
    flat prediction used during training logging."""
    stage1_probs = torch.softmax(stage1_logits, dim=1)   # [N, 2] -> Normal, Abnormal
    stage2_probs = torch.softmax(stage2_logits, dim=1)   # [N, 3] -> Pneumonia, TB, COVID-19

    p_normal = stage1_probs[:, 0:1]
    p_abnormal = stage1_probs[:, 1:2]
    p_diseases = p_abnormal * stage2_probs  # [N, 3], each scaled by P(abnormal)

    combined = torch.cat([p_normal, p_diseases], dim=1)  # [N, 4] in FLAT_LABELS order
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Which seeded training run to evaluate (must match "
                              "the --seed used for train.py).")
    parser.add_argument("--tag", type=str, default="",
                         help="Must match the --tag used for train.py, if any "
                              "(e.g. --tag masked -> reads outputs/checkpoints_masked/).")
    parser.add_argument("--stage1-threshold", type=float, default=0.5,
                         help="Stage 1 Normal/Abnormal decision threshold on P(Abnormal). "
                              "Default 0.5 (argmax). Tune this first with "
                              "scripts/tune_stage1_threshold.py — that script selects the "
                              "value on the VALIDATION set; pass the same value here to "
                              "reproduce it as your official test-set report.")
    args = parser.parse_args()
    seed = args.seed
    threshold = args.stage1_threshold

    seed_suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    suffix = seed_suffix + tag_suffix
    checkpoint_path = ROOT / "outputs" / f"checkpoints{suffix}" / "best_model.pt"
    eval_dir = ROOT / "outputs" / f"evaluation{suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating seed={seed} checkpoint: {checkpoint_path} | stage1_threshold={threshold}")

    _, _, test_loader = get_dataloaders(MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=0)

    model = MultiScaleEfficientNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_true, all_pred, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            stage1_logits, stage2_logits = model(images)

            # Soft joint probabilities — used ONLY for ROC-AUC scoring below,
            # where a continuous, sums-to-1 score per class is the correct input.
            # NOTE: ROC-AUC is threshold-independent by construction, so this
            # is unaffected by --stage1-threshold — it reports the model's
            # underlying ranking quality regardless of operating point.
            probs = get_combined_probs(stage1_logits, stage2_logits)  # [N, 4]

            # Hard sequential decision — used for the actual predicted LABEL
            # (confusion matrix, classification report, TB recall etc.).
            # Uses the tunable Stage 1 threshold (default 0.5 = plain argmax,
            # equivalent to the original combine_predictions()).
            preds = combine_predictions_thresholded(stage1_logits, stage2_logits,
                                                     abnormal_threshold=threshold)

            all_true.extend(batch["flat"].tolist())
            all_pred.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
            all_paths.extend(batch["path"])

    all_probs = np.concatenate(all_probs, axis=0)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # ---------------- Classification report ----------------
    report_txt = classification_report(all_true, all_pred, target_names=FLAT_LABELS, zero_division=0)
    print("\nClassification report:")
    print(report_txt)
    with open(eval_dir / "classification_report.txt", "w") as f:
        f.write(report_txt)

    # ---------------- Confusion matrix ----------------
    cm = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=FLAT_LABELS, yticklabels=FLAT_LABELS, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=FLAT_LABELS, yticklabels=FLAT_LABELS, ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {eval_dir / 'confusion_matrix.png'}")

    # Print explicitly what's feeding TB's false positives / Pneumonia's false negatives
    tb_idx = FLAT_LABELS.index("Tuberculosis")
    print("\nWhere TB false positives come from (true label -> predicted as TB):")
    for i, label in enumerate(FLAT_LABELS):
        if i != tb_idx and cm[i, tb_idx] > 0:
            print(f"  {label} -> Tuberculosis: {cm[i, tb_idx]} cases "
                  f"({cm_norm[i, tb_idx]*100:.1f}% of true {label})")

    # ---------------- Per-class ROC / AUC (one-vs-rest) ----------------
    plt.figure(figsize=(8, 8))
    aucs = {}
    for i, label in enumerate(FLAT_LABELS):
        y_true_binary = (all_true == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        aucs[label] = roc_auc
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Per-Class ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved ROC curves to {eval_dir / 'roc_curves.png'}")

    print("\nPer-class AUC summary:")
    for label, val in aucs.items():
        print(f"  {label:15s}: {val:.4f}")

    # Save raw predictions for later use (e.g. picking Grad-CAM examples)
    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "true_label": [FLAT_LABELS[i] for i in all_true],
        "pred_label": [FLAT_LABELS[i] for i in all_pred],
        **{f"prob_{label}": all_probs[:, i] for i, label in enumerate(FLAT_LABELS)},
    })
    pred_df.to_csv(eval_dir / "test_predictions.csv", index=False)
    print(f"Saved raw predictions to {eval_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
