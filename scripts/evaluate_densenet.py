"""
Evaluates a trained DenseNet-121 comparison model on the held-out TEST set and
writes the same kinds of artifacts as scripts/evaluate.py, but only inside the
run's own folder (nothing existing is touched):

    outputs/densenet/<run_name>/evaluation/
        classification_report.txt
        confusion_matrix.png
        roc_curves.png
        test_predictions.csv          (same columns as the EfficientNet one)
        metrics_summary.json
        threshold_sweep_val.csv       (only with --tune)

Stage 1 threshold (hierarchical variant only):
    --stage1-threshold 0.25     use a fixed threshold (default 0.5 = plain argmax)
    --tune                      select the threshold on the VALIDATION set (same
                                grid and criterion as scripts/tune_stage1_threshold.py),
                                then evaluate that threshold once on the test set.
                                The test set is never used to choose the threshold.

Examples (run from project root; flags must match how the model was trained):
    python scripts/evaluate_densenet.py --tune
    python scripts/evaluate_densenet.py --tune --tag masked
    python scripts/evaluate_densenet.py --variant flat
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, recall_score, confusion_matrix,
                             roc_curve, auc, classification_report)

from dataset import get_dataloaders, FLAT_LABELS
from model import combine_predictions_thresholded
from model_densenet import MultiScaleDenseNet, FlatMultiScaleDenseNet

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_ROOT = ROOT / "outputs" / "densenet"
DEFAULT_SEED = 42
BATCH_SIZE = 16
THRESHOLDS = np.arange(0.15, 0.55, 0.025)  # same grid as tune_stage1_threshold.py


def collect(model, loader, device, variant):
    """One pass over a loader -> cached raw outputs (so thresholds can be
    re-scored without re-running the network)."""
    s1_all, s2_all, flat_all, true_all, paths = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            if variant == "hierarchical":
                s1_all.append(out[0].cpu())
                s2_all.append(out[1].cpu())
            else:
                flat_all.append(out.cpu())
            true_all.extend(batch["flat"].tolist())
            paths.extend(batch["path"])
    if variant == "hierarchical":
        return {"s1": torch.cat(s1_all), "s2": torch.cat(s2_all)}, np.array(true_all), paths
    return {"flat": torch.cat(flat_all)}, np.array(true_all), paths


def predict(outputs, variant, threshold):
    if variant == "hierarchical":
        return combine_predictions_thresholded(outputs["s1"], outputs["s2"],
                                               abnormal_threshold=threshold).numpy()
    return outputs["flat"].argmax(dim=1).numpy()


def joint_probs(outputs, variant):
    """[N,4] probabilities in FLAT_LABELS order (same construction as evaluate.py)."""
    if variant == "hierarchical":
        p1 = F.softmax(outputs["s1"], dim=1)
        p2 = F.softmax(outputs["s2"], dim=1)
        return torch.cat([p1[:, 0:1], p1[:, 1:2] * p2], dim=1).numpy()
    return F.softmax(outputs["flat"], dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["hierarchical", "flat"], default="hierarchical")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--stage1-threshold", type=float, default=0.5)
    parser.add_argument("--tune", action="store_true",
                        help="Pick the Stage 1 threshold on the validation set (hierarchical only).")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1",
                        help="Metric maximised on the validation set when --tune is used.")
    args = parser.parse_args()
    variant = args.variant

    run_name = variant
    if args.seed != DEFAULT_SEED:
        run_name += f"_seed{args.seed}"
    if args.tag:
        run_name += f"_{args.tag}"
    run_dir = OUTPUT_ROOT / run_name
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No DenseNet checkpoint at {ckpt_path}. Train it first with "
                                f"scripts/train_densenet.py (same --variant/--seed/--tag).")
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | run: {run_name}")

    if variant == "hierarchical":
        model = MultiScaleDenseNet(pretrained=False, freeze_backbone=False).to(device)
    else:
        model = FlatMultiScaleDenseNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, val_loader, test_loader = get_dataloaders(MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=0)

    threshold = args.stage1_threshold
    if variant == "flat":
        threshold = None
        if args.tune:
            print("Note: --tune applies to the hierarchical variant only; ignoring.")
    elif args.tune:
        print("\nTuning Stage 1 threshold on the VALIDATION set...")
        val_out, val_true, _ = collect(model, val_loader, device, variant)
        rows = []
        for t in THRESHOLDS:
            vp = predict(val_out, variant, float(t))
            rows.append({
                "threshold": round(float(t), 3),
                "accuracy": accuracy_score(val_true, vp),
                "macro_f1": f1_score(val_true, vp, average="macro", zero_division=0),
                "pneumonia_recall": recall_score(val_true, vp, labels=[FLAT_LABELS.index("Pneumonia")],
                                                 average="micro", zero_division=0),
            })
        sweep = pd.DataFrame(rows)
        print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        threshold = float(sweep.loc[sweep[args.metric].idxmax(), "threshold"])
        sweep.to_csv(eval_dir / "threshold_sweep_val.csv", index=False)
        print(f"\nBest threshold on VAL by {args.metric}: {threshold}")

    print(f"\nEvaluating on TEST set (stage1_threshold={threshold})...")
    test_out, y_true, paths = collect(model, test_loader, device, variant)
    y_pred = predict(test_out, variant, threshold if threshold is not None else 0.5)
    probs = joint_probs(test_out, variant)

    report_txt = classification_report(y_true, y_pred, target_names=FLAT_LABELS, zero_division=0)
    header = f"DenseNet-121 | {run_name} | stage1_threshold={threshold}\n\n"
    print("\n" + header + report_txt)
    (eval_dir / "classification_report.txt").write_text(header + report_txt)

    # Confusion matrix (counts + row-normalised)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=FLAT_LABELS,
                yticklabels=FLAT_LABELS, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=FLAT_LABELS,
                yticklabels=FLAT_LABELS, ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalized)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC / AUC (one-vs-rest; threshold-independent)
    plt.figure(figsize=(8, 8))
    aucs = {}
    for i, label in enumerate(FLAT_LABELS):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
        aucs[label] = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {aucs[label]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-Class ROC Curves (One-vs-Rest) - DenseNet-121")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curves.png", dpi=150)
    plt.close()
    print("Per-class AUC:", {k: round(v, 4) for k, v in aucs.items()})

    # Per-image predictions in the same format as outputs/evaluation*/test_predictions.csv
    pd.DataFrame({
        "image_path": paths,
        "true_label": [FLAT_LABELS[i] for i in y_true],
        "pred_label": [FLAT_LABELS[i] for i in y_pred],
        **{f"prob_{label}": probs[:, i] for i, label in enumerate(FLAT_LABELS)},
    }).to_csv(eval_dir / "test_predictions.csv", index=False)

    summary = {
        "run_name": run_name,
        "variant": variant,
        "stage1_threshold": threshold,
        "threshold_tuned_on_validation": bool(args.tune and variant == "hierarchical"),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_auc": {k: float(v) for k, v in aucs.items()},
        "report": classification_report(y_true, y_pred, target_names=FLAT_LABELS,
                                       zero_division=0, output_dict=True),
    }
    with open(eval_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTest accuracy: {summary['test_accuracy']:.4f} | macro-F1: {summary['test_macro_f1']:.4f}")
    print(f"Artifacts saved to: {eval_dir}")
    print("Next: python scripts/compare_models.py --help")


if __name__ == "__main__":
    main()
