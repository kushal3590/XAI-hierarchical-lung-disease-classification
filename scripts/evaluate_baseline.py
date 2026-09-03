"""
Evaluates the FLAT baseline model and saves per-image predictions
(image_path, true_label, pred_label) in the same format as evaluate.py's
test_predictions.csv — needed to pair up predictions image-by-image for
McNemar's test against the hierarchical model.

Run from project root (after train_baseline.py has produced
outputs/checkpoints/baseline_flat_model.pt):
    python scripts/evaluate_baseline.py
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pandas as pd
from sklearn.metrics import classification_report

from dataset import get_dataloaders, FLAT_LABELS
from model import FlatMultiScaleEfficientNet

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
DEFAULT_SEED = 42

BATCH_SIZE = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Which seeded baseline training run to evaluate "
                              "(must match the --seed used for train_baseline.py).")
    args = parser.parse_args()
    seed = args.seed

    suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    checkpoint_path = ROOT / "outputs" / f"checkpoints{suffix}" / "baseline_flat_model.pt"
    eval_dir = ROOT / "outputs" / f"evaluation{suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating seed={seed} baseline checkpoint: {checkpoint_path}")

    _, _, test_loader = get_dataloaders(MANIFEST_DIR, batch_size=BATCH_SIZE, num_workers=0)

    model = FlatMultiScaleEfficientNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_true, all_pred, all_paths = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_true.extend(batch["flat"].tolist())
            all_pred.extend(preds.cpu().tolist())
            all_paths.extend(batch["path"])

    report = classification_report(all_true, all_pred, target_names=FLAT_LABELS, zero_division=0)
    print("\nBaseline classification report (re-confirmation):")
    print(report)

    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "true_label": [FLAT_LABELS[i] for i in all_true],
        "pred_label": [FLAT_LABELS[i] for i in all_pred],
    })
    out_path = eval_dir / "baseline_test_predictions.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved per-image baseline predictions to: {out_path}")
    print("Ready for scripts/mcnemar_test.py")


if __name__ == "__main__":
    main()
