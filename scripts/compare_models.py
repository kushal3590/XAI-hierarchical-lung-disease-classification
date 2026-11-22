"""
Paired comparison of two models on the SAME test images (EfficientNet-B0 vs
DenseNet-121 for the guide's Review-1 request). Works with any two
test_predictions.csv files that have columns image_path, true_label, pred_label
(the format written by scripts/evaluate.py and scripts/evaluate_densenet.py).

Reports, for each model: accuracy, macro-F1, per-class recall; then McNemar's
test on the paired correct/incorrect outcomes overall and within each true
class (exact binomial when there are fewer than 25 discordant pairs, otherwise
chi-square with continuity correction — same rule as scripts/mcnemar_test.py).

Read-only: it only reads the two CSVs and writes one small summary CSV to
outputs/densenet/comparison/.

Defaults compare your FINAL EfficientNet model against the DenseNet model
trained with the same improved recipe:
    python scripts/compare_models.py

Original-recipe comparison (seed 42, default threshold, EfficientNet 90.3%):
    python scripts/compare_models.py --a outputs/evaluation/test_predictions.csv \
        --b outputs/densenet/hierarchical/evaluation/test_predictions.csv \
        --name-a EfficientNet-B0 --name-b DenseNet-121
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from metrics import f1_score  # numpy-only shim (no sklearn needed)

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
ALPHA = 0.05


def load(path):
    df = pd.read_csv(path)
    missing = {"image_path", "true_label", "pred_label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[["image_path", "true_label", "pred_label"]]


def mcnemar_row(correct_a, correct_b):
    """McNemar on two boolean arrays (is model A / B correct per image)."""
    both = int((correct_a & correct_b).sum())
    neither = int((~correct_a & ~correct_b).sum())
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n_disc = a_only + b_only
    if n_disc == 0:
        stat, p, kind = float("nan"), 1.0, "no discordant pairs"
    else:
        exact = n_disc < 25
        res = mcnemar([[both, a_only], [b_only, neither]], exact=exact, correction=not exact)
        stat, p = float(res.statistic), float(res.pvalue)
        kind = "exact binomial" if exact else "chi-square (continuity corr.)"
    return {"both_correct": both, "both_wrong": neither, "a_only_correct": a_only,
            "b_only_correct": b_only, "test": kind, "statistic": stat, "p_value": p}


def verdict(row, name_a, name_b):
    if row["p_value"] < ALPHA:
        winner = name_a if row["a_only_correct"] > row["b_only_correct"] else name_b
        return f"SIGNIFICANT (p<{ALPHA}): {winner} is better"
    return f"not significant (p>={ALPHA}): difference plausibly chance"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=str(ROOT / "outputs" / "evaluation_masked" / "test_predictions.csv"),
                        help="Predictions CSV of model A (default: final EfficientNet model).")
    parser.add_argument("--b", default=str(ROOT / "outputs" / "densenet" / "hierarchical" /
                                           "evaluation" / "test_predictions.csv"),
                        help="Predictions CSV of model B (default: DenseNet, improved recipe).")
    parser.add_argument("--name-a", default="EfficientNet-B0")
    parser.add_argument("--name-b", default="DenseNet-121")
    args = parser.parse_args()

    A, B = load(args.a), load(args.b)
    print(f"A = {args.name_a}: {args.a}\nB = {args.name_b}: {args.b}")

    df = A.merge(B[["image_path", "pred_label"]], on="image_path", how="inner", suffixes=("_a", "_b"))
    print(f"\nMatched {len(df)} images by path (A: {len(A)}, B: {len(B)})")
    if len(df) < min(len(A), len(B)):
        print("[WARN] Not all images matched - check both CSVs come from the same test split.")
    if len(df) == 0:
        raise SystemExit("No overlapping images - nothing to compare.")

    y = df["true_label"].to_numpy()
    pa, pb = df["pred_label_a"].to_numpy(), df["pred_label_b"].to_numpy()
    ca, cb = pa == y, pb == y

    # ---------- headline metrics ----------
    print(f"\n{'':22s}{args.name_a:>16s}{args.name_b:>16s}")
    print(f"{'Accuracy':22s}{ca.mean():16.4f}{cb.mean():16.4f}")
    print(f"{'Macro-F1':22s}{f1_score(y, pa, average='macro', labels=LABELS):16.4f}"
          f"{f1_score(y, pb, average='macro', labels=LABELS):16.4f}")
    for lbl in LABELS:
        m = y == lbl
        if m.any():
            print(f"{'Recall - ' + lbl:22s}{(pa[m] == lbl).mean():16.4f}{(pb[m] == lbl).mean():16.4f}")

    # ---------- McNemar: overall ----------
    rows = []
    overall = mcnemar_row(ca, cb)
    overall.update(subset="overall", n=len(df))
    rows.append(overall)
    print("\n--- McNemar's test, overall (paired correct/incorrect) ---")
    print(f"  both correct: {overall['both_correct']} | both wrong: {overall['both_wrong']}")
    print(f"  {args.name_a} correct, {args.name_b} wrong: {overall['a_only_correct']}")
    print(f"  {args.name_b} correct, {args.name_a} wrong: {overall['b_only_correct']}")
    print(f"  {overall['test']}: statistic={overall['statistic']:.4f}, p={overall['p_value']:.4f}")
    print(f"  => {verdict(overall, args.name_a, args.name_b)}")

    # ---------- McNemar: within each true class ----------
    print("\n--- McNemar's test within each true class ---")
    for lbl in LABELS:
        m = y == lbl
        if not m.any():
            continue
        r = mcnemar_row(ca[m], cb[m])
        r.update(subset=lbl, n=int(m.sum()))
        rows.append(r)
        print(f"  {lbl:13s} n={int(m.sum()):5d} | {args.name_a} only: {r['a_only_correct']:4d} | "
              f"{args.name_b} only: {r['b_only_correct']:4d} | p={r['p_value']:.4f} | "
              f"{verdict(r, args.name_a, args.name_b)}")
    print("\nNote: small classes (e.g. Tuberculosis, n=164) have little statistical power - "
          "'not significant' there means 'not enough evidence', not 'equal'.")

    out_dir = ROOT / "outputs" / "densenet" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name_a}_vs_{args.name_b}.csv".replace(" ", "")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
