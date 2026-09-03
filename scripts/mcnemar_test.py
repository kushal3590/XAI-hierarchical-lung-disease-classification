"""
McNemar's test: is the accuracy difference between the hierarchical model
and the flat baseline statistically significant, or could it be noise from
the small (164-image) Tuberculosis test set dragging things around?

McNemar's test is the right tool here specifically because both models were
evaluated on the EXACT SAME test images — it looks at paired
disagreements (cases where one model got it right and the other didn't),
not just the two accuracy numbers in isolation.

Requires both:
    outputs/evaluation/test_predictions.csv           (hierarchical model)
    outputs/evaluation/baseline_test_predictions.csv  (flat baseline)

Run from project root:
    python scripts/mcnemar_test.py
"""
from pathlib import Path
import argparse

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Which seeded run's evaluation outputs to compare "
                              "(must match the --seed used for evaluate.py / "
                              "evaluate_baseline.py).")
    args = parser.parse_args()
    seed = args.seed

    suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    eval_dir = ROOT / "outputs" / f"evaluation{suffix}"
    hier_csv = eval_dir / "test_predictions.csv"
    baseline_csv = eval_dir / "baseline_test_predictions.csv"

    print(f"Comparing seed={seed} predictions from: {eval_dir}")

    if not hier_csv.exists() or not baseline_csv.exists():
        raise FileNotFoundError(
            f"Need both {hier_csv} (from evaluate.py --seed {seed}) and "
            f"{baseline_csv} (from evaluate_baseline.py --seed {seed})."
        )

    hier = pd.read_csv(hier_csv)[["image_path", "true_label", "pred_label"]].rename(
        columns={"pred_label": "hier_pred"}
    )
    base = pd.read_csv(baseline_csv)[["image_path", "pred_label"]].rename(
        columns={"pred_label": "base_pred"}
    )

    merged = hier.merge(base, on="image_path", how="inner")
    n_matched = len(merged)
    n_hier_total = len(hier)
    n_base_total = len(base)
    print(f"Matched {n_matched} images by path "
          f"(hierarchical set: {n_hier_total}, baseline set: {n_base_total})")
    if n_matched < min(n_hier_total, n_base_total):
        print("[WARN] Not all images matched — check both CSVs came from the same test.csv split.")

    merged["hier_correct"] = merged["hier_pred"] == merged["true_label"]
    merged["base_correct"] = merged["base_pred"] == merged["true_label"]

    both_correct = int((merged.hier_correct & merged.base_correct).sum())
    both_wrong = int((~merged.hier_correct & ~merged.base_correct).sum())
    hier_only = int((merged.hier_correct & ~merged.base_correct).sum())
    base_only = int((~merged.hier_correct & merged.base_correct).sum())

    print("\nPaired contingency table (overall accuracy):")
    print(f"  Both correct:                    {both_correct}")
    print(f"  Both wrong:                       {both_wrong}")
    print(f"  Hierarchical correct, baseline wrong: {hier_only}")
    print(f"  Baseline correct, hierarchical wrong: {base_only}")

    table = [[both_correct, hier_only], [base_only, both_wrong]]
    use_exact = (hier_only + base_only) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    print(f"\nMcNemar's test ({'exact binomial' if use_exact else 'chi-square with continuity correction'}):")
    print(f"  statistic = {result.statistic:.4f}")
    print(f"  p-value   = {result.pvalue:.4f}")

    alpha = 0.05
    if result.pvalue < alpha:
        winner = "hierarchical" if hier_only > base_only else "baseline"
        print(f"\n=> SIGNIFICANT at alpha={alpha}: the {winner} model's overall accuracy "
              f"advantage is unlikely to be due to chance.")
    else:
        print(f"\n=> NOT significant at alpha={alpha}: the {abs(hier_only - base_only)}-case "
              f"difference between the models could plausibly be noise, not a real effect.")

    tb_subset = merged[merged.true_label == "Tuberculosis"].copy()
    if len(tb_subset) > 0:
        tb_subset["hier_correct_tb"] = tb_subset["hier_pred"] == "Tuberculosis"
        tb_subset["base_correct_tb"] = tb_subset["base_pred"] == "Tuberculosis"

        tb_both_correct = int((tb_subset.hier_correct_tb & tb_subset.base_correct_tb).sum())
        tb_both_wrong = int((~tb_subset.hier_correct_tb & ~tb_subset.base_correct_tb).sum())
        tb_hier_only = int((tb_subset.hier_correct_tb & ~tb_subset.base_correct_tb).sum())
        tb_base_only = int((~tb_subset.hier_correct_tb & tb_subset.base_correct_tb).sum())

        print(f"\n--- Tuberculosis-specific subset (n={len(tb_subset)}) ---")
        print(f"  Both correctly identified as TB:      {tb_both_correct}")
        print(f"  Both missed:                           {tb_both_wrong}")
        print(f"  Hierarchical correct, baseline wrong:  {tb_hier_only}")
        print(f"  Baseline correct, hierarchical wrong:  {tb_base_only}")

        tb_table = [[tb_both_correct, tb_hier_only], [tb_base_only, tb_both_wrong]]
        tb_use_exact = (tb_hier_only + tb_base_only) < 25
        tb_result = mcnemar(tb_table, exact=tb_use_exact, correction=not tb_use_exact)
        print(f"  McNemar statistic = {tb_result.statistic:.4f}, p-value = {tb_result.pvalue:.4f}")
        if tb_result.pvalue < alpha:
            print(f"  => SIGNIFICANT: the TB-recognition difference is unlikely to be chance.")
        else:
            print(f"  => NOT significant: with only {len(tb_subset)} TB test images, "
                  f"this difference is plausibly noise.")


if __name__ == "__main__":
    main()
