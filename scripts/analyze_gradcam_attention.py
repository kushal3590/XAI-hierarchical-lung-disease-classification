"""
Turns the per-image Grad-CAM attention-zone data (from gradcam_explain.py
--all-test) into actual measured percentages, upgrading the report's
"seen in 2 independent Grad-CAM examples" limitation into a quantified claim
across the full test set.

Zones (see classify_attention_zone() in gradcam_explain.py for definitions
and caveats — this is a geometric heuristic based on typical CXR layout,
NOT a real lung-segmentation mask):
    lung_field      -> attention centered in the plausible lung region
    corner_marker   -> attention centered in a top corner (position markers)
    outer_margin    -> attention centered at the image's outer edge
    midline_strip   -> attention centered on the vertical mediastinal strip

Requires outputs/gradcam_full[_seed<N>]/gradcam_log.csv, produced by:
    python scripts/gradcam_explain.py --all-test [--seed N]

Run from project root:
    python scripts/analyze_gradcam_attention.py
    python scripts/analyze_gradcam_attention.py --seed 123
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 42

NON_ANATOMICAL_ZONES = {"corner_marker", "outer_margin", "midline_strip"}

# corner_marker and outer_margin are near-certainly non-anatomical (position
# markers, image edges outside the body). midline_strip is more ambiguous —
# it also covers the trachea/mediastinum/heart border, which ARE legitimate
# diagnostic territory. Reporting both lets you distinguish "the model looks
# at genuine artifacts" from "the model looks at the anatomical midline,"
# which is a much weaker and less interesting claim.
ARTIFACT_ZONES = {"corner_marker", "outer_margin"}


def zone_stats(df, stage_col):
    valid = df[df[stage_col].notna() & (df[stage_col] != "N/A") & (df[stage_col] != "undefined")]
    n = len(valid)
    if n == 0:
        return {"n": 0, "non_anatomical_pct": None, "artifact_only_pct": None,
                "corner_marker_pct": None, "outer_margin_pct": None,
                "midline_strip_pct": None, "lung_field_pct": None}
    counts = valid[stage_col].value_counts()

    def pct(zone_set):
        return 100.0 * sum(counts.get(z, 0) for z in zone_set) / n

    return {
        "n": n,
        "non_anatomical_pct": pct(NON_ANATOMICAL_ZONES),
        "artifact_only_pct": pct(ARTIFACT_ZONES),  # corner_marker + outer_margin only
        "corner_marker_pct": pct({"corner_marker"}),
        "outer_margin_pct": pct({"outer_margin"}),
        "midline_strip_pct": pct({"midline_strip"}),
        "lung_field_pct": pct({"lung_field"}),
    }


def print_breakdown(df, stage_col, label):
    print(f"\n--- {label} ---")
    stats = zone_stats(df, stage_col)
    if stats["n"] == 0:
        print("  (no valid attention-zone data)")
        return
    print(f"  n={stats['n']}")
    print(f"  non-anatomical (all 3 zones):     {stats['non_anatomical_pct']:.1f}%")
    print(f"  artifact-only (corner+margin):     {stats['artifact_only_pct']:.1f}%  <- near-certain shortcuts")
    print(f"  midline_strip alone:               {stats['midline_strip_pct']:.1f}%  <- ambiguous (real anatomy too)")
    print(f"  lung_field:                        {stats['lung_field_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Which seeded full-test-set Grad-CAM run to analyze.")
    args = parser.parse_args()

    suffix = "" if args.seed == DEFAULT_SEED else f"_seed{args.seed}"
    log_path = ROOT / "outputs" / f"gradcam_full{suffix}" / "gradcam_log.csv"

    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} not found. Run:\n"
            f"    python scripts/gradcam_explain.py --all-test"
            + (f" --seed {args.seed}" if args.seed != DEFAULT_SEED else "")
        )

    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} rows from {log_path}")

    df["correct"] = df["true_label"] == df["pred_label"]

    # ---------------- Overall: Stage 1 attention across the whole test set ----------------
    print_breakdown(df, "stage1_attention_zone", "Stage 1 — ALL test images")

    # ---------------- Split by correct vs incorrect ----------------
    print_breakdown(df[df.correct], "stage1_attention_zone", "Stage 1 — CORRECTLY classified images")
    print_breakdown(df[~df.correct], "stage1_attention_zone", "Stage 1 — MISCLASSIFIED images")

    # ---------------- The dominant failure mode from your report: Pneumonia -> Normal ----------------
    pneu_as_normal = df[(df.true_label == "Pneumonia") & (df.pred_label == "Normal")]
    print_breakdown(pneu_as_normal, "stage1_attention_zone",
                     f"Stage 1 — Pneumonia misclassified as Normal (n={len(pneu_as_normal)})")

    # For comparison: Pneumonia images Stage 1 got RIGHT (gated as Abnormal)
    pneu_correct_gate = df[(df.true_label == "Pneumonia") & (df.pred_label != "Normal")]
    print_breakdown(pneu_correct_gate, "stage1_attention_zone",
                     f"Stage 1 — Pneumonia correctly gated as Abnormal (n={len(pneu_correct_gate)})")

    # ---------------- Stage 2, only where it's actually meaningful (Stage 1 = Abnormal) ----------------
    stage2_valid = df[df["stage2_attention_zone"] != "N/A"]
    print_breakdown(stage2_valid, "stage2_attention_zone",
                     f"Stage 2 — images where Stage 1 predicted Abnormal (n={len(stage2_valid)})")

    # ---------------- Per true-class breakdown for Stage 1 ----------------
    print("\n=== Stage 1 attention %, by true class (non-anatomical / artifact-only) ===")
    for label in sorted(df["true_label"].dropna().unique()):
        subset = df[df.true_label == label]
        stats = zone_stats(subset, "stage1_attention_zone")
        if stats["n"] > 0:
            print(f"  {label:15s}: non-anatomical={stats['non_anatomical_pct']:5.1f}%  "
                  f"artifact-only={stats['artifact_only_pct']:5.1f}%  (n={stats['n']})")

    # ---------------- Save a summary CSV for easy pasting into the report ----------------
    summary_rows = []
    for name, subset in [
        ("all_images", df),
        ("correct", df[df.correct]),
        ("misclassified", df[~df.correct]),
        ("pneumonia_as_normal", pneu_as_normal),
        ("pneumonia_correct_gate", pneu_correct_gate),
    ]:
        stats = zone_stats(subset, "stage1_attention_zone")
        summary_rows.append({"subset": name, **stats})

    out_dir = log_path.parent
    summary_path = out_dir / "attention_zone_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary table to {summary_path}")


if __name__ == "__main__":
    main()
