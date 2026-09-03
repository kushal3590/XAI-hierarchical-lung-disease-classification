"""
SHAP and LIME explainability on a representative subset of the test set.
Implements the remaining piece of FR6 / the Explainability Layer (Grad-CAM
was already done in gradcam_explain.py) and NFR4 ("SHAP and LIME... for a
solid subset of the test set" — not the full set, since they're much slower
than Grad-CAM).

Both explainers use the SAME black-box predict function — the model's
combined 4-class probability output (Normal/Pneumonia/Tuberculosis/COVID-19,
derived from the two hierarchical heads exactly as in evaluate.py). This
keeps SHAP and LIME's explanations directly comparable to each other and to
the classification report, rather than explaining two different things.

Default subset size: 25 images, stratified across the 4 classes (~6 each),
matching your NFR4's "representative subset" framing.

Run from project root (after evaluate.py has been run at least once):
    python scripts/shap_lime_explain.py
    python scripts/shap_lime_explain.py --n 10       # smaller/faster
    python scripts/shap_lime_explain.py --n 50       # larger/slower
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

from preprocessing import load_grayscale, apply_clahe, resize, grayscale_to_rgb, IMAGENET_MEAN, IMAGENET_STD
from model import MultiScaleEfficientNet

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 42

FLAT_LABELS = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]


def get_display_image(path):
    """Returns a 224x224x3 uint8 RGB array (CLAHE-enhanced) — the same
    preprocessing as training, but kept as uint8 for SHAP/LIME's image APIs,
    which both expect standard image arrays rather than normalized tensors."""
    gray = load_grayscale(path)
    gray = apply_clahe(gray)
    gray = resize(gray)
    rgb = grayscale_to_rgb(gray)  # uint8, HWC, 0-255
    return rgb


def make_predict_fn(model, device):
    """Shared black-box predict function for both SHAP and LIME:
    takes a batch of uint8 RGB images (N,224,224,3), returns the combined
    4-class probability distribution (N,4) as a numpy array."""

    def predict_fn(images_uint8):
        images_uint8 = np.asarray(images_uint8)
        imgs = images_uint8.astype(np.float32) / 255.0
        imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # NHWC -> NCHW
        tensor = torch.from_numpy(imgs).float().to(device)

        probs_batches = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                chunk = tensor[i:i + batch_size]
                s1_logits, s2_logits = model(chunk)
                p1 = F.softmax(s1_logits, dim=1)
                p2 = F.softmax(s2_logits, dim=1)
                p_normal = p1[:, 0:1]
                p_abnormal = p1[:, 1:2]
                p_disease = p_abnormal * p2
                probs = torch.cat([p_normal, p_disease], dim=1)
                probs_batches.append(probs.cpu().numpy())
        return np.concatenate(probs_batches, axis=0)

    return predict_fn


def pick_subset(n, predictions_csv):
    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"{predictions_csv} not found. Run scripts/evaluate.py first (with matching --seed)."
        )
    df = pd.read_csv(predictions_csv)
    per_class = max(1, n // len(FLAT_LABELS))

    picked = []
    rng = np.random.RandomState(42)
    for label in FLAT_LABELS:
        subset = df[df.true_label == label]
        if len(subset) == 0:
            continue
        take = min(per_class, len(subset))
        picked.append(subset.sample(n=take, random_state=rng))
    result = pd.concat(picked, ignore_index=True)

    if len(result) < n:
        remaining = df.drop(result.index, errors="ignore")
        extra = remaining.sample(n=min(n - len(result), len(remaining)), random_state=rng)
        result = pd.concat([result, extra], ignore_index=True)

    return result.head(n)


def run_lime(predict_fn, display_img, true_label, pred_label, tag, out_dir):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        display_img,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=500,
        batch_size=16,
    )
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=8, hide_rest=False
    )
    overlay = mark_boundaries(temp / 255.0, mask)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_img)
    axes[0].set_title(f"Input\nTrue: {true_label} | Predicted: {pred_label}")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"LIME\nTop superpixels for: {FLAT_LABELS[top_label]}")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = out_dir / f"lime_{tag}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_shap(predict_fn, display_img, true_label, pred_label, tag, out_dir, max_evals=300):
    masker = shap.maskers.Image("blur(32,32)", display_img.shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=FLAT_LABELS)

    shap_values = explainer(
        np.expand_dims(display_img, 0),
        max_evals=max_evals,
        batch_size=16,
        outputs=shap.Explanation.argsort.flip[:1],
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_img)
    axes[0].set_title(f"Input\nTrue: {true_label} | Predicted: {pred_label}")
    axes[0].axis("off")

    sv = shap_values.values[0, ..., 0]
    sv_gray = sv.sum(axis=-1)
    vmax = np.abs(sv_gray).max() if np.abs(sv_gray).max() > 0 else 1.0
    axes[1].imshow(display_img)
    im = axes[1].imshow(sv_gray, cmap="bwr", alpha=0.5, vmin=-vmax, vmax=vmax)
    axes[1].set_title("SHAP\n(red = pushes toward predicted class)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    out_path = out_dir / f"shap_{tag}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="Subset size (default 25)")
    parser.add_argument("--shap-evals", type=int, default=300,
                         help="SHAP max_evals per image (default 300)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Which seeded checkpoint/predictions to use (must match "
                              "the --seed used for train.py / evaluate.py).")
    args = parser.parse_args()

    suffix = "" if args.seed == DEFAULT_SEED else f"_seed{args.seed}"
    checkpoint_path = ROOT / "outputs" / f"checkpoints{suffix}" / "best_model.pt"
    predictions_csv = ROOT / "outputs" / f"evaluation{suffix}" / "test_predictions.csv"
    out_dir = ROOT / "outputs" / f"shap_lime{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed={args.seed} | checkpoint: {checkpoint_path} | subset size: {args.n}")

    model = MultiScaleEfficientNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    predict_fn = make_predict_fn(model, device)

    subset = pick_subset(args.n, predictions_csv)
    print(f"Selected {len(subset)} images (stratified across {FLAT_LABELS})")

    t_start = time.time()
    for idx, row in subset.iterrows():
        tag = f"{idx:03d}_{row.true_label}_as_{row.pred_label}"
        display_img = get_display_image(row.image_path)

        t0 = time.time()
        lime_path = run_lime(predict_fn, display_img, row.true_label, row.pred_label, tag, out_dir)
        t_lime = time.time() - t0
        print(f"[{idx+1}/{len(subset)}] LIME done ({t_lime:.1f}s) -> {lime_path.name}")

        t0 = time.time()
        shap_path = run_shap(predict_fn, display_img, row.true_label, row.pred_label, tag,
                              out_dir, max_evals=args.shap_evals)
        t_shap = time.time() - t0
        print(f"[{idx+1}/{len(subset)}] SHAP done ({t_shap:.1f}s) -> {shap_path.name}")

    total = time.time() - t_start
    print(f"\nAll done in {total/60:.1f} minutes. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
