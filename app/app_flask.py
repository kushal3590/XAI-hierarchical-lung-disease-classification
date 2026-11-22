"""
Flask + plain HTML version of the lung disease XAI demo app.

Third framework attempt after Streamlit and Gradio both hit an unresolved
native crash (STATUS_ACCESS_VIOLATION / 0xC0000005) on this Windows machine.
Flask's built-in dev server is single-process, single-threaded by default
(no ScriptRunner-style background thread, no request-queue worker pool),
which rules out framework-thread interaction almost entirely. If this ALSO
crashes identically, the issue is not the web framework — see the bottom of
this file for what that would mean.

Run from project root:
    python app/app_flask.py

Then open http://127.0.0.1:5000 in your browser.

UI ("PulmoLens"): multi-page front end in app/templates + app/static
  /          overview          /analyze   upload + results (calls POST /api/analyze)
  /history   scans saved in this browser (localStorage)
  /about     project + team
Fonts are self-hosted in app/static/fonts, so the UI works fully offline.
Optional: set LUNG_XAI_CHECKPOINT to point at a specific best_model.pt.

Needs: outputs/checkpoints[_seed<N>]/best_model.pt (train a model first).
"""
import sys
import os
import io
import base64
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from flask import Flask, request, render_template, jsonify
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed — we only save figures to bytes
import matplotlib.pyplot as plt
import shap
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from preprocessing import apply_clahe, resize, grayscale_to_rgb, IMAGENET_MEAN, IMAGENET_STD
from model import MultiScaleEfficientNet, combine_predictions_thresholded

ROOT = Path(__file__).resolve().parent.parent
FLAT_LABELS = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
STAGE2_LABELS = ["Pneumonia", "Tuberculosis", "COVID-19"]

app = Flask(__name__)


# =============================================================================
# Model loading (simple dict cache)
# =============================================================================
class StageWrapper(torch.nn.Module):
    def __init__(self, base_model, stage):
        super().__init__()
        self.base_model = base_model
        self.stage = stage

    def forward(self, x):
        s1, s2 = self.base_model(x)
        return s1 if self.stage == "stage1" else s2


_model_cache = {}


def load_model(checkpoint_path_str: str):
    if checkpoint_path_str in _model_cache:
        return _model_cache[checkpoint_path_str]

    checkpoint_path = Path(checkpoint_path_str)
    device = torch.device("cpu")
    model = MultiScaleEfficientNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    stage1_wrapper = StageWrapper(model, "stage1").to(device).eval()
    stage2_wrapper = StageWrapper(model, "stage2").to(device).eval()
    target_layer = [model.backbone[8]]
    cam1 = GradCAM(model=stage1_wrapper, target_layers=target_layer)
    cam2 = GradCAM(model=stage2_wrapper, target_layers=target_layer)

    result = (model, device, cam1, cam2)
    _model_cache[checkpoint_path_str] = result
    return result


def default_ckpt_path(suffix: str) -> str:
    return str(ROOT / "outputs" / f"checkpoints{suffix}" / "best_model.pt")


# Finalized model registry. "masked" is the recommended default — trained
# with shortcut-region masking + Stage 1 class reweighting (pos_weight=1.3),
# evaluated at the tuned Stage 1 threshold (0.25) selected on the validation
# set. Test-set accuracy: 93.2% (up from 90.3% baseline). See
# outputs/evaluation_masked/classification_report.txt for the full numbers.
MODEL_VARIANTS = {
    "masked": {
        "label": "Masked + reweighted (recommended — 93.2% test acc)",
        "suffix": "_masked",
        "default_threshold": 0.25,
    },
    "seed42": {
        "label": "Seed 42 baseline (90.3% test acc)",
        "suffix": "",
        "default_threshold": 0.5,
    },
    "seed123": {
        "label": "Seed 123 baseline (92.0% test acc)",
        "suffix": "_seed123",
        "default_threshold": 0.5,
    },
}
DEFAULT_VARIANT = "masked"


# =============================================================================
# Preprocessing / inference helpers
# =============================================================================
def preprocess_for_model(display_rgb_uint8: np.ndarray) -> torch.Tensor:
    img = display_rgb_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float().unsqueeze(0)


def get_combined_probs(stage1_logits, stage2_logits):
    p1 = F.softmax(stage1_logits, dim=1)
    p2 = F.softmax(stage2_logits, dim=1)
    p_normal = p1[:, 0:1]
    p_abnormal = p1[:, 1:2]
    p_disease = p_abnormal * p2
    return torch.cat([p_normal, p_disease], dim=1)


def numpy_to_base64_png(arr_uint8: np.ndarray) -> str:
    img = Image.fromarray(arr_uint8.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# Attention-zone classification — SAME peak-based logic as
# scripts/gradcam_explain.py (kept in sync manually).
# =============================================================================
def classify_attention_zone(cam, corner_frac=0.18, margin_frac=0.07, midline_half_width=0.09):
    h, w = cam.shape
    if cam.max() <= 0:
        return "undefined"
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    fy, fx = peak_y / h, peak_x / w

    in_top_corner = (fy < corner_frac) and (fx < corner_frac or fx > 1 - corner_frac)
    in_outer_margin = (fx < margin_frac) or (fx > 1 - margin_frac) or (fy < margin_frac) or (fy > 1 - margin_frac)
    in_midline = abs(fx - 0.5) < midline_half_width

    if in_top_corner:
        return "corner_marker"
    if in_outer_margin:
        return "outer_margin"
    if in_midline:
        return "midline_strip"
    return "lung_field"


ZONE_NOTES = {
    "lung_field": (
        "Stage 1's attention is centered over the lung field — anatomically "
        "consistent with typical diagnostic reasoning."
    ),
    "midline_strip": (
        "Stage 1's attention peak is on the mediastinal midline. In this project's "
        "full-test-set analysis, this pattern occurred ~8x more often in missed "
        "Pneumonia cases (36.4%) than in correctly-flagged ones (4.6%) — treat this "
        "prediction with extra caution, especially if it says 'Normal'."
    ),
    "corner_marker": (
        "Stage 1's attention peak is on a corner region — often where position/"
        "laterality markers sit on a CXR, not lung tissue. This is a known shortcut "
        "pattern flagged in this project's XAI analysis."
    ),
    "outer_margin": (
        "Stage 1's attention peak is at the image's outer edge — likely outside the "
        "anatomical body region. Treat this prediction with caution."
    ),
    "undefined": "Attention map was flat/undefined — no clear peak to assess.",
}
ZONE_LEVEL = {  # for CSS styling: "ok" (green) or "warn" (amber)
    "lung_field": "ok",
    "midline_strip": "warn",
    "corner_marker": "warn",
    "outer_margin": "warn",
    "undefined": "warn",
}


# =============================================================================
# Core analysis (framework-agnostic — just takes raw bytes, returns a dict)
# =============================================================================
def analyze(file_bytes, ckpt_path, stage1_threshold, run_shap_flag, shap_evals):
    if not Path(ckpt_path).exists():
        return {"error": f"Checkpoint not found at: {ckpt_path}"}

    model, device, cam1, cam2 = load_model(ckpt_path)

    arr = np.frombuffer(file_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"error": "Could not decode the uploaded file as an image."}

    gray = apply_clahe(gray)
    gray = resize(gray)
    display_rgb = grayscale_to_rgb(gray)
    tensor = preprocess_for_model(display_rgb).to(device)

    with torch.no_grad():
        s1_logits, s2_logits = model(tensor)
        combined_probs = get_combined_probs(s1_logits, s2_logits)[0].cpu().numpy()

    # Stage 1's Normal/Abnormal decision uses the SAME threshold as the final
    # combined prediction below — otherwise the displayed "Stage 1: Normal"
    # label, the Grad-CAM stage-2 gating, and the final predicted class could
    # silently disagree with each other on borderline-probability cases.
    stage1_probs = torch.softmax(s1_logits, dim=1)
    stage1_pred_idx = int((stage1_probs[0, 1] >= stage1_threshold).item())
    stage2_pred_idx = s2_logits.argmax(dim=1).item()
    flat_pred_idx = combine_predictions_thresholded(s1_logits, s2_logits,
                                                      abnormal_threshold=stage1_threshold).item()
    flat_pred_label = FLAT_LABELS[flat_pred_idx]
    stage1_label = "Abnormal" if stage1_pred_idx == 1 else "Normal"
    stage2_meaningful = (stage1_pred_idx == 1)

    probs_list = [
        {"label": label, "pct": round(float(p) * 100, 1)}
        for label, p in zip(FLAT_LABELS, combined_probs)
    ]

    display_float = display_rgb.astype(np.float32) / 255.0
    grayscale_cam1 = cam1(input_tensor=tensor, targets=[ClassifierOutputTarget(stage1_pred_idx)])[0]
    overlay1 = show_cam_on_image(display_float, grayscale_cam1, use_rgb=True)
    stage1_zone = classify_attention_zone(grayscale_cam1)

    if stage2_meaningful:
        grayscale_cam2 = cam2(input_tensor=tensor, targets=[ClassifierOutputTarget(stage2_pred_idx)])[0]
        overlay2 = show_cam_on_image(display_float, grayscale_cam2, use_rgb=True)
        stage2_text = STAGE2_LABELS[stage2_pred_idx]
    else:
        overlay2 = display_rgb
        stage2_text = "N/A (Stage 1 = Normal, not used)"

    result = {
        "input_img_b64": numpy_to_base64_png(display_rgb),
        "gradcam1_b64": numpy_to_base64_png(overlay1),
        "gradcam2_b64": numpy_to_base64_png(overlay2),
        "flat_pred_label": flat_pred_label,
        "confidence": f"{combined_probs[flat_pred_idx] * 100:.1f}",
        "stage1_label": stage1_label,
        "stage2_text": stage2_text,
        "probs_list": probs_list,
        "stage1_abnormal_pct": round(float(stage1_probs[0, 1]) * 100, 1),
        "stage1_threshold_pct": round(float(stage1_threshold) * 100),
        "zone": stage1_zone,
        "zone_note": ZONE_NOTES.get(stage1_zone, ""),
        "zone_level": ZONE_LEVEL.get(stage1_zone, "warn"),
        "shap_b64": None,
    }

    if run_shap_flag:
        def predict_fn(images_uint8):
            images_uint8 = np.asarray(images_uint8)
            imgs = images_uint8.astype(np.float32) / 255.0
            imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            batch = torch.from_numpy(imgs).float().to(device)
            with torch.no_grad():
                s1, s2 = model(batch)
                probs = get_combined_probs(s1, s2)
            return probs.cpu().numpy()

        masker = shap.maskers.Image("blur(32,32)", display_rgb.shape)
        explainer = shap.Explainer(predict_fn, masker, output_names=FLAT_LABELS)
        shap_values = explainer(
            np.expand_dims(display_rgb, 0),
            max_evals=int(shap_evals),
            batch_size=16,
            outputs=shap.Explanation.argsort.flip[:1],
        )
        sv = shap_values.values[0, ..., 0].sum(axis=-1)
        vmax = np.abs(sv).max() if np.abs(sv).max() > 0 else 1.0

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(display_rgb)
        im = ax.imshow(sv, cmap="bwr", alpha=0.5, vmin=-vmax, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"SHAP for predicted class: {flat_pred_label}\n(red = pushes toward this class)")
        plt.colorbar(im, ax=ax, fraction=0.046)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        result["shap_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return result


# =============================================================================
# Routes — multi-page UI + JSON analysis API
# =============================================================================
NAV = [
    ("home", "/", "Overview"),
    ("analyze", "/analyze", "Analyze"),
    ("history", "/history", "History"),
    ("about", "/about", "About"),
]


@app.context_processor
def inject_nav():
    return {"nav_items": NAV}


@app.route("/")
def home():
    return render_template("home.html", page="home")


@app.route("/analyze")
def analyze_page():
    default_threshold = MODEL_VARIANTS[DEFAULT_VARIANT]["default_threshold"]
    return render_template("analyze.html", page="analyze",
                           default_threshold_pct=round(default_threshold * 100))


@app.route("/history")
def history_page():
    return render_template("history.html", page="history")


@app.route("/about")
def about_page():
    return render_template("about.html", page="about")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON endpoint used by the Analyze page.

    Form fields (all optional except `xray`):
      xray            image file (JPG/PNG)
      model_variant   key of MODEL_VARIANTS (defaults to the recommended model)
      threshold_pct   Stage 1 abnormal cutoff, 1-99 (defaults to the variant's tuned value)
      deep            "1" to also compute SHAP
      shap_evals      SHAP max_evals (100-500)
    """
    file = request.files.get("xray")
    if not file or not file.filename:
        return jsonify({"error": "Choose a chest X-ray image (JPG or PNG) to analyze."}), 400

    variant_key = request.form.get("model_variant", DEFAULT_VARIANT)
    variant = MODEL_VARIANTS.get(variant_key, MODEL_VARIANTS[DEFAULT_VARIANT])
    ckpt_path = os.environ.get("LUNG_XAI_CHECKPOINT") or default_ckpt_path(variant["suffix"])

    try:
        threshold = float(request.form.get("threshold_pct", variant["default_threshold"] * 100)) / 100
    except ValueError:
        threshold = variant["default_threshold"]
    threshold = min(max(threshold, 0.01), 0.99)

    deep = request.form.get("deep") == "1"
    try:
        shap_evals = int(request.form.get("shap_evals", 200))
    except ValueError:
        shap_evals = 200
    shap_evals = min(max(shap_evals, 100), 500)

    try:
        result = analyze(file.read(), ckpt_path, threshold, deep, shap_evals)
    except Exception as exc:  # surface model errors to the UI instead of a 500 page
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    if "error" in result:
        return jsonify(result), 422
    return jsonify(result)


if __name__ == "__main__":
    # threaded=False, debug=False, use_reloader=False: single process, single
    # thread, no auto-restart watcher — the most conservative execution model
    # available, to rule out any framework-threading interaction with torch.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False, use_reloader=False)
