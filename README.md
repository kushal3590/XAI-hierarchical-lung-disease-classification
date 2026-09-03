# XAI-Driven Multiscale Framework for Hierarchical Lung Disease Classification

Final-year engineering project (BMSCE, Dept. of CSE) — a hierarchical,
multi-scale EfficientNet-B0 classifier for chest X-rays (Normal / Pneumonia /
Tuberculosis / COVID-19), with Grad-CAM, SHAP, and LIME explainability, and
an interactive demo app.

**Team:** Kushal Naidu N, Mahesha GS, Nagaraja G
**Guide:** Dr. Bhavani K, Dept. of CSE, BMSCE

## Results (finalized model: shortcut-masking + Stage 1 reweighting, threshold-tuned)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.89 | 0.95 | 0.92 |
| Pneumonia | 0.96 | 0.88 | 0.92 |
| Tuberculosis | 0.82 | 0.96 | 0.88 |
| COVID-19 | 0.96 | 0.96 | 0.96 |

**Overall test accuracy: 93.2%** (up from a 90.3% initial baseline — see
`outputs/evaluation_masked/` for full artifacts, and the project history
below for how this was reached).

Hierarchical vs. flat-baseline accuracy advantage confirmed significant via
McNemar's test across 2 independent seeds (seed 42: χ²=4.11, p=0.0425; seed
123: χ²=14.13, p=0.0002).

## Repo layout

```
data/manifests/     # small CSVs (image paths + labels) — tracked in git
data/raw/           # actual downloaded images — NOT in git, see below
src/                # model, dataset, preprocessing, metrics
scripts/            # training, evaluation, explainability, analysis
app/                # Flask demo app (app_flask.py is the working version)
outputs/            # results — checkpoints/bulk images excluded from git,
                    #   small evaluation artifacts (reports, confusion
                    #   matrices, summary CSVs) are tracked
```

## Setup on a new machine

1. Clone the repo and create a virtual environment:
   ```
   git clone <repo-url>
   cd lung-disease-project
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. **Get the data.** `data/raw/` is not in git (too large). Follow
   `data/DATASET_ACQUISITION.md` to download the 5 source datasets, then
   rebuild the manifests (this also fixes absolute file paths for your
   machine):
   ```
   python scripts/build_manifest.py
   ```

3. **Get a trained model.** Two options:
   - **Fastest — download the finalized checkpoint:** [add your
     Google Drive / GitHub Release link here] and place it at
     `outputs/checkpoints_masked/best_model.pt`.
   - **Or retrain from scratch** (~hours on a single GPU):
     ```
     python scripts/train.py --tag masked --mask-shortcut-regions --stage1-pos-weight 1.3
     python scripts/evaluate.py --tag masked --stage1-threshold 0.25
     ```

4. **Run the demo app:**
   ```
   python app/app_flask.py
   ```
   Open `http://127.0.0.1:5000`.

## Reproducing the full pipeline

```
python scripts/train.py --tag masked --mask-shortcut-regions --stage1-pos-weight 1.3
python scripts/train_baseline.py
python scripts/evaluate.py --tag masked --stage1-threshold 0.25
python scripts/evaluate_baseline.py
python scripts/mcnemar_test.py
python scripts/gradcam_explain.py --all-test
python scripts/analyze_gradcam_attention.py
python scripts/shap_lime_explain.py --n 150
```

Add `--seed 123` to any script to reproduce the second-seed run used for
statistical validation.

## Known issues (both have working fixes already applied)

- **sklearn + tensorboard native-DLL conflict (Windows):** can silently
  crash `train.py` mid-run with no traceback. Fixed via `src/metrics.py`
  (numpy-only shim), already wired into `train.py`.
- **Streamlit / Gradio unresolved native crash (Windows):** the demo app
  was ported to Flask (`app/app_flask.py`) after both frameworks hit a
  persistent `STATUS_ACCESS_VIOLATION` crash on this project's dev machine.
  `app/archive/` holds the earlier attempts for reference — not
  actively maintained.

## Project history / key findings (for the report's Discussion section)

- Hierarchical model significantly outperforms a flat-classification
  baseline (McNemar's test, 2 seeds).
- Grad-CAM + SHAP + LIME (full test set for Grad-CAM, 150-image stratified
  subset for SHAP/LIME) found Stage 1 sometimes relies on the mediastinal
  midline instead of lung tissue — quantified at 36.4% of missed Pneumonia
  cases vs. 4.6% of correctly-flagged ones (~8x).
- Accuracy improved 90.3% → 93.2% via: (1) Stage 1 threshold tuning
  (validation-set selected), (2) shortcut-region masking augmentation
  during training, (3) Stage 1 class reweighting to address Pneumonia
  recall specifically.
