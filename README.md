# XAI-Driven Multiscale Framework for Hierarchical Lung Disease Classification

A two-stage, multi-scale EfficientNet-B0 classifier for chest X-rays
(Normal / Pneumonia / Tuberculosis / COVID-19) with Grad-CAM, SHAP and LIME
explanations and a local web app. Final-year project, Dept. of CSE, BMS College
of Engineering.

> **Research prototype. Not a medical device.** It has not been clinically
> validated and must not be used for diagnosis or treatment decisions.

**Team:** Kushal Naidu N, Mahesha GS, Nagaraja G  
**Guide:** Dr. Bhavani K, Assistant Professor, Dept. of CSE, BMSCE

---

## Contents

1. [Quick start: run the demo app](#1-quick-start-run-the-demo-app-5-minutes-no-dataset-needed)
2. [Full setup: data and training](#2-full-setup-data-and-training)
3. [Results](#3-results)
4. [Reproducing the experiments](#4-reproducing-the-experiments)
5. [Repository layout](#5-repository-layout)
6. [Troubleshooting](#6-troubleshooting)
7. [Key findings and limitations](#7-key-findings-and-limitations)

---

## 1. Quick start: run the demo app (5 minutes, no dataset needed)

The trained model is included in the repository at
`outputs/checkpoints_masked/best_model.pt`, so you can run the app without
downloading any dataset or training anything.

**You need:** Python 3.10 or newer (3.11 or 3.12 recommended), Git (or use
"Download ZIP" on GitHub), and about 3 GB of free disk space for the Python
packages. A GPU is not required; the app runs on CPU.

### Step 1. Get the code

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

No Git? On the GitHub page click **Code → Download ZIP**, unzip it, and open a
terminal inside the unzipped folder.

### Step 2. Create a virtual environment and install packages

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r app/requirements.txt
```
If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process Bypass` once in that window, then retry.
In Command Prompt use `venv\Scripts\activate.bat` instead.

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
```

### Step 3. Check the model file is there

```bash
# Windows PowerShell
Test-Path outputs\checkpoints_masked\best_model.pt

# macOS / Linux
ls -lh outputs/checkpoints_masked/best_model.pt
```
It should exist and be about 17 MB. If it is missing, see
[Troubleshooting](#6-troubleshooting).

### Step 4. Start the app

```bash
python app/app_flask.py
```

Open **http://127.0.0.1:5000** in your browser, upload any frontal chest
X-ray (JPG or PNG), and press **Analyze**.

- Prediction and Grad-CAM heatmaps take a few seconds on a laptop CPU.
- SHAP is optional and slower (roughly a minute on CPU).
- Images are processed locally on your own computer and are not sent anywhere.

Stop the app with `Ctrl + C` in the terminal.

---

## 2. Full setup: data and training

Only needed if you want to retrain, re-evaluate or regenerate the figures.

### 2.1 Install everything

```bash
pip install -r requirements.txt
```
This installs the extra packages used by the training, statistics and
explainability scripts. For GPU training, install the CUDA build of PyTorch
that matches your driver from <https://pytorch.org/get-started/locally/>
**before** running the command above.

Training used a laptop GPU with 6 GB of VRAM (batch size 16, mixed
precision). Training on CPU works but is very slow.

### 2.2 Download the datasets

The images are not in this repository (about 45,000 images, several GB). Five
public collections are used: the Kaggle Chest X-ray (Pneumonia) set,
COVID-QU-Ex, the Shenzhen and Montgomery TB sets, and the Qatar University TB
chest X-ray database.

1. Follow **`data/DATASET_ACQUISITION.md`** for exact download commands (a free
   Kaggle account and API token are needed for the Kaggle-hosted sets).
2. Place each collection under `data/raw/` in these folders:
   ```
   data/raw/kaggle_pneumonia/
   data/raw/covid_qu_ex/
   data/raw/shenzhen/
   data/raw/montgomery/
   data/raw/qatar_tb/
   ```
3. Build the manifests. **Do this on every new machine**: the manifests
   store file paths, and the committed copies point to the original
   developer's computer.
   ```bash
   python scripts/build_manifest.py
   ```
   This validates every image and writes `data/manifests/train.csv`,
   `val.csv`, `test.csv` and `full_manifest.csv` (a stratified 70/15/15
   split).

### 2.3 Train the final model

```bash
python scripts/train.py --tag masked --mask-shortcut-regions --stage1-pos-weight 1.3
python scripts/evaluate.py --tag masked --stage1-threshold 0.25
```
Training takes several hours on a 6 GB laptop GPU. The checkpoint is written
to `outputs/checkpoints_masked/best_model.pt`, which is the path the app loads
by default.

---

## 3. Results

Final model (shortcut-region masking, Stage 1 class weighting, Stage 1
threshold 0.25 chosen on the validation set). Internal test set, 6,717 images:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 88.6% | 95.3% | 91.8% |
| Pneumonia | 96.1% | 88.4% | 92.1% |
| Tuberculosis | 81.8% | 95.7% | 88.2% |
| COVID-19 | 97.7% | 96.4% | 97.1% |
| **Overall** | accuracy **93.2%** | macro-F1 **92.3%** | one-vs-rest AUC 0.983 to 0.999 |

Full artifacts (classification report, confusion matrix, ROC curves, per-image
predictions) are in `outputs/evaluation_masked/`.

> **Read this as an optimistic internal estimate.** The split is at image
> level, and an audit found that at least 21.7% of test images come from
> patients who also appear in training. Patient-level and external
> validation have not been done. See [Section 7](#7-key-findings-and-limitations).

**Hierarchical vs. flat classifier** (same backbone and data, original
recipe): the two-stage design was more accurate under two random seeds
(McNemar's test: seed 42, p = 0.043; seed 123, p = 0.0002).

**EfficientNet-B0 vs. DenseNet-121** (same splits, seed, schedule and recipe):

| Model | Test accuracy | Correct images |
|---|---|---|
| EfficientNet-B0 (final) | 93.2% | 6,261 / 6,717 |
| DenseNet-121 | 91.8% | 6,167 / 6,717 |

The difference is statistically significant (McNemar's χ² = 19.7,
p ≈ 9 × 10⁻⁶). The recipe was developed on EfficientNet and applied unchanged
to DenseNet, and each model's Stage 1 threshold was tuned on its own
validation set.

---

## 4. Reproducing the experiments

Run from the repository root with the environment active and the manifests
built (Section 2).

```bash
# Main model
python scripts/train.py --tag masked --mask-shortcut-regions --stage1-pos-weight 1.3
python scripts/tune_stage1_threshold.py --tag masked
python scripts/evaluate.py --tag masked --stage1-threshold 0.25

# Flat baseline and significance test (original recipe)
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
python scripts/mcnemar_test.py

# Explainability
python scripts/gradcam_explain.py --all-test
python scripts/analyze_gradcam_attention.py
python scripts/shap_lime_explain.py --n 150

# DenseNet-121 comparison (writes only to outputs/densenet/)
python scripts/train_densenet.py --tag masked --mask-shortcut-regions --stage1-pos-weight 1.3
python scripts/evaluate_densenet.py --tune --tag masked
python scripts/compare_models.py
```

Add `--seed 123` to the training and evaluation scripts to reproduce the
second-seed run. Each script's docstring explains its options.

---

## 5. Repository layout

```
data/
  DATASET_ACQUISITION.md    how to download the five datasets
  manifests/                image paths and labels (rebuild on each machine)
  raw/                      downloaded images (not in git)
src/
  preprocessing.py          CLAHE, resize, normalization
  dataset.py                Dataset, weighted sampler, augmentation, masking
  model.py                  hierarchical multi-scale EfficientNet-B0
  model_densenet.py         DenseNet-121 comparison models
  metrics.py                numpy metrics used by training (see Troubleshooting)
scripts/                    training, evaluation, statistics, explainability
app/
  app_flask.py              demo web app
  templates/                HTML templates
  archive/                  earlier Streamlit and Gradio attempts (unmaintained)
outputs/
  checkpoints_masked/       final model (best_model.pt)
  evaluation*/              reports, confusion matrices, ROC curves, predictions
  densenet/                 DenseNet-121 results and the model comparison
```

Large generated folders (extra checkpoints, full Grad-CAM image sets, SHAP and
LIME images, TensorBoard logs) are not stored in git; the scripts above
regenerate them.

---

## 6. Troubleshooting

**"Checkpoint not found" when starting or using the app.**
Confirm `outputs/checkpoints_masked/best_model.pt` exists (Section 1, Step 3).
If you downloaded a ZIP or cloned a partial copy and it is missing, retrain
(Section 2.3) or copy the file into that folder.

**Image or file-not-found errors in the training or evaluation scripts.**
The manifests contain paths from another computer. Run
`python scripts/build_manifest.py` on your machine after placing the data.

**`pip install` fails or is very slow for torch.**
Install PyTorch first from <https://pytorch.org/get-started/locally/> (choose
CPU if you have no NVIDIA GPU), then rerun the `pip install -r` command.

**Port 5000 is already in use.**
Stop the other program using it, or edit the `port=5000` value near the bottom
of `app/app_flask.py`. macOS can reserve port 5000 for AirPlay; changing the
port fixes that.

**`train.py` exits silently on Windows with no error.**
On some Windows machines scikit-learn and TensorBoard load conflicting native
libraries. The training scripts already use `src/metrics.py` (a numpy-only
replacement) to avoid this; scikit-learn is still used by the evaluation
scripts.

**The app crashes at startup with no message.**
This project's first two app versions (Streamlit and Gradio, kept in
`app/archive/`) hit an unresolved native crash on one Windows machine, which is
why the app is built on Flask. If Flask also fails, run
`python app/diagnose_imports.py` to find which library breaks, and try a clean
virtual environment with Python 3.11 or 3.12.

**SHAP is very slow.**
It re-runs the model hundreds of times and is much slower on CPU. Leave the
SHAP option off for quick checks.

---

## 7. Key findings and limitations

**Findings**
- The two-stage (hierarchical) design beat a flat classifier on the same
  backbone, under two seeds.
- The largest remaining weakness is the Stage 1 gate: 65% of final errors are
  abnormal images passed as Normal.
- A full-test-set Grad-CAM audit found that in missed Pneumonia cases the
  model's attention peaked on the central chest in 36.4% of images, versus
  4.6% in correctly caught cases (about 8 times more often). This motivated
  the masking augmentation, and the web app flags results where attention
  lands outside the lungs.
- Accuracy rose from 90.3% to 93.2% through threshold tuning (chosen on
  validation data), shortcut-region masking, and Stage 1 class weighting. The
  three changes were made together, so the gain cannot be attributed to any
  one of them.

**Limitations**
- Internal evaluation only: image-level split, at least 21.7% of test images
  share a patient with training, and some source collections overlap.
- Accuracy varies by source dataset, so part of the performance reflects
  dataset and label effects (for example, the non-COVID infection images
  account for almost all Pneumonia errors).
- Tuberculosis has only 164 test images, so TB-specific claims have low
  statistical power.
- Heatmaps are qualitative; no faithfulness test has been done, and the
  attention zones are a geometric heuristic, not a lung segmentation.
- Not clinically validated and not a medical device.

---

## License and data

Code is released under the MIT License (see `LICENSE`). The datasets are
**not** redistributed here; each has its own licence and terms of use, which
you must follow when downloading them.
