"""
Unifies all downloaded CXR datasets into one manifest CSV with columns:
    image_path, label, source_dataset

Then produces stratified train/val/test split manifests.

This implements O1 (dataset integration) and part of O2 (harmonization —
label harmonization specifically; pixel-level preprocessing happens later
in src/preprocessing.py) from your Phase 1 report.

Run from the project root:
    python scripts/build_manifest.py

If your downloaded folder names differ from data/DATASET_ACQUISITION.md,
adjust the *_DIR constants and label-folder maps below rather than renaming
your raw data.
"""
import csv
from pathlib import Path
from collections import Counter

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split



# ---------------------------------------------------------------------------
# Config — adjust these if your folder names differ after download
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST_DIR = ROOT / "data" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".png", ".jpg", ".jpeg"}

# Unified label set for this project (lung cancer/CT excluded — modality
# mismatch with the CXR-only pipeline; revisit as a stretch goal in Phase 2
# discussion if time permits)
LABELS = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]


def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    """Verify every image path actually loads before it goes into the
    manifest. Catches macOS resource-fork files that slipped past the name
    filter, truncated downloads, and any other silently corrupt files —
    across ALL source datasets, not just the one we already know about."""
    print("\nValidating images (this reads every file once, ~1-2 min for 40k+ images)...")
    good_mask = []
    bad_paths = []
    for i, path in enumerate(df["image_path"]):
        if i % 5000 == 0 and i > 0:
            print(f"  checked {i}/{len(df)}...")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ok = img is not None and img.size > 0
        good_mask.append(ok)
        if not ok:
            bad_paths.append(path)

    if bad_paths:
        print(f"\n[WARN] {len(bad_paths)} unreadable images dropped from manifest:")
        for p in bad_paths[:10]:
            print(f"    {p}")
        if len(bad_paths) > 10:
            print(f"    ... and {len(bad_paths) - 10} more")
    else:
        print("All images loaded successfully — none dropped.")

    return df[pd.Series(good_mask, index=df.index)].reset_index(drop=True)


def find_images(base_dir: Path):
    results = []
    for p in base_dir.rglob("*"):
        if p.suffix.lower() not in VALID_EXT:
            continue
        # Skip macOS zip artifacts: __MACOSX resource-fork folders and
        # "._filename" AppleDouble files — these have image extensions but
        # are metadata, not real images, and will fail to load.
        if p.name.startswith("._") or "__MACOSX" in p.parts:
            continue
        results.append(p)
    return results


# ---------------------------------------------------------------------------
# Per-dataset parsers
# ---------------------------------------------------------------------------
def parse_kaggle_pneumonia(rows):
    base = RAW / "kaggle_pneumonia"
    if not base.exists():
        print(f"[skip] {base} not found")
        return
    for img in find_images(base):
        parent = img.parent.name.upper()
        if parent == "NORMAL":
            label = "Normal"
        elif parent == "PNEUMONIA":
            label = "Pneumonia"
        else:
            continue
        rows.append((str(img), label, "kaggle_pneumonia"))


def parse_covid_qu_ex(rows):
    base_root = RAW / "covid_qu_ex"
    if not base_root.exists():
        print(f"[skip] {base_root} not found")
        return

    # COVID-QU-Ex ships two overlapping folders:
    #   - "Infection Segmentation Data": smaller (~5.8k), has infection masks
    #   - "Lung Segmentation Data": larger (~33.9k), has lung masks only
    # Use ONLY "Lung Segmentation Data" (the superset) to get full coverage
    # without double-counting images that appear in both.
    candidates = list(base_root.rglob("Lung Segmentation Data"))
    source_label = "Lung Segmentation Data"
    if not candidates:
        # fall back to the smaller set if that's all the user downloaded
        candidates = list(base_root.rglob("Infection Segmentation Data"))
        source_label = "Infection Segmentation Data"
    if not candidates:
        print(f"[skip] Neither segmentation folder found under {base_root}")
        return
    base = max(candidates, key=lambda p: len(p.parts))  # deepest nested match
    print(f"[covid_qu_ex] Using '{source_label}' as source")

    label_map = {
        "covid-19": "COVID-19",
        "covid19": "COVID-19",
        "non-covid": "Pneumonia",   # non-COVID viral/bacterial pneumonia
        "noncovid": "Pneumonia",
        "normal": "Normal",
    }
    skipped_masks = 0
    for img in find_images(base):
        # Only take actual X-rays, not lung/infection segmentation masks,
        # which live in sibling "lung masks" / "infection masks" folders.
        if img.parent.name.strip().lower() != "images":
            skipped_masks += 1
            continue
        matched = None
        for part in img.parts:
            key = part.lower().replace(" ", "").replace("_", "-")
            if key in label_map:
                matched = label_map[key]
                break
        if matched:
            rows.append((str(img), matched, "covid_qu_ex"))
    if skipped_masks:
        print(f"[covid_qu_ex] Skipped {skipped_masks} mask files (not X-rays)")


def parse_tb_dataset(rows, folder_name, source_name):
    base = RAW / folder_name
    if not base.exists():
        print(f"[skip] {base} not found")
        return
    for img in find_images(base):
        stem = img.stem
        if stem.endswith("_1"):
            label = "Tuberculosis"
        elif stem.endswith("_0"):
            label = "Normal"
        else:
            continue
        rows.append((str(img), label, source_name))


def parse_qatar_tb(rows):
    """Qatar University / U. Dhaka 'Tuberculosis (TB) Chest X-ray Database'
    (Kaggle: tawsifurrahman/tuberculosis-tb-chest-xray-dataset).
    Ships as class-named folders (Normal/, Tuberculosis/) rather than the
    filename-suffix convention used by Shenzhen/Montgomery, so this gets its
    own parser instead of reusing parse_tb_dataset(). Adds ~700 new TB images
    and ~3500 new Normal images on top of the existing manifest — added
    specifically to reduce the TB:Normal/Pneumonia imbalance measured on the
    Shenzhen+Montgomery-only manifest (previously ~39:1)."""
    base = RAW / "qatar_tb"
    if not base.exists():
        print(f"[skip] {base} not found")
        return
    n_tb, n_normal = 0, 0
    for img in find_images(base):
        parent = img.parent.name.strip().lower()
        if parent == "tuberculosis":
            label = "Tuberculosis"
            n_tb += 1
        elif parent == "normal":
            label = "Normal"
            n_normal += 1
        else:
            # Some releases of this dataset include a "Lung Masks" sibling
            # folder — skip anything that isn't clearly Normal/Tuberculosis.
            continue
        rows.append((str(img), label, "qatar_tb"))
    print(f"[qatar_tb] Added {n_tb} TB, {n_normal} Normal images")


def parse_nih(rows):
    base = RAW / "nih_chestxray14"
    csv_path = base / "Data_Entry_2017.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} not found")
        return

    # Build filename -> full path index (images may be nested in subfolders
    # like images_001/images/, images_002/images/, etc.)
    index = {p.name: p for p in find_images(base)}

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        finding = str(row["Finding Labels"]).strip()
        fname = row["Image Index"]
        if finding == "No Finding":
            label = "Normal"
        elif finding == "Pneumonia":
            label = "Pneumonia"
        else:
            continue  # skip multi-label / other-pathology rows for this project's scope
        img_path = index.get(fname)
        if img_path:
            rows.append((str(img_path), label, "nih_chestxray14"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = []
    parse_kaggle_pneumonia(rows)
    parse_covid_qu_ex(rows)
    parse_tb_dataset(rows, "shenzhen", "shenzhen_tb")
    parse_tb_dataset(rows, "montgomery", "montgomery_tb")
    parse_qatar_tb(rows)
    parse_nih(rows)

    if not rows:
        print("\n[ERROR] No images found. Have you downloaded the datasets into data/raw/ yet?")
        print("        See data/DATASET_ACQUISITION.md")
        return

    df = pd.DataFrame(rows, columns=["image_path", "label", "source_dataset"])
    df = df.drop_duplicates(subset="image_path").reset_index(drop=True)
    df = validate_images(df)

    full_path = MANIFEST_DIR / "full_manifest.csv"
    df.to_csv(full_path, index=False)

    print(f"\nSaved unified manifest: {full_path}")
    print(f"Total images: {len(df)}\n")

    print("Class distribution:")
    counts = Counter(df["label"])
    for label in LABELS:
        n = counts.get(label, 0)
        print(f"  {label:15s}: {n}")

    print("\nBy source dataset:")
    print(df.groupby(["source_dataset", "label"]).size())

    # Flag imbalance ratio (your report cites up to 10:1 TB:Normal/Pneumonia)
    if counts:
        max_class = max(counts.values())
        min_class = min(v for v in counts.values() if v > 0)
        ratio = max_class / min_class
        print(f"\nMax/min class ratio: {ratio:.1f}:1")
        if ratio > 5:
            print("[NOTE] Significant imbalance — plan to use weighted sampling or class weights in training.")

    # Stratified split: 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(MANIFEST_DIR / "train.csv", index=False)
    val_df.to_csv(MANIFEST_DIR / "val.csv", index=False)
    test_df.to_csv(MANIFEST_DIR / "test.csv", index=False)

    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print("Saved train.csv / val.csv / test.csv to data/manifests/")


if __name__ == "__main__":
    main()
