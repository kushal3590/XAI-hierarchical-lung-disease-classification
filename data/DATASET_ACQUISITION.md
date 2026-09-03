# Dataset Acquisition Guide

Download each dataset into `data/raw/<dataset_name>/` exactly as named below —
`build_manifest.py` expects these folder names. This is O1 from your Phase 1
report (multi-source CXR dataset unification).

## 1. Kaggle CLI setup (needed for most of these)
1. Create a Kaggle account, go to Account → Create New API Token → downloads `kaggle.json`.
2. Place it at `C:\Users\<you>\.kaggle\kaggle.json`.
3. `pip install kaggle` (already in requirements.txt).

## 2. Datasets

### Kaggle Chest X-ray Pneumonia (~1.2GB)
```
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw/kaggle_pneumonia --unzip
```
Expected structure after unzip: `data/raw/kaggle_pneumonia/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg`

### COVID-QU-Ex (~2.6GB)
```
kaggle datasets download -d anasmohammedtahir/covidqu -p data/raw/covid_qu_ex --unzip
```
Expected structure: folders containing `COVID-19`, `Non-COVID`, `Normal` subfolders
(exact nesting varies by version — the manifest script scans recursively by folder name,
so this is tolerant of some structural variation. Open the folder once after download
and confirm class folder names match COVID-19 / Non-COVID / Normal, adjust
`COVID_QU_LABEL_MAP` in build_manifest.py if they differ).

### Montgomery + Shenzhen TB datasets (~100MB combined)
```
kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen -p data/raw/shenzhen --unzip
kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery -p data/raw/montgomery --unzip
```
Filenames encode the label: `CHNCXR_00XX_0.png` = Normal, `CHNCXR_00XX_1.png` = TB
(Shenzhen). Montgomery uses a similar `_0`/`_1` suffix.

### NIH ChestX-ray14 (~45GB full — recommend the resized/subset version to start)
Full: https://nihcc.app.box.com/v/ChestXray-NIHCC (official NIH page, not Kaggle)
Lighter alternative for a first pass:
```
kaggle datasets download -d nih-chest-xrays/data -p data/raw/nih_chestxray14 --unzip
```
This includes `Data_Entry_2017.csv` with per-image labels. We only pull the
"No Finding" (→ Normal) and "Pneumonia" (→ Pneumonia) single-label rows for this
project's 4/5-class scope — see `build_manifest.py`.

> Given your local SSD (1TB free) this all fits comfortably. If you're short on
> time before a checkpoint deadline, start with kaggle_pneumonia + covid_qu_ex +
> shenzhen/montgomery first (small, fast) and add NIH once the pipeline works
> end-to-end — validate the pipeline on a smaller footprint before the big pull.

## 3. Folder layout after downloading everything
```
data/raw/
  kaggle_pneumonia/chest_xray/...
  covid_qu_ex/...
  shenzhen/...
  montgomery/...
  nih_chestxray14/
    images/*.png
    Data_Entry_2017.csv
```

Once these are in place, run:
```
python scripts/build_manifest.py
```
