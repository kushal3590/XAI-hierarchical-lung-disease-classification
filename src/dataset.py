"""
PyTorch Dataset + DataLoader utilities for the lung disease classifier.

Implements:
  - FR3 (data augmentation): flip, rotation, zoom, brightness — train split only
  - The 2-stage hierarchical label scheme (see chat discussion):
        stage1: 0 = Normal, 1 = Abnormal
        stage2: 0 = Pneumonia, 1 = Tuberculosis, 2 = COVID-19
                (ignore_index = -1 for Normal rows, since stage2 doesn't
                 apply to them — handled via CrossEntropyLoss(ignore_index=-1))
  - A weighted sampler to counter the ~50:1 TB imbalance you measured

"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from preprocessing import (
    load_grayscale, apply_clahe, resize, grayscale_to_rgb,
    IMAGENET_MEAN, IMAGENET_STD,
)

FLAT_LABELS = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
FLAT_LABEL_TO_IDX = {l: i for i, l in enumerate(FLAT_LABELS)}

# Stage 2 only covers the 3 abnormal categories
STAGE2_LABELS = ["Pneumonia", "Tuberculosis", "COVID-19"]
STAGE2_LABEL_TO_IDX = {l: i for i, l in enumerate(STAGE2_LABELS)}
STAGE2_IGNORE_INDEX = -1


def seed_worker(worker_id):
    """DataLoader worker_init_fn (Phase 1 NFR1 reproducibility). Each worker
    process forks with its own RNG state, so without this the per-sample
    augmentation randomness in CXRDataset._augment (flip/rotation/zoom/
    brightness — all np.random calls) is NOT reproducible run-to-run when
    num_workers > 0, even with the main-process seed fixed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CXRDataset(Dataset):
    def __init__(self, manifest_csv: str, train: bool = False, use_clahe: bool = True,
                 mask_shortcut_regions: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.train = train
        self.use_clahe = use_clahe
        # Opt-in only — existing training runs (seed 42/123) are unaffected
        # unless this is explicitly set. See _mask_shortcut_regions() below.
        self.mask_shortcut_regions = mask_shortcut_regions

        # Precompute hierarchical labels once
        self.stage1 = (self.df["label"] != "Normal").astype(int).values
        self.stage2 = self.df["label"].apply(
            lambda l: STAGE2_LABEL_TO_IDX.get(l, STAGE2_IGNORE_INDEX)
        ).values
        self.flat = self.df["label"].map(FLAT_LABEL_TO_IDX).values

    def __len__(self):
        return len(self.df)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """FR3: random horizontal flip, rotation up to 15deg, zoom, brightness.
        Applied on the grayscale image BEFORE RGB conversion/normalization,
        operating in uint8 space for simplicity with OpenCV."""
        h, w = img.shape[:2]

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)

        angle = np.random.uniform(-15, 15)
        zoom = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        brightness_delta = np.random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)

        return img

    def _mask_shortcut_regions(self, img: np.ndarray) -> np.ndarray:
        """Randomly blacks out the mediastinal midline strip and/or top
        corners during training, to discourage Stage 1 from relying on
        these non-lung-parenchyma shortcuts. Directly informed by this
        project's own full-test-set Grad-CAM attention-zone analysis:
        Pneumonia images misclassified as Normal showed the model's Stage 1
        attention peak on the midline in 36.4% of cases, vs only 4.6% for
        correctly-classified Pneumonia (~8x). Corner-marker attention was
        also flagged as a known shortcut in the SHAP/LIME/Grad-CAM findings.

        Applied AFTER geometric augmentation (flip/rotate/zoom), so the
        masked regions correspond to the SAME final image-space corners/
        midline that classify_attention_zone() checks at inference time —
        keeping the fix consistent with how the problem was measured.

        Each region is masked independently and randomly per-sample (not
        every image, every epoch) so the model still sees the full image
        most of the time — this is a nudge away from the shortcut, not a
        hard removal of that information.
        """
        h, w = img.shape[:2]
        img = img.copy()

        # Midline strip (mediastinum/trachea) — matches the zone definition
        # in scripts/gradcam_explain.py's classify_attention_zone()
        if np.random.rand() < 0.30:
            strip_half_width = int(0.09 * w)
            cx = w // 2
            fill = int(np.random.randint(0, 40))  # dark, roughly background-level
            img[:, max(0, cx - strip_half_width):min(w, cx + strip_half_width)] = fill

        # Top-left / top-right corners (position/laterality markers) —
        # matches classify_attention_zone()'s corner_frac definition
        corner_h = int(0.18 * h)
        corner_w = int(0.18 * w)
        if np.random.rand() < 0.25:
            fill = int(np.random.randint(0, 40))
            img[0:corner_h, 0:corner_w] = fill
        if np.random.rand() < 0.25:
            fill = int(np.random.randint(0, 40))
            img[0:corner_h, w - corner_w:w] = fill

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]

        gray = load_grayscale(path)
        if self.use_clahe:
            gray = apply_clahe(gray)
        gray = resize(gray)

        if self.train:
            gray = self._augment(gray)
            if self.mask_shortcut_regions:
                gray = self._mask_shortcut_regions(gray)

        rgb = grayscale_to_rgb(gray)
        img = rgb.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        return {
            "image": torch.from_numpy(img).float(),
            "stage1": torch.tensor(self.stage1[idx], dtype=torch.long),
            "stage2": torch.tensor(self.stage2[idx], dtype=torch.long),
            "flat": torch.tensor(self.flat[idx], dtype=torch.long),
            "path": path,
        }


def make_weighted_sampler(dataset: CXRDataset, generator: torch.Generator = None) -> WeightedRandomSampler:
    """Inverse-frequency sampling on the FLAT label (Normal/Pneumonia/TB/COVID)
    so that rare classes (TB) are seen far more often per epoch than their raw
    count would otherwise allow. This is the primary lever against the ~50:1
    imbalance measured on your manifest.

    `generator` should be a seeded torch.Generator (Phase 1 NFR1) — without
    it, WHICH samples get picked each epoch (not just augmentation) is
    non-reproducible, since replacement=True sampling draws from its own
    RNG stream independent of torch.manual_seed()."""
    flat = dataset.flat
    class_counts = np.bincount(flat, minlength=len(FLAT_LABELS))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[flat]

    print("Class counts:", dict(zip(FLAT_LABELS, class_counts)))
    print("Class weights (inverse freq):", dict(zip(FLAT_LABELS, class_weights)))

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def get_dataloaders(manifest_dir: str, batch_size: int = 24, num_workers: int = 4,
                     seed: int = None, mask_shortcut_train: bool = False):
    """Builds train/val/test DataLoaders.

    If `seed` is provided (Phase 1 NFR1), the weighted sampler and all
    DataLoader worker processes are seeded deterministically, so both
    *which* training samples get drawn each epoch and the per-sample
    augmentation applied to them are reproducible across runs. Pass the
    same seed you used for set_seed() in the calling training script.

    If `mask_shortcut_train` is True, the TRAINING set only gets the
    shortcut-region masking augmentation (see CXRDataset._mask_shortcut_regions).
    val/test are never masked — they must reflect real, unmodified images."""
    manifest_dir = Path(manifest_dir)

    train_ds = CXRDataset(manifest_dir / "train.csv", train=True,
                          mask_shortcut_regions=mask_shortcut_train)
    val_ds = CXRDataset(manifest_dir / "val.csv", train=False)
    test_ds = CXRDataset(manifest_dir / "test.csv", train=False)

    generator = None
    worker_init = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init = seed_worker

    sampler = make_weighted_sampler(train_ds, generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=generator,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=generator,
    )

    return train_loader, val_loader, test_loader
