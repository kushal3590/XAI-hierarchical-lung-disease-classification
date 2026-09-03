"""
Sanity check for the data pipeline: loads one batch from the train loader,
prints shapes/dtypes, verifies the weighted sampler is actually rebalancing
classes, and saves a grid of preprocessed images so you can visually confirm
CLAHE + augmentation look reasonable (not corrupted, not over-augmented).

Run from project root:
    python scripts/sanity_check_data.py
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders, FLAT_LABELS

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"


def denormalize(img_tensor):
    """Reverse ImageNet normalization for display purposes."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    print("Building dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(MANIFEST_DIR, batch_size=16, num_workers=0)

    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    print("\nBatch contents:")
    print(f"  image shape: {batch['image'].shape}, dtype: {batch['image'].dtype}")
    print(f"  stage1 shape: {batch['stage1'].shape}, unique values: {batch['stage1'].unique().tolist()}")
    print(f"  stage2 shape: {batch['stage2'].shape}, unique values: {batch['stage2'].unique().tolist()}")
    print(f"  flat shape: {batch['flat'].shape}")

    # Confirm the weighted sampler is actually rebalancing — pull several
    # batches and tally flat-label frequency, should look far more even than
    # raw dataset proportions (~50:1 -> should be closer to uniform here)
    print("\nSampling 20 batches to check class balance from the weighted sampler...")
    tally = Counter()
    it = iter(train_loader)
    for _ in range(20):
        b = next(it)
        for idx in b["flat"].tolist():
            tally[FLAT_LABELS[idx]] += 1
    print("Observed class frequency over 20 sampled batches:")
    for label in FLAT_LABELS:
        print(f"  {label:15s}: {tally.get(label, 0)}")

    # Save a visual grid for manual inspection
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(batch["image"]):
            ax.axis("off")
            continue
        img = denormalize(batch["image"][i])
        label_idx = batch["flat"][i].item()
        ax.imshow(img)
        ax.set_title(FLAT_LABELS[label_idx])
        ax.axis("off")
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "sanity_check_grid.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved visual sanity-check grid to: {out_path}")
    print("Open it and confirm: images look like real CXRs, CLAHE contrast looks reasonable,")
    print("augmentation isn't so aggressive that TB/Pneumonia features get destroyed.")


if __name__ == "__main__":
    main()
