"""
Image preprocessing pipeline implementing FR2 from the Phase 1 report:
  - Resize to 224x224 (EfficientNet input resolution)
  - Grayscale -> RGB via channel replication
  - CLAHE contrast enhancement (local histogram equalization)
  - Pixel normalization

This module is intentionally framework-agnostic (pure numpy/OpenCV) so it can
be reused identically whether called from a PyTorch Dataset, a standalone
visualization script, or a Grad-CAM overlay routine later.
"""
import cv2
import numpy as np

TARGET_SIZE = (224, 224)

# ImageNet stats — standard for any ImageNet-pretrained backbone (EfficientNet
# included), even though our images are grayscale-replicated. This is the
# convention transfer learning literature uses; see report section 6.1.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_grayscale(path: str) -> np.ndarray:
    """Load an image from disk as single-channel grayscale, regardless of
    how it was originally saved (some source images are already RGB)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization — preserves local
    contrast differences (report section 6.1), important for CXRs where
    intensity varies a lot across the image (dense mediastinum vs aerated lung)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def resize(img: np.ndarray, size=TARGET_SIZE) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def grayscale_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replicate the single channel 3x, as specified in FR2 — required
    because all reviewed backbones (EfficientNet et al.) are pretrained on
    RGB ImageNet."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def normalize_for_model(rgb_uint8: np.ndarray) -> np.ndarray:
    """Scale to [0,1], then apply ImageNet mean/std normalization.
    Returns float32 array, still in HWC layout — caller (PyTorch Dataset)
    is responsible for converting to CHW tensor."""
    img = rgb_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def preprocess_image(path: str, use_clahe: bool = True) -> np.ndarray:
    """Full pipeline: load -> (CLAHE) -> resize -> RGB -> normalize.
    Returns a normalized float32 HWC array ready for tensor conversion.
    """
    gray = load_grayscale(path)
    if use_clahe:
        gray = apply_clahe(gray)
    gray = resize(gray)
    rgb = grayscale_to_rgb(gray)
    return normalize_for_model(rgb)


def preprocess_image_for_display(path: str, use_clahe: bool = True) -> np.ndarray:
    """Same pipeline but returns uint8 RGB (no normalization) — useful for
    sanity-checking preprocessing visually or for Grad-CAM overlays later."""
    gray = load_grayscale(path)
    if use_clahe:
        gray = apply_clahe(gray)
    gray = resize(gray)
    return grayscale_to_rgb(gray)
