"""
Run this FIRST after setting up your environment to confirm PyTorch can see
your RTX 4050 GPU. This corresponds to NFR3 (training efficiency) in your
Phase 1 report.

Usage:
    python scripts/verify_environment.py
"""
import sys


def check_torch():
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed. Install it first — see requirements.txt header.")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("[WARN] CUDA not detected. Training will fall back to CPU (very slow for EfficientNet).")
        print("       Check: (1) NVIDIA driver installed, (2) you installed the CUDA build of torch,")
        print("       not the CPU-only wheel. Reinstall with the cu121/cu124 index URL from pytorch.org.")
        return

    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {device_name}")
    print(f"VRAM: {total_mem_gb:.1f} GB")

    # Quick sanity matmul on GPU
    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print(f"[OK] Test matmul succeeded on GPU. Result shape: {tuple(z.shape)}")

    if total_mem_gb < 5:
        print("[NOTE] With ~6GB VRAM, use batch_size=16-24 at 224x224 with mixed precision (AMP).")
    else:
        print("[NOTE] VRAM looks sufficient for batch_size=32 at 224x224 with AMP.")


def check_other_libs():
    libs = ["numpy", "pandas", "cv2", "sklearn", "matplotlib", "pytorch_grad_cam", "shap", "lime"]
    for lib in libs:
        try:
            __import__(lib)
            print(f"[OK] {lib}")
        except ImportError:
            print(f"[MISSING] {lib} — run: pip install -r requirements.txt")


if __name__ == "__main__":
    print("=" * 60)
    print("Environment check")
    print("=" * 60)
    check_torch()
    print("-" * 60)
    check_other_libs()
