"""Offline environment inspection and reproducibility helpers; no model loading."""
import importlib.metadata
import random
import sys

PACKAGES = ("ultralytics", "opencv-python", "roboflow", "numpy", "torch", "nbformat", "ipykernel", "matplotlib", "pandas", "jupyterlab", "shapely", "lap", "torchvision", "opencv-python-headless")


def environment_report():
    """Report versions without importing vision libraries or downloading assets."""
    report = {"python": sys.version.split()[0]}
    for package in PACKAGES:
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = "NOT INSTALLED"
    return report


def seed_everything(seed=42):
    """Seed Python, NumPy and PyTorch after dependencies are installed.

    Seeds reduce variability; hardware and nondeterministic kernels can still
    change results. Future training must also pass seed and deterministic flags.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
