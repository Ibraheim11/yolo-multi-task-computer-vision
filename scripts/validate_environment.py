"""Offline imports, CPU tensor runtime, and installed API inspection; no model loading."""
import os
import sys
import inspect
import importlib.util
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep package settings inside this project and disable runtime dependency installs.
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "outputs" / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs" / ".matplotlib")
from src.environment import environment_report, seed_everything

for name, value in environment_report().items():
    print(f"{name}: {value}")
try:
    is_colab = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    is_colab = False
if is_colab:
    print("Runtime: Google Colab (managed Python environment)")
else:
    assert sys.prefix != sys.base_prefix, "Use the project virtual environment"
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "Use this project's .venv"
    print("Runtime: local project .venv")
from src.fire_smoke import audit_dataset
assert callable(audit_dataset)
print("PASS: src.fire_smoke import")
print("Checking imports...", flush=True)
import cv2
import torch
import torchvision
import roboflow
import matplotlib
import numpy as np
import pandas as pd
print("PASS: core vision/data imports", flush=True)
import nbformat
import ipykernel
import jupyterlab
print("PASS: notebook imports", flush=True)
import lap
import shapely
from ultralytics import YOLO, solutions
from ultralytics.solutions.solutions import BaseSolution, SolutionResults

seed_everything(42)
x = torch.tensor([[1., 2.], [3., 4.]])
assert torch.equal(x @ x, torch.tensor([[7., 10.], [15., 22.]]))
assert cv2.cvtColor(np.zeros((8, 8, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY).shape == (8, 8)
assert pd.DataFrame({"x": [1, 2]})["x"].sum() == 3
assert torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([0.9]), 0.5).tolist() == [0]
assert callable(YOLO.predict) and callable(YOLO.track)
assert "detect" in inspect.getsource(YOLO.task_map.fget)
print("PASS: all imports, Torch CPU matrix multiplication, torchvision NMS, OpenCV, pandas, and seeds")
print(f"CUDA available: {torch.cuda.is_available()}; Torch CUDA build: {torch.version.cuda}")
if torch.cuda.is_available():
    assert (torch.ones(2, device="cuda") + 1).cpu().tolist() == [2., 2.]
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
for cls in (solutions.ObjectCounter, solutions.VisionEye):
    print(f"{cls.__name__} constructor: {inspect.signature(cls)}")
    print(f"{cls.__name__}.process: {inspect.signature(cls.process)}")
    assert issubclass(cls, BaseSolution)
    assert callable(cls.__call__)
    source = inspect.getsource(cls.process)
    assert "SolutionResults" in source and "plot_im" in source
    print(f"PASS: {cls.__name__} process returns SolutionResults with plot_im (source inspection)")
print("BaseSolution.__call__:")
print(inspect.getsource(BaseSolution.__call__))
from ultralytics.solutions.config import SolutionConfig
for config in (
    dict(model="models/local-checkpoint.pt", region=[(0, 10), (100, 10)], tracker="bytetrack.yaml", show=False),
    dict(model="models/local-checkpoint.pt", vision_point=(20, 20), tracker="bytetrack.yaml", show=False),
):
    cfg = SolutionConfig().update(**config)
    for key, value in config.items():
        assert getattr(cfg, key) == value
print("PASS: actual SolutionConfig accepts planned constructor kwargs without loading a model")
assert "self.model = YOLO" in inspect.getsource(BaseSolution.__init__)
assert "persist=True" in inspect.getsource(BaseSolution.extract_tracks)
print("PASS: constructor loads model; frame extraction uses persistent tracking (source inspection)")
print("SolutionResults signature:", inspect.signature(SolutionResults))
print("PASS: installed detection/tracking methods and solution API inspection; no models constructed")
