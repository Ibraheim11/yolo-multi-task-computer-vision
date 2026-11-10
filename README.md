# YOLO Multi-Task Computer Vision Assignment

Four independent Jupyter notebooks for custom detection, video counting, and object mapping. **Phase 2A: Fire and Smoke dataset preparation and notebook implementation.** No datasets, model weights, training runs, predictions, or evaluation results are included.

## Assignment notebooks

| Notebook | Scope |
| --- | --- |
| `notebooks/01_fire_smoke_detection.ipynb` | Fire/smoke detection; Roboflow version 14 |
| `notebooks/02_object_counting.ipynb` | Pretrained YOLO, tracking, OpenCV, ObjectCounter |
| `notebooks/03_visioneye_mapping.ipynb` | VisionEye tracked object mapping and video export |
| `notebooks/04_neural_ocean_detection.ipynb` | Neural Ocean custom detection; Roboflow version 3 |

The Fire and Smoke notebook contains executable, guarded dataset preparation, training, validation, and video workflows. The other three notebooks remain Phase 1 scaffolds. No training or inference has been executed.

## Local setup (PowerShell, Python 3.11)

Run from this repository's root. Preserve the existing uv-managed `.venv`:

```powershell
.\.venv\Scripts\python.exe --version
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/validate_environment.py
.\.venv\Scripts\python.exe scripts/validate_notebooks.py
.\.venv\Scripts\python.exe -m ipykernel install --user --name yolo-multitask --display-name "YOLO Multi-Task (Python 3.11)"
.\.venv\Scripts\python.exe -m jupyterlab
```

If uv created an environment without pip, use `uv pip check --python .\.venv\Scripts\python.exe` instead of `python -m pip check`. Select the registered kernel in Jupyter. Installation is an explicit user action; setup scripts do not install packages or load models. Torch GPU availability depends on your driver and wheel; consult the official PyTorch installation selector when choosing a CUDA build. CPU is sufficient for setup checks.

## Google Colab: clone and import the project

These instructions apply **after** the repository has been created and pushed to GitHub. No GitHub repository or remote URL has been configured yet. Replace `YOUR_GITHUB_USERNAME` with the actual owner and adjust the repository name if needed. A public repository can be cloned without credentials. For a private repository, use a private authenticated workflow or upload a ZIP of the source; never put a token in a URL, notebook cell, or output.

1. Open a new notebook at [Google Colab](https://colab.research.google.com/) and select a GPU runtime for later training. In its first code cell:

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/yolo-multi-task-computer-vision.git /content/yolo-multi-task-computer-vision
%cd /content/yolo-multi-task-computer-vision
```

Clone once per fresh runtime. If that directory already exists, inspect/reuse it rather than rerunning clone or deleting it automatically. For reproducibility, record `!git rev-parse HEAD` and use the same commit for subsequent experiments.

2. Install the project dependencies in Colab. This installs packages, not datasets or model weights:

```python
%pip install -r requirements.txt ultralytics==8.4.208
```

Restart the runtime if requested, then rerun `%cd /content/yolo-multi-task-computer-vision`. Use `requirements.txt` on Colab; `requirements-lock.txt` is the tested **Windows** snapshot and includes Windows-specific packages. Colab controls Python, Torch, CUDA, and GPU versions, so validate that environment independently.

3. Verify the source import from the repository root, then run offline checks:

```python
from pathlib import Path
import src.fire_smoke as fire_smoke
assert Path(fire_smoke.__file__).resolve() == Path("src/fire_smoke.py").resolve()
print("Source import OK:", fire_smoke.__file__)
%run scripts/validate_environment.py
%run scripts/validate_notebooks.py
```

The environment validator accepts Colab's managed Python environment and checks required imports/APIs without constructing models. No editable package installation is required: the repository root must be the working directory (or on `sys.path`).

4. Open `notebooks/01_fire_smoke_detection.ipynb` using **File ? Open notebook ? GitHub**, or upload the notebook separately. Opening a notebook from GitHub does **not** clone `src/` or other files; run the clone/installation steps in that runtime too. The notebook recognizes the checkout at `/content/yolo-multi-task-computer-vision` even if the current directory is `/content`.

5. Run its setup cells and authorize the Colab-only Google Drive mount for persistent outputs. Store `ROBOFLOW_API_KEY` in Colab Secrets, an environment variable, or the hidden runtime prompt. Credentials and data are intentionally absent from Git. Dataset download is a later explicit action; keep `DOWNLOAD_DATASET=False` for readiness checks. Keep `RUN_TRAINING`, `RUN_VALIDATION`, and `RUN_VIDEO_INFERENCE` False until their execution is authorized.

Open each assignment notebook independently. Keep `show=False` for video solutions and use inline playback instead of GUI windows. Colab's `/content` checkout and datasets are ephemeral; checkpoints and outputs must go to the notebook's Drive output directory or be copied out before runtime shutdown. Drive is not required for local Windows execution.

## API verification and model policy

Official documentation reviewed on **2026-11-09**:

- [ObjectCounter](https://docs.ultralytics.com/guides/object-counting/): `solutions.ObjectCounter(model=..., region=..., tracker="bytetrack.yaml", show=False)`; a line uses two points and a polygon uses at least three.
- [VisionEye](https://docs.ultralytics.com/guides/vision-eye/): `solutions.VisionEye(model=..., vision_point=(x, y), tracker="bytetrack.yaml", show=False)`.
- [Models](https://docs.ultralytics.com/models/): current guides recommend YOLO26. `yolo26n.pt` is a candidate for later pretrained tasks, not a loaded or tested checkpoint here.
- [Solution reference](https://docs.ultralytics.com/reference/solutions/solutions/): frame processing returns a solution result; verify `plot_im` against the installed release before implementing video loops.

Do not use legacy constructor or counting examples without checking the installed version. The validator checks class availability without instantiation. Phase 2A verifies the `yolo26n.pt` detection asset/config against installed Ultralytics 8.4.208 and configures it for later training. No pretrained checkpoint or real frame inference has been executed.

## Reproducibility and dependencies

`requirements.txt` provides bounded dependency ranges for Python 3.11, not an exact lock. After installation and successful validation, capture the tested resolution with:

```powershell
uv pip freeze --python .\.venv\Scripts\python.exe > requirements-lock.txt
```

Review and commit the lock file only after validation; use it to recreate that environment. Record GPU/driver details separately. Shared utilities seed Python, NumPy, and Torch with 42; future training must explicitly set seed and deterministic options. Record hyperparameters, dataset export/version, splits, class map, model checkpoint provenance, and video identity. Hardware and nondeterministic operations may still affect results.

## Files and artifact policy

- `src/environment.py`: reusable offline version report and seed helper.
- `scripts/validate_environment.py`: dependency, import, solution-class, and CUDA availability checks; no model construction.
- `scripts/validate_notebooks.py`: notebook JSON, schema (when nbformat is installed), Python syntax, and clean-output checks.
- `assets/images/`, `assets/videos/`: local input assets.
- `datasets/`: local dataset exports; verify licenses and original split provenance.
- `models/`: local checkpoints and weights.
- `outputs/<task>/`: generated videos, metrics, plots, and run records.
- `.gitignore`: excludes data, media, weights, outputs, secrets, caches, and virtual environments while retaining folder placeholders.

Keep all local artifacts in their designated directories. Never hard-code API keys or commit notebook outputs containing secrets. Use `ROBOFLOW_API_KEY` from an environment variable or Colab Secrets. The Fire/Smoke workspace/project identifiers are configured in its notebook; Neural Ocean identifiers and licensed source videos must be supplied during later task implementation; version numbers alone do not identify datasets. Check repository diffs before publishing. Review Ultralytics and dataset licensing before distribution.

## Phase boundary

Phase 2A implements the Fire and Smoke workflow but leaves training, validation, and inference disabled. Download and audit the actual Roboflow export in Colab before a separately authorized 50-epoch training run.

## Phase 1.5 validation

Use `uv pip check --python .\.venv\Scripts\python.exe` to verify installed dependency metadata. The environment validator also checks all requested imports, Torch CPU arithmetic, torchvision NMS, OpenCV image conversion, and installed solution signatures/source without constructing a YOLO model. Source/API checks are not end-to-end detection or tracking tests. No pretrained weights are downloaded.

Ultralytics and Roboflow request different OpenCV wheel distributions that share `cv2`. Both are recorded in the lock; avoid independently uninstalling one without reinstalling the other. Basic OpenCV operation is checked, but video codec support must be tested with real input in a later subtask.

See [ENVIRONMENT_VALIDATION.md](ENVIRONMENT_VALIDATION.md) for the Phase 1.5 results and resolved compatibility issues. The exact tested Windows dependency snapshot is in requirements-lock.txt; pandas is restricted to 2.x after a local 3.1.1 import stall.

## Phase 2A: Fire and Smoke notebook

Open `notebooks/01_fire_smoke_detection.ipynb` in Colab with the full repository available under `/content/yolo-multi-task-computer-vision`. Select a GPU runtime, optionally enable the installation cell, then run imports and mount Drive for persistent outputs. Enable only `DOWNLOAD_DATASET` for preparation; obtain `ROBOFLOW_API_KEY` through Colab Secrets, an environment variable, or the hidden prompt. The download targets workspace `gfo-fc1de`, project `fire-and-smoke-detector-yykya`, version 14, YOLOv8 detection export. Audit outputs and representative annotations are produced only from real downloaded data.

`src/fire_smoke.py` audits split paths, actual class names, images, YOLO labels, pairing, and exact cross-split duplicates, and writes a sanitized absolute-path training YAML after a clean audit. No source samples are silently changed. Resolve all audit issues before training. Missing labels are reviewed rather than silently converted to background.

The installed model asset/config confirms `yolo26n.pt` support in Ultralytics 8.4.208. All model loading is inside disabled action branches. Training is configured for 50 epochs at image size 640; no weights were downloaded or training started during Phase 2A. Keep `RUN_TRAINING`, `RUN_VALIDATION`, and `RUN_VIDEO_INFERENCE` False for now. Notebook sections explain checkpoints, metrics, saved video, and limitations.

Offline validation: `.\.venv\Scripts\python.exe scripts/validate_notebooks.py` and `.\.venv\Scripts\python.exe scripts/validate_fire_smoke.py`. The latter uses clearly synthetic temporary fixtures for audit edge cases, checks secure error redaction, and executes the default notebook with network and model actions forbidden. It does not download datasets, load pretrained models, or train. PyYAML and Pillow are now explicit requirements; their existing installed versions already satisfy the constraints.

## Phase 2B: investigate invalid Fire/Smoke labels

For the seven polygon-like rows now diagnosed in Colab, follow [the visual review and polygon recovery workflow](POLYGON_RECOVERY_COLAB.md). It creates image/polygon/box overlays and pending decisions on Drive, requires explicit source-format and visible-object confirmation for every row, and generates the existing reviewed-plan schema only after review. The nine-token row remains format-ambiguous until confirmed. Cleaned data is created separately and fully audited; training and inference remain disabled.

The saved Drive audit was read on 2026-11-10: seven field-count errors across six invalid annotation files. The test file `fire32frame52...txt` has two invalid lines. See [verified evidence and exact filenames](PHASE2B_DRIVE_EVIDENCE.md), [exact Colab restart instructions](PHASE2B_COLAB.md), and [diagnosis/review policy](FIRE_SMOKE_LABEL_DIAGNOSIS.md). Raw rejected rows still require restoring the original export in Colab.

`scripts/diagnose_fire_smoke_labels.py` writes complete JSON/CSV/text diagnostics from the actual export and/or saved report, prints raw rows and token counts, and reconciles every saved issue against the restored export without basename guessing. It never repairs by default. Optional repairs require exact original hashes and human-supplied line replacements in a new dataset copy, followed automatically by a full audit. No images or object rows are silently discarded, and audit gates remain in force before training. Run `scripts/validate_label_diagnostics.py` for offline regression checks. No real dataset diagnosis or repair was executed locally; the Drive report was read through the connector.
## Phase 2C provenance verification

See [the read-only COCO verification procedure and exact Colab cells](PHASE2C_PROVENANCE_COLAB.md). This compares independently obtained v14 source exports without changing annotations or pending approvals. Training remains disabled.
