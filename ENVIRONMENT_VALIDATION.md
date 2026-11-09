# Phase 1.5 Environment Validation

Validated locally on 2026-11-09. Result: **PASS**, with the limitations below.

The existing project `.venv` was reused with uv 0.13.7. Global Python packages and unrelated projects were not modified. The actual existing project path is `D:\AI&MlL\yolo-multi-task-computer-vision`.

## Installed versions

| Component | Version |
| --- | --- |
| python | 3.11.18 |
| ultralytics | 8.4.208 |
| opencv-python | 4.14.0.94 |
| roboflow | 1.7.3 |
| numpy | 2.4.6 |
| torch | 2.15.0 |
| nbformat | 5.11.1 |
| ipykernel | 7.4.0 |
| matplotlib | 3.11.2 |
| pandas | 2.3.3 |
| jupyterlab | 4.6.6 |
| shapely | 2.2.0 |
| lap | 0.5.14 |
| torchvision | 0.29.2 |
| opencv-python-headless | 4.14.0.94 |

All 132 resolved packages are recorded in `requirements-lock.txt`. This is a snapshot of the tested Windows/Python 3.11 environment; Colab must resolve and validate its own Linux environment.

## Checks

- PASS: uv dependency consistency check (132 packages, no metadata conflicts).
- PASS: imports for Ultralytics, Torch, torchvision, cv2, Roboflow, Matplotlib, NumPy, pandas, nbformat, ipykernel, JupyterLab, LAP, and Shapely.
- PASS: CPU tensor matrix multiplication, torchvision NMS, OpenCV color conversion, pandas dataframe operation, and fixed-seed helper execution.
- CUDA: unavailable; `torch.version.cuda` is None. Torch runtime is 2.15.0+cpu. Local CPU execution is acceptable; no GPU performance is claimed.
- PASS: all four notebooks parse as JSON and pass official nbformat schema validation. Their code cells pass Python syntax checks, have no stored outputs, and retain null execution counts.
- PASS: each notebook's existing setup cells execute from both repository root and notebooks directory. No notebook file outputs were saved.
- PASS: syntax validation for every module in src/ and scripts/.
- PASS: Git check-ignore verified environment, credentials, datasets, weights, runs, generated media and outputs are ignored. README, requirements, notebook sources and directory placeholders remain visible.

## Installed Ultralytics API

Ultralytics 8.4.208 provides YOLO.predict, YOLO.track and the detection task mapping. Its bundled model configuration/source supports the documented YOLO26 candidate; no model checkpoint was constructed or tested.

Both `solutions.ObjectCounter` and `solutions.VisionEye` use `__init__(**kwargs)`. Actual SolutionConfig updates accepted model, tracker, show, and task-specific region/vision_point settings without constructing a solution. Planned settings:

- ObjectCounter: model=local checkpoint path, region=[(x1,y1),(x2,y2)] (or polygon), tracker="bytetrack.yaml", show=False.
- VisionEye: model=local checkpoint path, vision_point=(x,y), tracker="bytetrack.yaml", show=False.

Both expose `process(self, im0) -> SolutionResults`. The callable wrapper `solution(frame)` delegates to process and adds timing information. Installed source confirms results.plot_im, persistent model.track(..., persist=True), ObjectCounter IN/OUT fields, and VisionEye total_tracks. Constructors load a model and can download absent weights, so constructor execution and real frame inference were intentionally excluded. These are import/configuration/source compatibility checks, not an end-to-end vision test.

## Compatibility issues and fixes

- The first resolution selected opencv-python 4.14 and opencv-python-headless 5.0, which share cv2. Both are now pinned to 4.14.0.94 and the image runtime check passes. The shared namespace remains an upstream packaging caveat; avoid independently uninstalling one distribution. Real video codec checks await supplied videos.
- pandas 3.1.1 stalled during compiled-extension import locally, including outside the sandbox. pandas is constrained to >=2.2,<3; resolved pandas 2.3.3 imports and passes a dataframe runtime check. Stalled validator processes were terminated; stale 3.1.1 metadata was removed after a DLL-lock interrupted replacement. The final dependency check passes.
- Matplotlib's default user cache was unwritable under the sandbox. The validator sets MPLCONFIGDIR and YOLO_CONFIG_DIR under ignored outputs/ and disables Ultralytics automatic dependency installation.
- uv used file copying because cache/project drives do not support cross-drive hard links. Installation completed successfully.

## Requirements changes

Added pandas>=2.2,<3, shapely>=2.0,<3 and lap>=0.5.12,<0.6. Pinned opencv-python and added opencv-python-headless at the aligned 4.14.0.94 release. Other baseline ranges were retained. Added an exact requirements-lock.txt snapshot.

## Scope boundary

No Roboflow datasets or pretrained weights were downloaded; no training, predictions, evaluation metrics, or subtask implementation was performed. Google Colab GPU behavior and real video inference remain untested. Subtask 1 requires a separate approval.
