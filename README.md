# YOLO Multi-Task Computer Vision

**Four implemented computer-vision workflows** covering custom object detection, multi-object tracking and counting, and observer-point visualization. Built with Python, Ultralytics YOLO26n, OpenCV, Roboflow, ByteTrack, and Google Colab.

> **Portfolio status:** Four experiments were completed in Colab and their outputs/checkpoints were saved to private Google Drive. Clean source notebooks and documented measured results are included here. **A fresh, fully sequential notebook rerun has not been verified**; use the execution notes and preserved source artifacts rather than treating cleared outputs as proof of rerun success.

## The four tasks

| Task | Notebook | Implementation | Outcome |
| --- | --- | --- | --- |
| Fire & Smoke Detection | [01_fire_smoke_detection.ipynb](notebooks/01_fire_smoke_detection.ipynb) | Roboflow v14 label audit, reviewed polygon-to-box corrections, YOLO26n fine-tuning, test evaluation, image/video inference | **mAP@50 0.6302**, mAP@50–95 0.2800 |
| Person Object Counting | [02_object_counting.ipynb](notebooks/02_object_counting.ipynb) | Pretrained YOLO26n + ByteTrack + Ultralytics ObjectCounter, two video demos | 4 crossings (62 frames); 8 crossings (712 frames) |
| VisionEye Mapping | [03_visioneye_mapping.ipynb](notebooks/03_visioneye_mapping.ipynb) | Pretrained YOLO26n + ByteTrack + VisionEye perspective rays | 712 video frames processed |
| Neural Ocean Detection | [04_neural_ocean_detection.ipynb](notebooks/04_neural_ocean_detection.ipynb) | Roboflow Neural Ocean v3, YOLO26n fine-tuning, held-out test evaluation, sample predictions | **mAP@50 0.8287**, mAP@50–95 0.5160 |

## Measured detection results

| Held-out test metric | Fire & Smoke | Neural Ocean |
| --- | ---: | ---: |
| Precision | 0.7086 | 0.8064 |
| Recall | 0.5473 | 0.8044 |
| mAP@50 | 0.6302 | 0.8287 |
| mAP@50–95 | 0.2800 | 0.5160 |
| Test images | 639 | 501 |
| Test annotations | 992 | 966 |
| Classes | 2 | 15 |
| Training epochs | 50 | 50 |

The counting figures are observed **line-crossing events** (IN/OUT), not unique-person counts or accuracy percentages. No manually labeled ground-truth counting accuracy has been established. VisionEye provides an image-plane visualization, **not calibrated world-space distance or 3-D localization**.

## Data, models and reproducibility

- **Fire & Smoke:** [Roboflow version 14](https://universe.roboflow.com/gfo-fc1de/fire-and-smoke-detector-yykya/dataset/14), 5,596 images (3,940 train / 1,017 val / 639 test). Seven problematic polygon-like annotations were reviewed against an independent COCO export and repaired in a **separate cleaned copy**, with the clean dataset audit reporting no issues. Original source was preserved. See [label diagnosis](FIRE_SMOKE_LABEL_DIAGNOSIS.md), [review procedure](POLYGON_RECOVERY_COLAB.md) and [provenance verification](PHASE2C_PROVENANCE_COLAB.md).
- **Neural Ocean:** [Roboflow version 3](https://universe.roboflow.com/neural-ocean/neural_ocean/dataset/3), 5,127 images (3,626 train / 1,000 val / 501 test), 15 underwater object categories. Structural label checks passed, but filename-overlap checks **do not** rule out visually duplicate images.
- **Training:** pretrained `yolo26n.pt`, 50 epochs, image size 640, batch 16, seed 42, NVIDIA T4, Ultralytics 8.4.208. Hardware, package versions and nondeterministic operations can affect reproduction.
- **Tracking:** pretrained `yolo26n.pt` with `bytetrack.yaml`. No custom training for ObjectCounter or VisionEye.
- **Artifacts:** trained `best.pt` / `last.pt`, metrics JSON, plots, dataset archives, input/output videos and sample images are retained in private Google Drive, **not committed**. Do not assume they are publicly downloadable.

### Private artifact locations (Google Drive)

```text
MyDrive/YOLO_Multi_Task/
  fire_smoke_v14/
    polygon_review_20261110T141343824922Z/cleaned_detection_dataset/
    training/fire_smoke_v14_seed42_20261114T125518Z/
  object_counting/person_counting_demo/
  object_counting/street_walk_counting/
  visioneye/street_walk_visioneye/
  neural_ocean/
    datasets/neural_ocean_v3_backup.zip
    training/neural_ocean_v3_seed42_20261114T192943Z/
  final_submission/notebook_originals/
```

The saved Fire and Neural Ocean training run directories contain `weights/best.pt` and `test_metrics.json`. Paths refer to the author's private Drive and must be adapted to reproduce the work.

## Running the notebooks

1. Clone the repository:
   ```bash
   git clone https://github.com/Ibraheim11/yolo-multi-task-computer-vision.git
   cd yolo-multi-task-computer-vision
   ```
2. Use a compatible Python environment. The original local baseline was Python 3.11. In a fresh Google Colab runtime, mount Drive when needed, clone the **complete** repository, and install dependencies before executing code:
   ```python
   %pip install -r requirements.txt
   ```
3. Open the relevant notebook in Colab and inspect its prerequisites, dataset/video paths, API credentials and GPU checks. Put `ROBOFLOW_API_KEY` in Colab Secrets or an environment variable, **never** in source code.
4. Choose the **specific stage** to execute. Training cells are expensive and GPU-dependent; saved-results review does not require rerunning training. Several cells document the original experiment and may rely on paths/state created during previous Colab sessions. A fresh run-through is not currently guaranteed.
5. Run offline notebook validation from repository root:
   ```bash
   python -m pip install nbformat
   python scripts/validate_notebooks.py
   ```
   This verifies notebook schema, cell IDs, cleared outputs and basic Python syntax; it does **not** train models or execute inference.

## Limitations and responsible use

- Fire/Smoke smoke recall is lower than fire recall; this experiment is **not a validated safety-critical fire alarm**.
- Underwater classes have uneven test representation; rare-class scores may be unreliable, and diver images have sometimes produced false positives.
- Counting errors may result from occlusion, tracker identity switches and crossings near the line. Manual ground-truth labeling remains future work.
- VisionEye rays describe pixel-space relationships, not metric depth.
- Demo footage, stock media, raw dataset exports and checkpoint redistribution require separate license/usage review. The repository deliberately excludes heavy media, credentials, datasets and checkpoints.

## Repository layout

```text
notebooks/                 # Four cleaned Colab notebooks (no execution outputs)
src/                       # Environment, Fire/Smoke audit and polygon review utilities
scripts/                   # Offline notebook and data-workflow validators
.github/workflows/         # Notebook static-validation CI
docs/                      # Measured results and portfolio case study
requirements.txt           # Project dependencies (bounded versions)
requirements-lock.txt      # Historical environment snapshot
```

See [detailed results](docs/RESULTS.md), [portfolio case study](docs/PORTFOLIO_CASE_STUDY.md), and existing Phase 2A–2C data-quality reports for methodology and evidence.

## Development policy

All four notebooks are versioned with cleared outputs and without credentials. Keep original experimental outputs and exact private data on Drive. Review source licensing and notebook diffs before sharing artifacts. This code is research/educational work; no production performance, calibrated inference, or external safety guarantees are claimed.
