# Experiment results and evidence

Measurements below were generated during the completed Google Colab experiments and saved with the original run artifacts on the author's Google Drive. The GitHub notebook source has had execution outputs cleared to avoid embedding media and potentially sensitive output.

## Task 1 — Fire & Smoke detection

- Dataset: Roboflow Fire & Smoke v14; 5,596 images; 3,940 train / 1,017 validation / 639 test; 2 detection classes.
- Original anomaly: seven polygon-like label records across six files. All were visually reviewed and checked against an independent COCO export, then corrected into a separate detection dataset with an audit reporting zero outstanding issues.
- Model: YOLO26n, pretrained `yolo26n.pt`, 50 epochs, image size 640, batch 16, seed 42, T4 GPU.
- Held-out test: **precision 0.708612**, **recall 0.547283**, **mAP@50 0.630184**, **mAP@50–95 0.280042**; 639 test images, 992 annotations.
- Video inference: 384 frames processed; 314 frames containing detections. These are detection events, not verified alarm performance.

**Caveat:** Smoke recall is relatively weak; this work is unsuitable as a standalone life-safety alarm.

## Task 2 — Person counting

| Video | Resolution | Frames | IN | OUT | Crossings |
|---|---|---:|---:|---:|---:|
| Short sample | 640×360 | 62 | 2 | 2 | 4 |
| Street Walk | 1920×1080 | 712 | 2 | 6 | 8 |

- Pretrained YOLO26n, Ultralytics ObjectCounter and ByteTrack.
- Short sample crossing region: horizontal line y=180; street sample: vertical line x=960.
- Annotated output videos and JSON reports saved privately.
- **No manually labeled counting ground truth.** Crossings must not be described as confirmed counting accuracy.

## Task 3 — VisionEye mapping

- Pretrained YOLO26n + ByteTrack + Ultralytics VisionEye.
- Street Walk video: 712 frames at 1920×1080.
- Fixed observation point: (960, 1000).
- Annotated video with tracking IDs and observer rays was generated and visually inspected.
- The rendered rays describe image coordinates only, not calibrated depth, perspective reconstruction or metric distance.

## Task 4 — Neural Ocean detection

- Dataset: [Neural Ocean v3](https://universe.roboflow.com/neural-ocean/neural_ocean/dataset/3): 5,127 images (3,626 train / 1,000 validation / 501 test), 15 classes.
- Initial structural label audit: zero reported invalid rows; no overlapping filenames across splits. Filename checks do not prove freedom from visual duplicates.
- Model: YOLO26n, pretrained `yolo26n.pt`, 50 epochs, 640-pixel images, batch 16, seed 42, T4 GPU.
- Held-out test: **precision 0.806394**, **recall 0.804353**, **mAP@50 0.828681**, **mAP@50–95 0.515967**; 501 test images, 966 annotations.
- Predictions saved on six sample test images.
- Possible false positives on divers; rare classes have very few test instances.

## Reproduction and evidence limits

- Checkpoints, JSON metrics, dataset archives, image/video predictions and experiment reports were saved to private `MyDrive/YOLO_Multi_Task`.
- The GitHub repository contains documentation and code rather than confidential Drive assets, model weights or licensed media.
- Full clean-room execution on a new Colab runtime was not performed as part of repository packaging.
- Python syntax, JSON/schema and notebook-output hygiene are verified by static checks; those checks do not measure inference quality.
