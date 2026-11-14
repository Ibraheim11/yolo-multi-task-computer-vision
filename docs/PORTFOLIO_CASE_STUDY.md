# Portfolio Case Study — YOLO Multi-Task Computer Vision

## Goal

Build four hands-on computer vision workflows: two custom-trained object detectors and two video analytics applications. Show how dataset quality, tracking, reproducible training, deployment constraints and evaluation affect practical system outcomes.

## Technology

Python 3.11 baseline · Ultralytics 8.4.208 · YOLO26n · PyTorch · OpenCV · Roboflow · ByteTrack · Google Colab · Google Drive · GitHub.

## Engineering contributions

**Dataset quality and provenance:** Diagnosed ambiguous Fire/Smoke label formats, prepared visual overlays, reconciled the suspect annotations with a separate COCO export, approved seven corrections and produced an audited, separately copied YOLO detection dataset without overwriting the raw source.

**Custom detection:** Fine-tuned YOLO26n for 50 epochs on two datasets and measured precision/recall/mAP on reserved test splits. Preserved checkpoints, dataset provenance and JSON metrics.

**Video analytics:** Built annotated counting pipelines with persistent tracking and line-crossing reports, then a VisionEye experiment mapping tracked people visually to a specified observation point.

**Operational evidence:** Stored artifacts outside short-lived Colab runtimes; maintained progress checkpoints and reduced published notebooks to clean source-only versions with static validation and CI.

## Outcomes

| Workflow | Verified technical result |
| --- | --- |
| Fire/Smoke custom detector | Test mAP@50 **63.02%**, mAP@50–95 **28.00%** |
| Person counting | Two video executions producing 4 and 8 crossing events |
| VisionEye mapping | 712 annotated video frames |
| Neural Ocean detector | Test mAP@50 **82.87%**, mAP@50–95 **51.60%** |

## Trade-offs and limitations

The Fire/Smoke model's smoke recall is insufficient for safety-critical alerting. Counting output remains unverified against manually labeled crossing events. VisionEye rays are not 3D measurements. Rare Neural Ocean classes have sparse evaluation samples and may produce false positives. Re-running Colab experiments may require updating private artifact paths and acquiring suitable hardware, datasets and media permissions.

## Next engineering steps

Add a reproducible, parameterized inference entry point independent of Colab paths; package a small licensed, non-sensitive validation sample; establish manually labeled counting ground truth; measure per-class detector performance and failure modes; and add separate smoke recall threshold and robustness investigations before considering applications beyond demos.
