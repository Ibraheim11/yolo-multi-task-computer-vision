# Phase 2B: Fire and Smoke Label Diagnosis

## Evidence and current limits

Saved Drive audit verified on 2026-11-10; see [source links and affected paths](PHASE2B_DRIVE_EVIDENCE.md). These results were read from Drive, not reproduced locally:

| Split | Images | Parsed boxes | Invalid annotation files |
| --- | ---: | ---: | ---: |
| train | 3940 | 6227 | 1 |
| validation | 1017 | 1653 | 1 |
| test | 639 | 987 | 4 |

Total images: **5596**. Actual class names reported: **fire, smoke**. Saved audit issues: **7**; dataset ready: **False**. No missing annotations or corrupt images were reported. These saved Drive results were verified through the connector, not reproduced locally. Parsed boxes can include valid rows in otherwise invalid files; these counts do not prove every annotation is valid.

The export is absent locally and from the current Colab session. The Drive report confirms seven field-count errors across six files; `fire32frame52...txt` fails on lines 1 and 2. Rejected raw rows are not saved in the report, so token counts and formats require the restored export. Do not infer segmentation from field counts.

## Why a longer row can be meaningful

`src/fire_smoke.py:parse_label` enforces the intended detection task and remains strict. Its `split()` parsing already accepts repeated spaces, tabs, and ordinary Unicode whitespace; UTF-8 BOM and CRLF are supported. Empty labels represent backgrounds, not invalid annotations.

- Detection: `class x_center y_center width height` (5 fields).
- [Segmentation](https://docs.ultralytics.com/datasets/segment/): class followed by at least three normalized x/y pairs (7 or more fields).
- [Oriented boxes](https://docs.ultralytics.com/datasets/obb/): class followed by four normalized corners (9 fields). A nine-field row can also resemble a four-point polygon.
- [Pose](https://docs.ultralytics.com/datasets/pose/): detection box followed by keypoints, with optional visibility; the YAML `kpt_shape` is relevant.

The new classifier reports **structural candidates**, not confirmed task types or valid polygons. Shape, normalized values, distinct points and nonzero area are useful clues but do not prove semantic correctness, a simple non-self-intersecting polygon, or correct image alignment. Extra values could also be malformed detection data, confidence scores, concatenated objects, or truncation. Never truncate every row to its first five values: polygon vertices are not center/width/height values.

## Run against the real export in Colab

Start with [the exact Colab restart instructions](PHASE2B_COLAB.md), including preserving the original audit before running notebook audit cells. The examples below also support an already restored runtime.

Update/upload the new `src/label_diagnostics.py` and diagnostic script into the full project checkout. No new dependency or model download is needed. From a separate code cell after setup (using the existing notebook variables):

```python
from src.label_diagnostics import diagnose_labels, write_diagnosis

saved_report = AUDIT_DIR / "report.json"
diagnosis = diagnose_labels(
    dataset_dir=DATASET_DIR,
    audit_report=saved_report if saved_report.is_file() else None,
)
print(write_diagnosis(diagnosis, OUTPUT_ROOT / "label_diagnostics"))
```

This rescans actual labels and performs the existing full dataset audit. It reads images for integrity/hashing but runs no model, training or inference. Outputs go outside the original export, preferably to the existing Drive output path. The complete report is not limited to the first 50 entries.

Equivalent CLI (substitute the real output path, which may be on Drive):

```python
%run scripts/diagnose_fire_smoke_labels.py --dataset /content/yolo-multi-task-computer-vision/datasets/fire_smoke_v14 --report "/content/drive/MyDrive/YOLO_Multi_Task/fire_smoke_v14/dataset_audit/report.json" --output "/content/drive/MyDrive/YOLO_Multi_Task/fire_smoke_v14/label_diagnostics"
```

If only the saved report is available:

```python
%run scripts/diagnose_fire_smoke_labels.py --report "/actual/path/dataset_audit/report.json" --output "/actual/path/label_diagnostics"
```

Report-only mode reports every original issue and its saved line number, but cannot reconstruct raw rows or confirm annotation formats. It does not follow stale absolute file paths. Use `--dataset` for real row diagnosis. The supplied original report and the fresh audit have separate counts so discrepancies remain visible.

Generated reports:

- `diagnosis.json`: complete original/current issue lists, invalid annotation files, exact raw rows, line numbers, field counts, error reasons, candidate interpretations, file hashes, and count definitions.
- `invalid_lines.csv`: one row per diagnosed finding; report-only mode may include non-label findings and explicitly unavailable raw content.
- `diagnosis.txt`: full human-readable per-file/per-line findings plus all original/current audit issues, including non-label issues.

Do not treat `diagnostic_finding_count`, `invalid_annotation_file_count`, and `current_audit_issue_count` as interchangeable. Dataset scanning covers all split label files, not just saved report entries. The CSV/JSON may contain project paths and label data; retain them under ignored outputs, not Git.

## Smallest safe correction strategy

1. Read the complete diagnosis and account for all seven original issues. Open each corresponding original image/label in Roboflow or an annotation viewer. Confirm the intended source annotation format and object boundaries. Preserve every original file.
2. Prefer a corrected detection re-export from Roboflow into a **new directory** if the exporter mixed formats. Record source version and export settings; do not overwrite the version-14 export.
3. For a confirmed polygon/OBB instance that should become an axis-aligned detection box, the geometric proposal is `xmin=min(x_i)`, `xmax=max(x_i)`, `ymin=min(y_i)`, `ymax=max(y_i)`, then `xc=(xmin+xmax)/2`, `yc=(ymin+ymax)/2`, `w=xmax-xmin`, `h=ymax-ymin`. This must be reviewed on the image; it loses mask/orientation detail. The tool deliberately does not infer or apply this conversion.
4. A truncated/missing coordinate, uncertain class, concatenated row or ambiguous shape requires source annotation review or re-export. Do not guess, clip boxes, delete an object row, remove its image, or relax the detector's five-field validation.
5. If a human can specify a justified replacement, create the explicit plan below. Only diagnosed invalid lines can be changed. Each replacement is one valid detection row, so no image or nonblank annotation row is discarded. A duplicate object row that should genuinely be removed needs a separate reviewed policy; this tool does not delete it.

### Explicit reviewed-copy plan

Create this JSON outside the original export, replacing all placeholder values with real diagnostic evidence. This is a schema example, **not an actual repair plan**:

```json
{
  "files": [
    {
      "path": "train/labels/ACTUAL_FILENAME.txt",
      "original_sha256": "COPY_THE_REAL_SHA256_FROM_DIAGNOSIS",
      "replacements": [
        {
          "line": 1,
          "original": "COPY_THE_EXACT_ORIGINAL_LINE",
          "replacement": "CLASS_ID XC YC WIDTH HEIGHT_FROM_CONFIRMED_REVIEW",
          "reason": "Describe the image/source review, confirmed format, and chosen correction"
        }
      ]
    }
  ]
}
```

`path` is relative to the original export; `line` is one-based. Include every necessary replacement in a planned file, because the preflight rejects any planned file that remains invalid. Unknown formats are never automatically repaired. The reviewer, not the classifier, supplies corrected coordinates.

Only after reviewing that plan, optionally create a separate copy:

```python
%run scripts/diagnose_fire_smoke_labels.py --dataset /content/yolo-multi-task-computer-vision/datasets/fire_smoke_v14 --output "/actual/output/label_diagnostics" --reviewed-plan "/actual/output/reviewed_plan.json" --clean-copy /content/yolo-multi-task-computer-vision/datasets/fire_smoke_v14_reviewed
```

The destination must not exist and must be outside the original export. No automatic overwrite, in-place repair, deletion, model construction or training occurs. The copy preserves every image and untouched label byte-for-byte, preserves untouched lines and newline/BOM conventions in edited labels, records original/copy hashes, rebases the copied YAML split paths, and writes `repair_manifest.json` and `repair_audit.json`. A valid annotation row cannot be edited by this tool. File hashes prevent applying stale plans to changed originals.

A copy may still fail its audit due to unrelated issues, including leakage. **Copy creation is not training approval.** Inspect both reports and re-audit the copy. The original remains unchanged.

## Training remains blocked

The existing notebook already requires a clean `DATASET_REPORT`, a generated training YAML, explicit training opt-in, and an unchanged fresh audit before model construction. This phase does not weaken or bypass those gates. `write_training_yaml` still rejects failed audits.

If a reviewed copy passes and you later choose to use it, set `DATASET_DIR` explicitly to that copy and rerun the original notebook audit and visual inspection. Do not simply change `ready` to True or reuse the old original-export training YAML. Keep `RUN_TRAINING=False` for Phase 2B.

## Local validation

```powershell
.\.venv\Scripts\python.exe scripts/validate_label_diagnostics.py
.\.venv\Scripts\python.exe scripts/validate_fire_smoke.py
.\.venv\Scripts\python.exe scripts/validate_notebooks.py
```

All diagnosis/repair tests use labeled synthetic fixtures solely for regression tests and remove temporary files afterward. They are not dataset statistics, model predictions, or task results. A negative training-gate test verifies rejection before any model call; no training library is imported for that test.
