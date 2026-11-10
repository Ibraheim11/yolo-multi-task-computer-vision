# Phase 2B: exact Colab restart procedure

No real dataset download, correction or training was performed locally. Use a CPU runtime for diagnosis. Upload the updated complete project to `/content/yolo-multi-task-computer-vision` and run notebook setup, installation, imports and Drive/output-path cells. Keep `RUN_TRAINING=False` and inference switches disabled. Do not run all cells.

## 1. Preserve and read the saved audit before downloading

Do this **before the notebook audit cell**, which overwrites `dataset_audit/report.json`. The original Drive report and CSVs must remain intact. Reuse the snapshot on subsequent runs.

```python
from pathlib import Path
import shutil
from google.colab import drive
drive.mount('/content/drive')
%cd /content/yolo-multi-task-computer-vision
PROJECT_ROOT = Path.cwd()
BASE = Path('/content/drive/MyDrive/YOLO_Multi_Task/fire_smoke_v14')
SAVED = BASE / 'dataset_audit_original_phase2b'
assert (BASE / 'dataset_audit/report.json').is_file(), 'Locate the original Drive report first'
if not SAVED.exists():
    shutil.copytree(BASE / 'dataset_audit', SAVED)
assert (SAVED / 'report.json').is_file()
from src.label_diagnostics import diagnose_labels, write_diagnosis
print(write_diagnosis(diagnose_labels(audit_report=SAVED / 'report.json'),
                      BASE / 'phase2b_report_only'))
RUN_TRAINING = False
```

The expected saved report has seven issues in six files. See [verified Drive evidence](PHASE2B_DRIVE_EVIDENCE.md). If your Drive folder differs, change `BASE` to the folder containing the verified `dataset_audit` directory.

## 2. Restore only the original version-14 export in Colab

In the existing notebook **Secure Roboflow version-14 download** cell, set `DOWNLOAD_DATASET=True` and execute that cell only. Use its existing Colab secret/hidden prompt. It selects workspace `gfo-fc1de`, project `fire-and-smoke-detector-yykya`, version `14`, export `yolov8`. Do not create the export target directory in advance, overwrite an existing export, or print credentials. Reset `DOWNLOAD_DATASET=False` afterward. Preserve partial downloads for inspection if the helper refuses to continue. Do not run training.

## 3. Reconcile all seven issues and audit the restored export

```python
from datetime import datetime, timezone
DATASET_DIR = PROJECT_ROOT / 'datasets/fire_smoke_v14'
assert DATASET_DIR.is_dir(), 'Restore the exact original export in Colab first'
DIAG_DIR = BASE / ('phase2b_diagnosis_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
diagnosis = diagnose_labels(DATASET_DIR, SAVED / 'report.json')
print(write_diagnosis(diagnosis, DIAG_DIR))
rows = diagnosis['saved_issue_reconciliation']
assert len(rows) == 7, 'Unexpected saved report; review it first'
assert len({r['saved_issue']['path'] for r in rows}) == 6
print('Changed or unresolved:', [r for r in rows if r['status'] != 'still_invalid'])
RUN_TRAINING = False
```

This runs a full image/label audit and prints every invalid raw line, token count and exact rejection reason. Outputs include `diagnosis.txt`, `diagnosis.json` and `invalid_lines.csv`. JSON includes structural polygon/OBB/pose candidates. Saved paths are rebased relative to the saved `data.yaml`; no basename guessing or reads through stale absolute paths occur. A missing file, missing line, or row that no longer fails is explicitly marked for review. Investigate changed exports before correction.

## 4. Review recovery and create a separate copy only after approval

**For the confirmed diagnostic findings of seven polygon-like rows, use [the dedicated visual-review procedure](POLYGON_RECOVERY_COLAB.md) for this step.** It saves overlays, explicit decisions, the reviewed plan, manifest, and full clean audit to Drive. The generic plan/copy example below remains supported but does not itself perform image review.

Open each affected original image and label in an annotation viewer. A candidate polygon is potentially recoverable only after confirming source format, class, object identity and coordinates. A confirmed polygon's enclosing box is `(min(x)+max(x))/2, (min(y)+max(y))/2, max(x)-min(x), max(y)-min(y)`. This loses mask/orientation detail. A nine-token row can also be an OBB. Degenerate, self-intersecting, truncated, concatenated, nonfinite or unknown-format rows require source review or re-export; never truncate or clip them automatically.

Follow the [explicit reviewed-plan schema](FIRE_SMOKE_LABEL_DIAGNOSIS.md#explicit-reviewed-copy-plan). The reviewer supplies one justified strict detection replacement per invalid row and the diagnosed original file hash. Do not fabricate a plan while raw rows are unavailable. Keep the plan outside the original export. No automatic repair occurs.

After approval, substitute the actual plan path and a new destination in this cell:

```python
import json
from src.label_diagnostics import create_reviewed_copy
PLAN = BASE / 'reviewed_plan.json'  # actual approved plan, not the schema example
CLEAN = PROJECT_ROOT / 'datasets/fire_smoke_v14_reviewed'  # must not exist
manifest = create_reviewed_copy(DATASET_DIR, CLEAN, PLAN)
print('Full post-correction audit ready:', manifest['ready'])
print(write_diagnosis(diagnose_labels(CLEAN), BASE / 'phase2b_clean_diagnosis'))
clean_audit = json.loads((CLEAN / 'repair_audit.json').read_text())
assert clean_audit['ready'], 'Copy still fails audit; training remains blocked'
RUN_TRAINING = False
```

Copy creation preserves all images and original annotations, verifies byte hashes, changes only approved invalid lines and the copied YAML, and automatically runs a full audit. `repair_manifest.json` records originals, copies and exact changes. Failed audits remain blocked. To later use a passing copy, explicitly select it and rerun the notebook's original audit and visual inspection; no training is authorized in Phase 2B.

## Offline regression checks

```powershell
.venv/Scripts/python.exe scripts/validate_label_diagnostics.py
.venv/Scripts/python.exe scripts/validate_fire_smoke.py
.venv/Scripts/python.exe scripts/validate_notebooks.py
```

All data used by these checks is synthetic. No model or real dataset is downloaded and no training runs.

Verified locally on 2026-11-10: all 11 diagnostic regression tests passed; all 6 audit tests passed; the default notebook smoke execution passed with network/model/training/inference forbidden; all four notebooks passed syntax, clean-output and official nbformat schema validation. `git diff --check` passed. No real-data correction or post-correction audit has occurred because the original export still needs restoration in Colab.
