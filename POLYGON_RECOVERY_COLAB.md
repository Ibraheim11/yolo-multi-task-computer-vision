# Fire & Smoke polygon recovery: reviewed Colab workflow

For the current seven pending decisions, first follow [Phase 2C read-only COCO provenance verification](PHASE2C_PROVENANCE_COLAB.md). Reuse the existing review packet and preserve decisions; do not rerun the decision-writing loop while investigating provenance.

The user reports seven polygon-like rows, token counts **119, 11, 11, 13, 17, 9, 11**, in six files, all reconciled as `still_invalid`, with no additional audit issues. This is diagnostic evidence, not confirmation that any polygon matches a visible object. No real image review, conversion or clean-dataset audit has been performed locally.

The original `create_reviewed_copy` schema remains compatible: `files` contains relative label paths, original SHA-256 hashes, and one-based line replacements with exact original text and a reason. The new workflow fills that schema only after explicit human review. Added review metadata records image hashes and reviewer decisions; the copy manifest retains it. Nothing runs training or inference.

## 1. Restore the updated code and use the original Colab export

Upload or update the complete project, including `src/polygon_review.py`, under `/content/yolo-multi-task-computer-vision`. Use the existing notebook dependency setup if needed. CPU is sufficient. Do not run all notebook cells. If the runtime lost the export, follow [the original restoration procedure](PHASE2B_COLAB.md) in Colab only, preserving the saved audit first. Do not reuse a previously cleaned export as the original.

```python
from pathlib import Path
from datetime import datetime, timezone
import json, os
from google.colab import drive
drive.mount('/content/drive')
%cd /content/yolo-multi-task-computer-vision
PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / 'datasets/fire_smoke_v14'
BASE = Path('/content/drive/MyDrive/YOLO_Multi_Task/fire_smoke_v14')
SAVED_REPORT = BASE / 'dataset_audit_original_phase2b/report.json'
# If your preserved report is elsewhere, set its actual path above.
assert DATASET_DIR.is_dir() and SAVED_REPORT.is_file()
RUN_TRAINING = RUN_VALIDATION = RUN_VIDEO_INFERENCE = False
DOWNLOAD_DATASET = False
os.environ.setdefault('MPLCONFIGDIR', '/content/matplotlib-config')
from src.label_diagnostics import diagnose_labels, create_reviewed_copy, write_diagnosis
from src.polygon_review import prepare_visual_review, write_reviewed_plan
from src.fire_smoke import audit_dataset

diagnosis = diagnose_labels(DATASET_DIR, SAVED_REPORT)
assert diagnosis['original_audit_issue_count'] == 7
assert diagnosis['current_audit_issue_count'] == 7
assert diagnosis['invalid_annotation_file_count'] == 6
assert diagnosis['diagnostic_finding_count'] == 7
assert diagnosis['names'] == ['fire', 'smoke']
assert len(diagnosis['saved_issue_reconciliation']) == 7
assert all(r['status'] == 'still_invalid' for r in diagnosis['saved_issue_reconciliation'])
assert sorted(r['field_count'] for r in diagnosis['saved_issue_reconciliation']) == sorted([119,11,11,13,17,9,11])
source_audit = audit_dataset(DATASET_DIR)
assert sum(s['images'] for s in source_audit['splits'].values()) == 5596
assert len(source_audit['issues']) == 7
```

Any assertion failure requires investigation before reviewing or correcting. Do not force the expected counts or change `ready` manually.

## 2. Generate the review packet directly on Drive

```python
REVIEW_DIR = BASE / ('polygon_review_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
packet = prepare_visual_review(DATASET_DIR, SAVED_REPORT, REVIEW_DIR)
(REVIEW_DIR / 'source_audit.json').write_text(json.dumps(source_audit, indent=2))
assert len(packet['items']) == 7
assert len({r['path'] for r in packet['items']}) == 6
print('Saved review:', REVIEW_DIR)
```

This creates `diagnosis/diagnosis.json`, `diagnosis/diagnosis.txt`, `diagnosis/invalid_lines.csv`, seven PNG previews, `review.json`, and pending `decisions.json`. Each preview shows the unaltered decoded image beside the original cyan polygon/vertices and yellow axis-aligned bounding box. It names the class, image, original line, token count, and repeated-closing-point status. Preview annotations never alter original images. Nontrivial EXIF orientation blocks preparation so display transforms cannot silently misalign coordinates.

Geometry checks require an integer in-range class ID, finite coordinates in `[0,1]`, complete x/y pairs, at least three unique vertices, nonzero area, positive box dimensions, and bounds within the normalized image. A final vertex equal to the first is reported and accepted without changing the original line. Other duplicate vertices, touching/crossing edges and overlapping edges block conversion. No clipping or truncation occurs. Geometry checks cannot determine whether the object or class is correct.

## 3. Inspect and explicitly decide on each of the seven rows

Run this cell interactively. Each question defaults to no. Answer only after inspecting the full-resolution PNG and the source annotation/export. Decisions are saved after every row, so interruptions preserve progress. To resume without regenerating the packet, set `REVIEW_DIR` to its printed Drive path and reload `review.json`.

```python
from IPython.display import display, Image as DisplayImage
packet = json.loads((REVIEW_DIR / 'review.json').read_text())
decision_path = REVIEW_DIR / 'decisions.json'
decisions = json.loads(decision_path.read_text())
decisions['reviewer'] = input('Reviewer name or identifier: ').strip()
by_id = {d['id']: d for d in decisions['items']}
for index, item in enumerate(packet['items'], 1):
    print(f"Review {index}/7: {item['id']}")
    print('Exact original:', repr(item['original']))
    print('Geometry:', item.get('proposal') or item.get('blocked_reason'))
    display(DisplayImage(filename=str(REVIEW_DIR / item['preview'])))
    decision = by_id[item['id']]
    decision.update(approve=False, class_confirmed=False, matches_visible_object=False,
                    confirmed_format='', format_evidence='', reason='')
    if item['proposal'] is not None:
        # For nine tokens, inspect source provenance: four corners may be OBB,
        # a four-vertex polygon, or a triangle with a repeated closing point.
        decision['confirmed_format'] = input('Confirmed source format (polygon/obb, blank if unknown): ').strip().lower()
        decision['format_evidence'] = input('Source/export evidence confirming this format: ').strip()
        decision['class_confirmed'] = input('Does the class match this object? [yes/no]: ').strip().lower() == 'yes'
        decision['matches_visible_object'] = input('Do vertices and box match the visible object extent? [yes/no]: ').strip().lower() == 'yes'
        decision['reason'] = input('Review notes: object boundary, box suitability, any uncertainty: ').strip()
        decision['approve'] = input('Approve this exact min/max conversion? [yes/no]: ').strip().lower() == 'yes'
    else:
        print('BLOCKED: obtain corrected source annotations or investigate; do not guess.')
    decision_path.write_text(json.dumps(decisions, indent=2))
```

Inspect whether the polygon follows the intended flame/smoke instance, whether it includes unrelated objects or misses visible extent, whether the class is correct, and whether losing segmentation/orientation information is appropriate for detection. Do not approve solely because a box looks plausible. A nine-token row receives no inferred format: require source evidence for `polygon` or `obb`. OBB confirmation requires four unique corners, without a closing repeat. An OBB still needs an explicit decision that its enclosing axis-aligned box suits the detection task. Unknown/ambiguous annotations stay unapproved, and plan creation stops until every issue is resolved. Return to the source or obtain a corrected export; never invent missing coordinates.

## 4. Generate the existing-schema reviewed plan (no copy yet)

```python
PLAN = write_reviewed_plan(DATASET_DIR, REVIEW_DIR)
print(PLAN.read_text())  # inspect all seven exact proposed replacements
```

This re-audits the source, recomputes geometry, and compares all label/image hashes with the packet. It requires every finding to have explicit approval, class confirmation, object-match confirmation, format evidence, reviewer identity, and review notes. New or changed findings block the plan. A blocked row cannot be approved by setting flags. An existing `reviewed_plan.json` is never overwritten; use a new review directory for a revised plan.

For each confirmed instance, the generated detection row is:

`class (xmin+xmax)/2 (ymin+ymax)/2 (xmax-xmin) (ymax-ymin)`

All coordinate values come from the original vertices. The plan changes neither images nor annotations. It records exact rows, label/image hashes, reviewer decisions and review-file hashes. Review the plan before the next, separate action.

## 5. Only after reviewing the plan, create a separate cleaned dataset on Drive

Run this cell deliberately after approving the plan. It never overwrites a destination. Images, valid labels, and untouched rows stay byte-identical. BOM/newline conventions are preserved in edited labels. The copied YAML points to the cleaned splits. The original export remains unchanged and the manifest records its full hash inventory.

```python
CLEAN = REVIEW_DIR / 'cleaned_detection_dataset'  # new separate dataset, on Drive
assert not CLEAN.exists()
manifest = create_reviewed_copy(DATASET_DIR, CLEAN, PLAN)
# Manifest and automatic full audit already reside on Drive inside CLEAN.
clean_audit = audit_dataset(CLEAN)  # independent full recheck after conversion
(REVIEW_DIR / 'clean_audit.json').write_text(json.dumps(clean_audit, indent=2))
print(write_diagnosis(diagnose_labels(CLEAN), REVIEW_DIR / 'clean_diagnosis'))
assert manifest['ready'] and clean_audit['ready'] is True
assert clean_audit['issues'] == []
assert clean_audit['names'] == ['fire', 'smoke']
assert {k: v['images'] for k,v in clean_audit['splits'].items()} == {'train':3940, 'val':1017, 'test':639}
assert all(v['invalid_labels'] == 0 for v in clean_audit['splits'].values())
RUN_TRAINING = RUN_VALIDATION = RUN_VIDEO_INFERENCE = False
print('Clean audit passed. Training and inference remain disabled.')
```

Inspect `CLEAN/repair_manifest.json` and `CLEAN/repair_audit.json` as well as `REVIEW_DIR/clean_audit.json`. Failed copies remain available for inspection and must not be used for training. Do not change `ready` or delete samples to make checks pass. Diagnosis, previews, decisions, plan, manifest, cleaned dataset and audit are all under this Drive review folder. No executed notebook or generated review artifacts should enter Git. A clean audit is necessary before any future separately authorized training run; rerun the original notebook audit with the explicitly selected clean dataset at that time.

## Local validation

Run `.venv/Scripts/python.exe scripts/validate_label_diagnostics.py`, `.venv/Scripts/python.exe scripts/validate_fire_smoke.py`, `.venv/Scripts/python.exe scripts/validate_notebooks.py`, and `git diff --check`. Tests use only synthetic fixtures, including seven rows with the reported token counts across six files. Synthetic approval is not evidence about real images.

Verified on 2026-11-10: 18 diagnostic/recovery tests passed; 6 audit tests and the guarded default notebook execution passed; all four notebooks passed schema/syntax validation. All five Python execution blocks above passed syntax checks with the Colab directory magic normalized. All six changed/new files passed whitespace checks, and the synthetic preview layout was visually inspected. No real dataset was downloaded, reviewed, converted or trained locally.
