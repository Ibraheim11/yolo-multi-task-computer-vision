# Phase 2C: read-only source provenance verification

The supplied Colab evidence is 5,596 images, 8,867 valid detection rows and seven unusual rows in six files. All seven pass polygon checks and repeat the closing point. Visual previews were reviewed; seven decisions remain pending. These facts do not establish source format. No real export or Drive packet is available locally; local tests use synthetic data only.

## What COCO can establish

An independently obtained export of workspace `gfo-fc1de`, project `fire-and-smoke-detector-yykya`, version **14**, can provide corroborating geometry. COCO links annotations to images by IDs **within each JSON**, maps category IDs through category names, stores pixel polygons in `segmentation`, and stores top-left pixel `[x,y,width,height]` boxes in `bbox`. IDs and filenames need not agree between formats or splits.

Sources: [Roboflow COCO format](https://roboflow.com/formats/coco-json), [official COCO schema](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm), [Roboflow segmentation exports](https://rfdetr.roboflow.com/latest/cookbooks/fine-tune_segmentation/).

Object Detection project classification does not prove segmentation exists in this particular export. A detection COCO export might contain empty, missing, or generated segmentation. A box match establishes box correspondence only. Even a matching polygon is export correspondence, not proof of original authoring history or intended annotation type. Preserve export receipt/version evidence and inspect the source annotation editor or seek the dataset owner's clarification if needed. The failed Universe image query provides no evidence either way.

## Utility behavior and limits

`src/coco_provenance.py::verify_coco` revalidates the existing review packet against the unchanged original export; it reads but never updates decisions. It discovers `*_annotations*.coco.json` recursively. Nonstandard JSON names must be investigated rather than silently treated as matching exports. Missing exports produce `unavailable`; malformed IDs, unsafe paths, mismatched image dimensions or changed source/review data raise an error without producing a report.

Every COCO image is decoded and hashed. Identity requires equal dimensions and either equal file SHA-256 or exactly equal decoded RGB pixels. Filenames only locate COCO files; they are never identity evidence. Lossy recompression, resizing, EXIF transforms or ambiguous duplicate content require investigation; no fuzzy matching or coordinate rescaling is inferred. IDs are scoped to the containing JSON. Multiple matching image records remain inconclusive, even across splits.

Each report retains every candidate annotation's source category, ID, bbox, segmentation, crowd flag and other fields. Categories match exact names, never numeric IDs across exports. Normalized review vertices are multiplied by decoded width/height. Whole single-ring polygons are compared with a default **0.05 pixel absolute coordinate tolerance** (recorded in the report), allowing only cyclic start, reversed winding and an optional exact closing repeat. There is no simplification, clipping or arbitrary vertex removal. Bboxes are separately compared to polygon extrema using the same tolerance. RLE and multipart segmentation remain inconclusive. A unique same-category whole-polygon match with matching box and noncrowd annotation yields `export_polygon_correspondence`; it never authorizes conversion. Conflicts and missing geometry yield `inconclusive_geometry_or_category`.

Annotation 6 must be identified by the existing packet's sixth item **and its printed ID/raw text**, not COCO annotation ID 6. The nine tokens describe three distinct vertices plus a repeated first vertex in the current interpretation. That cannot be a nondegenerate four-corner OBB. The utility reports this and records whether the COCO ring itself retains the closing repeat. A matching triangle with omitted closure is geometric correspondence, not proof that the YOLO exporter intentionally emitted a polygon. Missing/box-only source data leaves format unresolved. Never remove the closing vertex from the original label.

## Exact next Colab cells

Upload the updated project code (including `src/coco_provenance.py`) to the current Colab project. Keep the restored original dataset and **existing** Drive review folder. Do not regenerate the packet or run the interactive decision-writing cell from the earlier recovery guide. CPU is sufficient; do not run all notebook cells.

```python
from pathlib import Path
from datetime import datetime, timezone
import json
from google.colab import drive
drive.mount('/content/drive')
%cd /content/yolo-multi-task-computer-vision
RUN_TRAINING = RUN_VALIDATION = RUN_VIDEO_INFERENCE = DOWNLOAD_DATASET = False
DATASET_DIR = Path('/content/yolo-multi-task-computer-vision/datasets/fire_smoke_v14')
BASE = Path('/content/drive/MyDrive/YOLO_Multi_Task/fire_smoke_v14')
# List existing packets; choose the one whose previews were already reviewed.
print(*sorted(BASE.glob('polygon_review_*/review.json')), sep='\n')
```

```python
REVIEW_DIR = BASE / 'REPLACE_WITH_EXISTING_REVIEW_FOLDER'
assert (REVIEW_DIR / 'review.json').is_file()
assert (REVIEW_DIR / 'decisions.json').is_file()
packet = json.loads((REVIEW_DIR / 'review.json').read_text())
assert len(packet['items']) == 7
assert len({i['path'] for i in packet['items']}) == 6
six = packet['items'][5]
print('Annotation 6:', six['id'], repr(six['original']), six['proposal'])
assert six['proposal']['token_count'] == 9
assert six['proposal']['unique_points'] == 3
assert six['proposal']['repeated_closing_point']
```

Obtain the **original version 14 COCO JSON export with images** from Roboflow's export UI, choosing that workspace/project/version explicitly. Download it separately; preserve the archive and its export receipt or screenshot on Drive. Extract to a new directory such as `/content/fire_smoke_v14_coco_source`, never over the YOLO export or review folder. Do not convert the suspect YOLO rows to COCO: that would be circular evidence. Do not regenerate a dataset version. If COCO or original polygons are unavailable, record that outcome and leave approvals pending. Do not expose an API key in notebook output. This procedure deliberately leaves acquisition outside the verification utility.

```python
from src.coco_provenance import verify_coco
from src.fire_smoke import sha256
COCO_DIR = Path('/content/fire_smoke_v14_coco_source')
print(*sorted(COCO_DIR.rglob('*_annotations*.coco.json')), sep='\n')
# If files have other names/layouts, inspect them before proceeding.
decision_hash = sha256(REVIEW_DIR / 'decisions.json')
REPORT = BASE / ('provenance_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')) / 'report.json'
report = verify_coco(DATASET_DIR, REVIEW_DIR, COCO_DIR, REPORT,
    source_note='REPLACE with actual export date, workspace/project/version and preserved receipt/archive location; or explain unavailability')
assert sha256(REVIEW_DIR / 'decisions.json') == decision_hash
for row in report['items']:
    print(row['review_index'], row['id'], row['status'])
print(json.dumps(report['items'][5], indent=2))
print('Saved read-only evidence:', REPORT)
RUN_TRAINING = RUN_VALIDATION = RUN_VIDEO_INFERENCE = DOWNLOAD_DATASET = False
```

Review each candidate's `source`, `category_match`, `whole_polygon_match`, `bbox_match` and closure flags, especially item 6. Compare export/version evidence independently. If an error occurs, preserve inputs and investigate; do not bypass it with fabricated mappings or edited review proposals. The output hashes record the JSONs, COCO images, affected original labels/images, YAML and existing review/decision files; inputs are rehashed before writing the report. Reports use exclusive creation outside all input directories. The utility writes no plan, clean copy, approval or training YAML. This phase ends with an evidence report and seven pending decisions unless a later explicit human review resolves them.

## Automated validation

```powershell
.venv/Scripts/python.exe scripts/validate_coco_provenance.py
.venv/Scripts/python.exe scripts/validate_label_diagnostics.py
.venv/Scripts/python.exe scripts/validate_fire_smoke.py
.venv/Scripts/python.exe scripts/validate_notebooks.py
git diff --check
```

Tests exercise renamed images, unrelated IDs, closure/winding/start invariance, tolerance, bbox/category conflicts, absent segmentation/export, RLE/multipart, ambiguous duplicate images/annotations, changed reviews, unsafe paths, dimensions and dangling IDs. They also compare complete input directory inventories to prove preservation. No real provenance has been confirmed locally. No push or training is part of this procedure.
