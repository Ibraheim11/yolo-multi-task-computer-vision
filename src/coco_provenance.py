"""Read-only, offline COCO export comparison. Never authorizes a conversion."""
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from src.fire_smoke import locate_yaml, sha256
from src.label_diagnostics import diagnose_labels
from src.polygon_review import _items


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe source path: {relative}")
    return path


def fingerprint(path):
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("EXIF transform requires manual coordinate review")
        rgb = image.convert("RGB")
        return {"size": list(rgb.size), "sha256": sha256(path),
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}


def numbers(values, count=None):
    return (isinstance(values, list) and (count is None or len(values) == count)
            and all(type(v) in (int, float) and math.isfinite(v) for v in values))


def ring_match(points, raw, width, height, tolerance):
    """Compare complete rings, allowing cyclic start, winding and closing repeat only."""
    if not numbers(raw) or len(raw) < 6 or len(raw) % 2:
        return False
    source = list(zip(raw[::2], raw[1::2]))
    if source[0] == source[-1]:
        source = source[:-1]
    target = [(x * width, y * height) for x, y in points]
    if target[0] == target[-1]:
        target = target[:-1]
    if len(source) != len(target):
        return False
    for order in (source, source[::-1]):
        for start in range(len(order)):
            shifted = order[start:] + order[:start]
            if all(abs(a - b) <= tolerance for p, q in zip(target, shifted) for a, b in zip(p, q)):
                return True
    return False


def verify_coco(dataset_dir, review_dir, coco_dir, output_file, *, source_note="", tolerance_px=0.05):
    """Compare an existing review packet with all *_annotations*.coco.json files.

    Requires a new output file outside every input directory. Missing COCO exports
    produce an unavailable report. Malformed exports fail closed without a report.
    The caller records how the export was obtained; this assertion is not verified.
    """
    root, review, coco = [Path(p).resolve() for p in (dataset_dir, review_dir, coco_dir)]
    output = Path(output_file).resolve()
    if any(output.is_relative_to(p) for p in (root, review, coco)) or output.exists():
        raise ValueError("Use a new output file outside all input directories")
    if not math.isfinite(tolerance_px) or not 0 <= tolerance_px <= 0.5:
        raise ValueError("Tolerance must be between zero and 0.5 pixels")
    packet_path, decisions_path = review / "review.json", review / "decisions.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    current = _items(root, diagnose_labels(root))
    if not current or current != [{k: v for k, v in i.items() if k != "preview"} for i in packet["items"]]:
        raise ValueError("Source or review proposal changed; investigate before comparison")
    inputs = {str(packet_path): sha256(packet_path), str(decisions_path): sha256(decisions_path)}
    yaml_path = locate_yaml(root)
    inputs[str(yaml_path)] = sha256(yaml_path)
    for item in current:
        for key in ("path", "image"):
            path = inside(root, item[key])
            inputs[str(path)] = sha256(path)
    records, exports = [], []
    for path in sorted(coco.rglob("*_annotations*.coco.json")):
        path = inside(coco, str(path.relative_to(coco)))
        inputs[str(path)] = sha256(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in data["categories"]}
        images = {i["id"]: i for i in data["images"]}
        annotations = data["annotations"]
        if (len(categories) != len(data["categories"]) or len(images) != len(data["images"])
                or len({a["id"] for a in annotations}) != len(annotations)):
            raise ValueError("Duplicate IDs inside COCO JSON")
        if any(a["image_id"] not in images or a["category_id"] not in categories for a in annotations):
            raise ValueError("Dangling COCO image/category reference")
        exports.append({"json": str(path), "sha256": inputs[str(path)], "info": data.get("info"),
                        "images": len(images), "annotations": len(annotations), "categories": data["categories"]})
        by_image = {image_id: [] for image_id in images}
        for annotation in annotations:
            by_image[annotation["image_id"]].append(dict(annotation, category_name=categories[annotation["category_id"]]))
        for image_id, image in images.items():
            file = inside(coco, str((path.parent / image["file_name"]).relative_to(coco)))
            fp = fingerprint(file)
            inputs[str(file)] = fp["sha256"]
            if fp["size"] != [image["width"], image["height"]]:
                raise ValueError("COCO dimensions disagree with decoded image")
            records.append({"json": str(path), "image": image, "fingerprint": fp,
                            "annotations": by_image[image_id]})
    results = []
    for index, item in enumerate(current, 1):
        fp = fingerprint(inside(root, item["image"]))
        matches = [r for r in records if r["fingerprint"]["size"] == fp["size"] and
                   (r["fingerprint"]["sha256"] == fp["sha256"] or
                    r["fingerprint"]["rgb_sha256"] == fp["rgb_sha256"])]
        proposal = item["proposal"]
        row = {"review_index": index, "id": item["id"], "original": item["original"],
               "proposal": proposal, "image_identity_candidates": len(matches), "candidates": [],
               "status": "unavailable" if not exports else "inconclusive_image_identity"}
        for record in matches:
            width, height = fp["size"]
            candidate = {"json": record["json"], "image": record["image"],
                         "identity": "bytes" if record["fingerprint"]["sha256"] == fp["sha256"] else "decoded_RGB",
                         "annotations": []}
            for ann in record["annotations"]:
                seg = ann.get("segmentation")
                rings = seg if isinstance(seg, list) else []
                geometry = bool(proposal and len(rings) == 1 and
                                ring_match(proposal["points"], rings[0], width, height, tolerance_px))
                box = ann.get("bbox")
                bbox_match = False
                if proposal and numbers(box, 4) and box[2] > 0 and box[3] > 0:
                    l, t, r, b = proposal["box_xyxy"]
                    expected = [l * width, t * height, (r-l) * width, (b-t) * height]
                    bbox_match = all(abs(a-b) <= tolerance_px for a, b in zip(box, expected))
                candidate["annotations"].append({"source": ann,
                    "category_match": bool(proposal and ann["category_name"] == proposal["class_name"]),
                    "whole_polygon_match": geometry, "bbox_match": bbox_match,
                    "segmentation_kind": "RLE_unsupported" if isinstance(seg, dict) else
                        "polygon_list" if rings else "absent_or_empty",
                    "source_repeated_closure": [bool(numbers(r) and len(r) >= 6 and r[:2] == r[-2:]) for r in rings]})
            row["candidates"].append(candidate)
        if len(matches) == 1:
            annotations = row["candidates"][0]["annotations"]
            hits = [a for a in annotations if a["category_match"] and a["whole_polygon_match"]]
            row["status"] = ("export_polygon_correspondence" if len(hits) == 1 and hits[0]["bbox_match"]
                             and hits[0]["source"].get("iscrowd", 0) == 0 else "inconclusive_geometry_or_category")
        if proposal and proposal["nine_token_format_ambiguous"]:
            row["nine_token_investigation"] = {
                "unique_vertices": proposal["unique_points"],
                "repeated_closing_vertex": proposal["repeated_closing_point"],
                "interpretation": ("Three vertices plus repeated closure cannot establish a four-corner OBB; source evidence still required."
                    if proposal["unique_points"] == 3 and proposal["repeated_closing_point"] else
                    "Nine tokens alone do not distinguish polygon from OBB; source evidence required."),
                "annotation_6": index == 6}
        results.append(row)
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise ValueError("Input changed during verification")
    report = {"training_enabled": False, "conversion_authorized": False,
              "source_note_unverified": source_note, "tolerance_px": tolerance_px,
              "limitations": "Export correspondence is not original authoring history or visual approval. "
                  "BBox-only, RLE, multipart, transformed/recompressed images or ambiguous identity remain inconclusive. "
                  "A matching export may contain generated geometry; establish workspace/project/version independently.",
              "input_sha256": inputs, "exports": exports, "items": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
