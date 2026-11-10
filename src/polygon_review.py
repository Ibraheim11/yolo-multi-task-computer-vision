"""Offline polygon proposals and explicit human review; no model imports/actions."""
import json
import math
from pathlib import Path

from PIL import Image
import yaml

from src.fire_smoke import IMAGE_SUFFIXES, class_names, locate_yaml, resolve_split, sha256
from src.label_diagnostics import diagnose_labels, write_diagnosis


def polygon_proposal(raw, names):
    """Validate a simple polygon and propose its enclosing box; never approve it."""
    parts = raw.split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError("Expected class plus at least three complete x/y pairs")
    cls = int(parts[0])
    if cls not in range(len(names)):
        raise ValueError("Class ID outside names range")
    values = [float(p) for p in parts[1:]]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Coordinates must be finite and normalized within [0, 1]")
    points = list(zip(values[::2], values[1::2]))
    closing = points[0] == points[-1]
    vertices = points[:-1] if closing else points
    if len(set(vertices)) < 3 or len(set(vertices)) != len(vertices):
        raise ValueError("Require at least three unique vertices; only a closing repeat is allowed")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, p):
        return cross(a, b, p) == 0 and all(min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in (0, 1))

    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    for i, (a, b) in enumerate(edges):
        # Adjacent edges may share an endpoint, but must not double back.
        c = vertices[(i + 2) % len(vertices)]
        if on(a, b, c) or on(b, c, a):
            raise ValueError("Overlapping adjacent polygon edges require source review")
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            c, d = edges[j]
            if (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
                    or any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))):
                raise ValueError("Self-intersecting or touching polygon edges require source review")
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)) / 2
    xs, ys = values[::2], values[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if area <= 1e-12 or right <= left or bottom <= top:
        raise ValueError("Degenerate polygon or zero-sized bounding box")
    box = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]
    return {"class_id": cls, "class_name": names[cls], "token_count": len(parts),
            "points": [list(p) for p in points], "unique_points": len(set(vertices)),
            "repeated_closing_point": closing, "area": area,
            "nine_token_format_ambiguous": len(parts) == 9,
            "box_xyxy": [left, top, right, bottom], "box_xywh": box,
            "replacement": str(cls) + " " + " ".join(format(v, ".17g") for v in box)}


def _items(root, diagnosis):
    config_path = locate_yaml(root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = class_names(config)
    items = []
    for file in diagnosis["files"]:
        images = resolve_split(config_path, config[file["split"]], root)
        label = Path(file["path"])
        relative = label.relative_to(images.parent / "labels").with_suffix("")
        matches = [p for p in (images / relative.parent).iterdir()
                   if p.stem == relative.name and p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
        if len(matches) != 1 or not matches[0].resolve().is_relative_to(root):
            raise ValueError(f"Expected one in-export image for {label}")
        image = matches[0]
        with Image.open(image) as source:
            if source.getexif().get(274, 1) != 1:
                raise ValueError("Nontrivial EXIF orientation requires explicit coordinate review")
        for row in file["findings"]:
            item = {"id": f"{file['relative_path']}:{row['line']}", "path": file["relative_path"],
                    "line": row["line"], "original": row["raw"], "label_sha256": sha256(label),
                    "image": image.relative_to(root).as_posix(), "image_sha256": sha256(image),
                    "visual_match": "unreviewed", "proposal": None}
            try:
                item["proposal"] = polygon_proposal(row["raw"], names)
            except (TypeError, ValueError) as exc:
                item["blocked_reason"] = str(exc)
            items.append(item)
    return items


def prepare_visual_review(dataset_dir, saved_report, output_dir):
    """Save diagnosis, PNGs, and pending decisions outside the untouched export."""
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Use a new review directory outside the original export")
    diagnosis = diagnose_labels(root, saved_report)
    if any(r["status"] != "still_invalid" for r in diagnosis["saved_issue_reconciliation"]):
        raise ValueError("Saved issues changed or are unresolved; investigate export identity first")
    items = _items(root, diagnosis)
    output.mkdir(parents=True)
    write_diagnosis(diagnosis, output / "diagnosis")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    for index, item in enumerate(items, 1):
        with Image.open(root / item["image"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        figure, axes = plt.subplots(1, 2, figsize=(18, 9))
        try:
            for ax in axes:
                ax.imshow(image)
                ax.axis("off")
            axes[0].set_title("Original image — inspect visible object")
            axes[1].set_title("Cyan: original vertices | yellow: proposed detection box")
            proposal = item["proposal"]
            if proposal:
                axes[1].add_patch(Polygon([(x * width, y * height) for x, y in proposal["points"]],
                                         fill=False, edgecolor="cyan", linewidth=2))
                axes[1].plot(*zip(*[(x * width, y * height) for x, y in proposal["points"]]),
                             linestyle="none", marker=".", color="cyan")
                left, top, right, bottom = proposal["box_xyxy"]
                axes[1].add_patch(Rectangle((left * width, top * height), (right - left) * width,
                                           (bottom - top) * height, fill=False, edgecolor="yellow", linewidth=2))
                caption = (f"Class: {proposal['class_name']} ({proposal['class_id']}) | tokens: {proposal['token_count']} | "
                           f"closing point repeated: {proposal['repeated_closing_point']}\n"
                           + ("NINE TOKENS: confirm polygon vs OBB from source. " if proposal['nine_token_format_ambiguous'] else "")
                           + "UNREVIEWED: visible object alignment requires human confirmation")
            else:
                caption = "BLOCKED: " + item["blocked_reason"]
            figure.suptitle(f"{item['image']} | original line {item['line']}\n{caption}", fontsize=12, wrap=True)
            figure.tight_layout(rect=(0, 0, 1, 0.88))
            figure.savefig(output / f"review_{index:02d}.png", dpi=140)
        finally:
            plt.close(figure)
        item["preview"] = f"review_{index:02d}.png"
    packet = {"dataset": str(root), "items": items, "note": "Geometry is checked; image match is unreviewed."}
    (output / "review.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    decisions = {"reviewer": "", "items": [{"id": item["id"], "approve": False,
        "confirmed_format": "", "format_evidence": "", "matches_visible_object": False,
        "class_confirmed": False, "reason": ""} for item in items]}
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return packet


def write_reviewed_plan(dataset_dir, review_dir):
    """Recheck source bytes and geometry; require explicit approval of every row."""
    root, review = Path(dataset_dir).resolve(), Path(review_dir).resolve()
    if review.is_relative_to(root):
        raise ValueError("Review and plan must be outside the original export")
    packet = json.loads((review / "review.json").read_text(encoding="utf-8"))
    decisions = json.loads((review / "decisions.json").read_text(encoding="utf-8"))
    current = _items(root, diagnose_labels(root))
    expected = [{k: v for k, v in item.items() if k != "preview"} for item in packet["items"]]
    if not current or current != expected or packet["dataset"] != str(root):
        raise ValueError("Source/proposals changed or no issues remain; regenerate visual review")
    rows = decisions.get("items", [])
    if len(rows) != len(current) or {r["id"] for r in rows} != {r["id"] for r in current}:
        raise ValueError("Decisions must cover every finding exactly once")
    reviewer = decisions.get("reviewer", "").strip()
    if not reviewer:
        raise ValueError("Reviewer identity is required")
    by_id = {r["id"]: r for r in rows}
    files = {}
    for item in current:
        decision = by_id[item["id"]]
        if (not item["proposal"] or any(decision.get(k) is not True for k in
                ("approve", "matches_visible_object", "class_confirmed"))
                or decision.get("confirmed_format") not in ("polygon", "obb")
                or any(not isinstance(decision.get(k), str) or not decision[k].strip()
                       for k in ("reason", "format_evidence"))):
            raise ValueError(f"Human/source review incomplete or blocked: {item['id']}")
        if decision["confirmed_format"] == "obb" and (not item["proposal"]["nine_token_format_ambiguous"]
                or item["proposal"]["unique_points"] != 4 or item["proposal"]["repeated_closing_point"]):
            raise ValueError("OBB confirmation requires exactly four corners")
        entry = files.setdefault(item["path"], {"path": item["path"],
            "original_sha256": item["label_sha256"], "replacements": []})
        entry["replacements"].append({"line": item["line"], "original": item["original"],
            "replacement": item["proposal"]["replacement"],
            "reason": f"{reviewer}: {decision['reason']}; format={decision['confirmed_format']}; evidence={decision['format_evidence']}",
            "visual_review": decision, "image": item["image"], "image_sha256": item["image_sha256"]})
    plan = {"files": list(files.values()), "source_yaml_sha256": sha256(locate_yaml(root)),
        "review": {"reviewer": reviewer,
        "review_sha256": sha256(review / "review.json"), "decisions_sha256": sha256(review / "decisions.json")}}
    with (review / "reviewed_plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
    return review / "reviewed_plan.json"
