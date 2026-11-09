"""Offline YOLO detection dataset audit. Never downloads data or trains models."""
from collections import Counter
from pathlib import Path
import hashlib
import math
import random

import yaml
from PIL import Image, ImageOps, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_yaml(dataset_dir):
    files = sorted(Path(dataset_dir).rglob("data.yaml"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one data.yaml; found {len(files)}. Inspect the export directory.")
    return files[0]


def class_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        if not all(isinstance(k, (int, str)) and not isinstance(k, bool)
                   and str(k) == str(int(k)) for k in names):
            raise ValueError("Class IDs must be integer keys")
        names = {int(k): v for k, v in names.items()}
        if set(names) != set(range(len(names))):
            raise ValueError("Class IDs must be contiguous integers starting at zero")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise ValueError("data.yaml must contain nonempty class names")
    if "nc" in config and config["nc"] != len(names):
        raise ValueError("nc does not match names")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    return names


def resolve_split(yaml_path, value, dataset_dir):
    """Resolve Roboflow directory exports, including ../train/images layout.

    Only existing paths inside the selected export are accepted. Ambiguous or
    unsupported list/manifest formats are rejected instead of silently guessed.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Split must specify an image directory string")
    base = yaml_path.parent
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    path_value = config.get("path") or "."
    if not isinstance(path_value, str):
        raise ValueError("Dataset path must be a string")
    declared_root = Path(path_value)
    if not declared_root.is_absolute():
        declared_root = base / declared_root
    raw = Path(value.replace("\\", "/"))
    candidates = [raw] if raw.is_absolute() else [declared_root / raw, base / raw, Path(dataset_dir) / raw]
    # Common Roboflow exports use ../train/images beside data.yaml.
    if value.replace("\\", "/").startswith("../"):
        candidates.append(base / value.replace("\\", "/")[3:])
    export_root = Path(dataset_dir).resolve()
    matches = {p.resolve() for p in candidates if p.is_dir() and p.resolve().is_relative_to(export_root)}
    if len(matches) != 1:
        raise ValueError(f"Expected one existing in-export directory for split {value!r}; found {len(matches)}")
    return matches.pop()


def parse_label(path, names):
    boxes, issues = [], []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"Unreadable annotation ({type(exc).__name__})"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("expected 5 fields: class x_center y_center width height")
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if cls < 0 or cls >= len(names):
                raise ValueError("class ID outside names range")
            if not all(math.isfinite(v) for v in coords):
                raise ValueError("nonfinite box coordinate")
            x, y, w, h = coords
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("coordinates must be normalized; width/height must be positive")
            if min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1 + 1e-5:
                raise ValueError("box extends outside image")
            box = (cls, x, y, w, h)
            if box in boxes:
                raise ValueError("duplicate box row")
            boxes.append(box)
        except ValueError as exc:
            issues.append(f"line {number}: {exc}")
    return boxes, issues


def audit_dataset(dataset_dir):
    yaml_path = locate_yaml(dataset_dir)
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("data.yaml must be a mapping")
    names = class_names(config)
    report = {"source_yaml": str(yaml_path), "source_yaml_sha256": sha256(yaml_path),
              "names": names, "splits": {}, "issues": [], "records": []}
    hashes = {}
    def issue(split, path, message):
        report["issues"].append({"split": split, "path": str(path), "message": message})
    for split in ("train", "val", "test"):
        summary = {"images": 0, "annotation_files": 0, "missing_labels": 0, "empty_labels": 0,
                   "corrupt_images": 0, "invalid_labels": 0, "orphan_labels": 0, "boxes": 0,
                   "class_instances": dict.fromkeys(names, 0)}
        report["splits"][split] = summary
        try:
            images_dir = resolve_split(yaml_path, config.get(split), dataset_dir)
            if images_dir.name != "images":
                raise ValueError("Expected an images/ directory in this Roboflow export")
        except (ValueError, OSError) as exc:
            issue(split, yaml_path, str(exc))
            continue
        summary["images_dir"] = str(images_dir)
        labels_dir = images_dir.parent / "labels"
        images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(labels_dir.rglob("*.txt")) if labels_dir.is_dir() else []
        summary.update(images=len(images), annotation_files=len(labels))
        if not images:
            issue(split, images_dir, "No recognized images")
        if not labels_dir.is_dir():
            issue(split, labels_dir, "Missing labels directory")
        expected_labels = Counter(labels_dir / p.relative_to(images_dir).with_suffix(".txt") for p in images)
        for label, count in expected_labels.items():
            if count > 1:
                issue(split, label, "Multiple images share one annotation stem")
        for label in labels:
            if label not in expected_labels:
                summary["orphan_labels"] += 1
                issue(split, label, "Annotation has no matching image")
                _, errors = parse_label(label, names)
                if errors:
                    summary["invalid_labels"] += 1
                for error in errors:
                    issue(split, label, error)
        for image_path in images:
            good = True
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
                    if min(image.size) <= 0:
                        raise ValueError("Nonpositive image size")
                digest = sha256(image_path)
                previous = hashes.get(digest)
                if previous and previous[0] != split:
                    issue(split, image_path, f"Exact image duplicate across splits: {previous[0]} / {previous[1]}")
                else:
                    hashes[digest] = (split, str(image_path))
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                summary["corrupt_images"] += 1
                issue(split, image_path, f"Unreadable/corrupt image ({type(exc).__name__})")
                good = False
            label = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
            boxes = []
            if not label.is_file():
                summary["missing_labels"] += 1
                issue(split, label, "Missing annotation; review before treating image as background")
                good = False
            else:
                boxes, errors = parse_label(label, names)
                if errors:
                    summary["invalid_labels"] += 1
                    for error in errors:
                        issue(split, label, error)
                    good = False
                elif not boxes:
                    summary["empty_labels"] += 1
                summary["boxes"] += len(boxes)
                for cls, *_ in boxes:
                    summary["class_instances"][names[cls]] += 1
            if good:
                report["records"].append({"split": split, "image": str(image_path), "label": str(label), "boxes": boxes, "image_sha256": digest, "label_sha256": sha256(label)})
    for split in ("train", "val", "test"):
        if report["splits"][split]["images"] and report["splits"][split]["boxes"] == 0:
            issue(split, yaml_path, "No usable foreground boxes; review detection split annotations")
    report["ready"] = not report["issues"] and all(s["images"] > 0 for s in report["splits"].values())
    return report


def write_training_yaml(report, destination):
    """Write only audited paths/names; exclude download hooks and metadata secrets."""
    if not report["ready"]:
        raise ValueError("Resolve dataset audit issues before creating training YAML")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = {split: report["splits"][split]["images_dir"] for split in ("train", "val", "test")}
    config.update(names=report["names"], nc=len(report["names"]))
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def show_samples(report, per_split=2, seed=42):
    """Display only decoded, label-valid examples from the actual export."""
    import matplotlib.pyplot as plt
    rng = random.Random(seed)
    samples = []
    for split in ("train", "val", "test"):
        available = [r for r in report["records"] if r["split"] == split]
        samples.extend(rng.sample(available, min(per_split, len(available))))
    if not samples:
        print("No valid images available for visualization")
        return
    figure, axes = plt.subplots(len(samples), 1, figsize=(10, 5 * len(samples)), squeeze=False)
    for ax, record in zip(axes[:, 0], samples):
        with Image.open(record["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for cls, x, y, w, h in record["boxes"]:
            left, top = (x - w / 2) * width, (y - h / 2) * height
            draw.rectangle((left, top, (x + w / 2) * width, (y + h / 2) * height), outline="red", width=3)
            draw.text((left, max(0, top - 12)), report["names"][cls], fill="red")
        ax.imshow(image)
        ax.set_title(f"{record['split']}: {Path(record['image']).name} ({len(record['boxes'])} boxes)")
        ax.axis("off")
    figure.tight_layout()
    plt.show()
