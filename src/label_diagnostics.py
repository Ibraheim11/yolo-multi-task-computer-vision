"""Read-only label diagnosis and explicit reviewed-copy repairs. No model imports."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil

import yaml
from src.fire_smoke import audit_dataset, class_names, locate_yaml, parse_label, resolve_split, sha256


def classify_row(raw, names, kpt_shape=None):
    """Describe structural candidates, never infer intended task from token count alone."""
    parts = raw.split()
    result = {"field_count": len(parts), "candidates": [], "interpretation": "malformed_or_unknown"}
    if not parts:
        result["interpretation"] = "blank_background_line"
        return result
    try:
        cls = int(parts[0])
        values = [float(v) for v in parts[1:]]
    except ValueError:
        result["detail"] = "Noninteger class ID or nonnumeric coordinate/token"
        return result
    if cls not in range(len(names)) or not all(math.isfinite(v) for v in values):
        result["detail"] = "Out-of-range class ID or nonfinite coordinates"
        return result
    if len(parts) == 5:
        result["interpretation"] = "detection_shaped"
        result["detail"] = "Five fields; use strict detection checks for bounds, sizes and duplicates"
        return result
    normalized = all(0 <= v <= 1 for v in values)
    if len(values) >= 6 and len(values) % 2 == 0 and normalized:
        points = list(zip(values[::2], values[1::2]))
        area = abs(sum(x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
                       for i, (x, y) in enumerate(points))) / 2
        result["polygon_points"] = len(points)
        result["polygon_signed_area_magnitude"] = area
        if len(set(points)) >= 3 and area > 1e-12:
            result["candidates"].append("segmentation_polygon_like")
            if len(values) == 8:
                result["candidates"].append("four_corner_obb_like")
        else:
            result["detail"] = "Coordinate pairs form a degenerate polygon; do not convert automatically"
    if (isinstance(kpt_shape, list) and len(kpt_shape) == 2
            and all(isinstance(v, int) and v > 0 for v in kpt_shape)
            and kpt_shape[1] in (2, 3) and len(parts) == 5 + kpt_shape[0] * kpt_shape[1]):
        result["candidates"].append("pose_like_matching_declared_kpt_shape")
    elif len(parts) > 5 and ((len(parts) - 5) % 2 == 0 or (len(parts) - 5) % 3 == 0):
        result["candidates"].append("possible_pose_or_extra_fields_unconfirmed")
    if result["candidates"]:
        result["interpretation"] = "other_yolo_format_candidate_requires_review"
    result["detail"] = result.get("detail", "Not a five-field detection row. Confirm the source task/export and inspect its image before any conversion.")
    return result


def reconcile_saved_issues(original, root, names, config):
    """Rebase exact report-relative paths, never search by an ambiguous basename."""
    saved_base = PurePosixPath(original.get("source_yaml", "").replace("\\", "/")).parent
    current_base = locate_yaml(root).parent
    results = []
    for issue in original["issues"]:
        item = {"saved_issue": issue, "status": "unresolved", "raw": None, "field_count": None}
        try:
            saved = PurePosixPath(issue.get("path", "").replace("\\", "/"))
            relative = saved.relative_to(saved_base)
            if ".." in relative.parts or "labels" not in relative.parts or relative.suffix != ".txt":
                raise ValueError("Not an in-export annotation path")
            path = (current_base / Path(*relative.parts)).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Resolved path escapes the selected export")
            item["path"] = str(path)
            match = re.match(r"line (\d+): (.*)", issue.get("message", ""))
            if not match:
                raise ValueError("Saved issue has no annotation line number")
            number = int(match.group(1))
            item.update(line=number, saved_rejection_reason=match.group(2))
            rows = path.read_text(encoding="utf-8-sig").splitlines()
            if not 1 <= number <= len(rows):
                raise ValueError("Saved line no longer exists")
            raw = rows[number - 1]
            _, errors = parse_label(path, names)
            reasons = [e for e in errors if e.startswith(f"line {number}:")]
            item.update(raw=raw, sha256=sha256(path), current_rejections=reasons,
                        status="still_invalid" if reasons else "no_longer_rejected_requires_review")
            item.update(classify_row(raw, names, config.get("kpt_shape")))
        except (ValueError, OSError, UnicodeError) as exc:
            item["resolution_error"] = str(exc)
        results.append(item)
    return results


def diagnose_labels(dataset_dir=None, audit_report=None):
    """Accept a real export, a saved report, or both; report-only cannot recover raw rows."""
    if dataset_dir is None and audit_report is None:
        raise ValueError("Provide a dataset directory or audit report")
    original = json.loads(Path(audit_report).read_text(encoding="utf-8")) if audit_report else None
    if audit_report and (not isinstance(original, dict) or not isinstance(original.get("issues"), list)):
        raise ValueError("Expected this project's audit JSON with an issues list")
    root = Path(dataset_dir).resolve() if dataset_dir else None
    fresh = audit_dataset(root) if root else None
    selected = fresh if fresh is not None else original
    if not isinstance(selected, dict) or not isinstance(selected.get("issues"), list):
        raise ValueError("Expected this project's audit JSON with an issues list")
    result = {"mode": "dataset_scan" if root else "report_only", "dataset": str(root) if root else None,
              "original_audit_issue_count": len(original["issues"]) if original is not None else None,
              "current_audit_issue_count": len(selected["issues"]),
              "original_audit_issues": original["issues"] if original is not None else None,
              "current_audit_issues": selected["issues"], "files": [], "split_issues": [],
              "audit_ready": selected.get("ready", False),
              "note": "Format candidates are structural evidence only; no conversion is automatic."}
    if root:
        yaml_path = locate_yaml(root)
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = class_names(config)
        result["names"] = names
        result["saved_issue_reconciliation"] = reconcile_saved_issues(original, root, names, config) if original else []
        seen = set()
        for split in ("train", "val", "test"):
            try:
                images = resolve_split(yaml_path, config.get(split), root)
                if images.name != "images":
                    raise ValueError("Expected images/ directory")
                labels_dir = images.parent / "labels"
            except (ValueError, OSError) as exc:
                result["split_issues"].append({"split": split, "message": str(exc)})
                continue
            for path in sorted(labels_dir.rglob("*.txt")):
                real_path = path.resolve()
                if not real_path.is_relative_to(root):
                    result["split_issues"].append({"split": split, "path": str(path), "message": "External symlink not read"})
                    continue
                identity = (split, str(path))
                if identity in seen:
                    continue
                seen.add(identity)
                _, errors = parse_label(path, names)
                if not errors:
                    continue
                try:
                    rows = path.read_text(encoding="utf-8-sig").splitlines()
                    digest = sha256(path)
                except (OSError, UnicodeError):
                    rows, digest = [], None
                findings = []
                for error in errors:
                    match = re.match(r"line (\d+): (.*)", error)
                    line = int(match.group(1)) if match else None
                    raw = rows[line - 1] if line is not None and line <= len(rows) else None
                    finding = {"line": line, "raw": raw, "root_cause": match.group(2) if match else error}
                    if raw is not None:
                        finding.update(classify_row(raw, names, config.get("kpt_shape")))
                    else:
                        finding["interpretation"] = "unreadable_annotation"
                    findings.append(finding)
                result["files"].append({"split": split, "path": str(path), "relative_path": path.relative_to(root).as_posix(),
                                        "sha256": digest, "issue_count": len(findings), "findings": findings})
    else:
        grouped = {}
        for issue in selected["issues"]:
            key = (issue.get("split"), issue.get("path"))
            grouped.setdefault(key, []).append(issue)
        for (split, path), issues in grouped.items():
            findings = []
            for issue in issues:
                message = issue.get("message", "")
                match = re.match(r"line (\d+): (.*)", message)
                findings.append({"line": int(match.group(1)) if match else None, "raw": None,
                                 "root_cause": match.group(2) if match else message,
                                 "interpretation": "raw_row_unavailable_cannot_determine_format"})
            result["files"].append({"split": split, "path": path, "issue_count": len(findings), "findings": findings})
    result["affected_file_count"] = len(result["files"])
    result["diagnostic_finding_count"] = sum(f["issue_count"] for f in result["files"])
    result["invalid_annotation_file_count"] = (len(result["files"]) if root else None)
    result["finding_counts_by_split"] = dict(Counter(f["split"] for f in result["files"] for _ in f["findings"]))
    return result


def write_diagnosis(result, output_dir):
    """Write full JSON/CSV/text; do not truncate affected files or annotation rows."""
    output_dir = Path(output_dir)
    if result.get("dataset") and output_dir.resolve().is_relative_to(Path(result["dataset"]).resolve()):
        raise ValueError("Diagnostic output must be outside the original export")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "invalid_lines.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "path", "line", "field_count", "root_cause", "interpretation", "candidates", "raw"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for file in result["files"]:
            for finding in file["findings"]:
                writer.writerow({"split": file["split"], "path": file["path"], **{key: finding.get(key) for key in fields[2:]}})
    text = [f"Original report issues: {result['original_audit_issue_count']}",
            f"Current audit issues: {result['current_audit_issue_count']}",
            f"Affected files in diagnosis: {result['affected_file_count']}",
            f"Diagnostic findings: {result['diagnostic_finding_count']}",
            "These counts differ: audit issues can include non-label problems or multiple lines per file."]
    for file in result["files"]:
        text.append(f"\n{file['split']} | {file['path']} | {file['issue_count']} issue(s)")
        for row in file["findings"]:
            text.append(f"  line {row['line']}: tokens={row.get('field_count')} | {row['root_cause']} | {row['interpretation']} | candidates={row.get('candidates', [])}")
            text.append(f"    raw={row['raw']!r}")
    if result["original_audit_issues"] is not None:
        text.append("\nSaved issue reconciliation against the selected export:")
        for item in result.get("saved_issue_reconciliation", []):
            text.append(f"  {item.get('path', item['saved_issue'].get('path'))} | line {item.get('line')} | {item['status']} | tokens={item['field_count']}")
            text.append(f"    raw={item['raw']!r} | saved reason={item.get('saved_rejection_reason')} | current={item.get('current_rejections')} | error={item.get('resolution_error')}")
        text.append("\nAll original saved audit issues (may differ from the current dataset):")
        for issue in result["original_audit_issues"]:
            text.append(f"  {issue.get('split')} | {issue.get('path')} | {issue.get('message')}")
    text.append("\nAll current audit issues (including non-label issues):")
    for issue in result["current_audit_issues"]:
        text.append(f"  {issue.get('split')} | {issue.get('path')} | {issue.get('message')}")
    (output_dir / "diagnosis.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    return "\n".join(text)


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not supported for reviewed dataset copies")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def create_reviewed_copy(dataset_dir, destination, plan_path):
    """Apply ONLY explicit line replacements in a new copy; never infer or delete rows.

    Plan: {files: [{path, original_sha256, replacements: [{line, original,
    replacement, reason}]}]}. Every replacement must be strict detection format.
    Existing valid rows cannot be changed. Human-reviewed ambiguous rows are
    accepted only through this explicit plan, never automatically converted.
    """
    source, destination = Path(dataset_dir).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Use a new destination outside the original export and its ancestors")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    entries = plan.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Plan must explicitly list reviewed file changes")
    source_yaml = locate_yaml(source)
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = class_names(config)
    original_inventory = inventory(source)
    reserved = {"repair_manifest.json", "repair_audit.json"}
    if reserved & set(original_inventory):
        raise ValueError("Source contains reserved repair report filenames")
    edits, changes = {}, []
    for entry in entries:
        relative = Path(entry["path"])
        path = (source / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(source) or path.suffix != ".txt" or "labels" not in path.parts:
            raise ValueError("Only in-export annotation files may be edited")
        key = path.relative_to(source).as_posix()
        if key in edits or sha256(path) != entry.get("original_sha256"):
            raise ValueError("Repeated file or original hash mismatch; regenerate the diagnosis")
        original_bytes = path.read_bytes()
        original_text = original_bytes.decode("utf-8-sig")
        rows = original_text.splitlines()
        full_rows = original_text.splitlines(keepends=True)
        _, errors = parse_label(path, names)
        invalid_lines = {int(m.group(1)) for error in errors if (m := re.match(r"line (\d+):", error))}
        used = set()
        replacements = entry.get("replacements", [])
        if not replacements:
            raise ValueError("Every plan file requires explicit replacements")
        for replacement in replacements:
            number = replacement["line"]
            if not isinstance(number, int) or number not in invalid_lines or number in used:
                raise ValueError("Only diagnosed invalid lines may be replaced, once per line")
            if rows[number - 1] != replacement.get("original") or not str(replacement.get("reason", "")).strip():
                raise ValueError("Original row and human review reason are required")
            value = replacement.get("replacement")
            if not isinstance(value, str) or len(value.split()) != 5 or "\n" in value or "\r" in value:
                raise ValueError("Replacement must be exactly one nonempty detection row")
            used.add(number)
            ending = full_rows[number - 1][len(rows[number - 1]):]
            full_rows[number - 1] = value + ending
        prefix = b"\xef\xbb\xbf" if original_bytes.startswith(b"\xef\xbb\xbf") else b""
        edits[key] = prefix + "".join(full_rows).encode("utf-8")
        changes.append(entry)
    # Preflight replacements before copying anything, using a sibling temporary file.
    import tempfile
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="label-plan-validation-", dir=destination.parent) as temporary:
        for key, content in edits.items():
            check_path = Path(temporary) / "label.txt"
            check_path.write_bytes(content)
            if parse_label(check_path, names)[1]:
                raise ValueError(f"Planned file still fails strict detection validation: {key}")
    # No overwrite and no source deletion; source inventory is checked before/after.
    if inventory(source) != original_inventory:
        raise ValueError("Source changed during preflight")
    split_paths = {split: resolve_split(source_yaml, config.get(split), source) for split in ("train", "val", "test")}
    shutil.copytree(source, destination)
    for key, content in edits.items():
        (destination / key).write_bytes(content)
    cleaned_yaml = {split: str(destination / path.relative_to(source)) for split, path in split_paths.items()}
    cleaned_yaml.update(names=names, nc=len(names))
    target_yaml = destination / source_yaml.relative_to(source)
    target_yaml.write_text(yaml.safe_dump(cleaned_yaml, sort_keys=False), encoding="utf-8")
    if inventory(source) != original_inventory:
        raise RuntimeError("Original changed during copy; stop and inspect both directories")
    copied_inventory = inventory(destination)
    permitted = set(edits) | {source_yaml.relative_to(source).as_posix()}
    if set(copied_inventory) != set(original_inventory) or any(copied_inventory[k] != v for k, v in original_inventory.items() if k not in permitted):
        raise RuntimeError("Unexpected copy differences; do not use the copy")
    audit = audit_dataset(destination)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "original": str(source),
                "destination": str(destination), "changes": changes, "original_sha256_inventory": original_inventory,
                "copy_sha256_inventory": copied_inventory, "yaml_change": "Rebased split paths; retained class map; removed optional download/metadata hooks",
                "ready": audit["ready"], "note": "Only the copy was edited; no image or nonblank annotation row was removed."}
    (destination / "repair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "repair_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return manifest
