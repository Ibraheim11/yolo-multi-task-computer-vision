"""Offline synthetic regression tests: never use real data, model weights, or training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import yaml
from src.fire_smoke import audit_dataset, parse_label, sha256, write_training_yaml
from src.label_diagnostics import classify_row, create_reviewed_copy, diagnose_labels, inventory, write_diagnosis

ROOT = Path(__file__).resolve().parents[1]
BOX = "0 0.5 0.5 0.4 0.4"
POLYGON = "0 0.1 0.1 0.4 0.1 0.2 0.4"


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="diagnostic-fixture-", dir=ROOT / "outputs")
        self.base = Path(self.temp.name)
        self.root = self.base / "original"
        self.root.mkdir()
        for split, color in (("train", "red"), ("valid", "green"), ("test", "blue")):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir()
            Image.new("RGB", (16, 16), color).save(self.root / split / "images" / "sample.png")
            (self.root / split / "labels" / "sample.txt").write_text(BOX + "\n")
        (self.root / "data.yaml").write_text(yaml.safe_dump({"train": "train/images", "val": "valid/images", "test": "test/images", "names": ["fixture"]}))
        self.label = self.root / "train" / "labels" / "sample.txt"

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, replacement="0 0.25 0.25 0.3 0.3", reason="Fixture-only human review"):
        content = {"files": [{"path": "train/labels/sample.txt", "original_sha256": sha256(self.label),
                    "replacements": [{"line": 2, "original": POLYGON, "replacement": replacement, "reason": reason}]}]}
        path = self.base / "plan.json"
        path.write_text(json.dumps(content))
        return path

    def test_clean_detection_and_unusual_whitespace(self):
        self.label.write_bytes(b"\xef\xbb\xbf  0\t0.5   0.5\t0.4  0.4  \r\n\r\n")
        self.assertFalse(parse_label(self.label, ["fixture"])[1])
        self.assertEqual(diagnose_labels(self.root)["invalid_annotation_file_count"], 0)
        self.label.write_text("0\u00a00.5\t0.5 0.4 0.4\n", encoding="utf-8")
        self.assertFalse(parse_label(self.label, ["fixture"])[1])

    def test_malformed_lines_and_exact_reporting(self):
        self.label.write_text(BOX + "\n0 0.5 0.5\n0 nan 0.5 0.4 0.4\n")
        diagnosis = diagnose_labels(self.root)
        self.assertEqual(diagnosis["current_audit_issue_count"], 2)
        self.assertEqual(diagnosis["diagnostic_finding_count"], 2)
        self.assertEqual(diagnosis["invalid_annotation_file_count"], 1)
        findings = diagnosis["files"][0]["findings"]
        self.assertEqual([f["line"] for f in findings], [2, 3])
        self.assertEqual(findings[0]["raw"], "0 0.5 0.5")
        before = inventory(self.root)
        write_diagnosis(diagnosis, self.base / "diagnosis")
        self.assertEqual(before, inventory(self.root))
        self.assertIn("line 2", (self.base / "diagnosis/diagnosis.txt").read_text())

    def test_segmentation_obb_pose_candidates_are_not_valid_detection(self):
        classification = classify_row(POLYGON, ["fixture"])
        self.assertIn("segmentation_polygon_like", classification["candidates"])
        obb = "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8"
        candidates = classify_row(obb, ["fixture"])["candidates"]
        self.assertIn("segmentation_polygon_like", candidates)
        self.assertIn("four_corner_obb_like", candidates)
        pose = "0 0.5 0.5 0.4 0.4 0.2 0.2 2"
        self.assertIn("pose_like_matching_declared_kpt_shape", classify_row(pose, ["fixture"], [1, 3])["candidates"])
        self.label.write_text(BOX + "\n" + POLYGON + "\n")
        self.assertFalse(audit_dataset(self.root)["ready"])
        with self.assertRaises(ValueError):
            write_training_yaml(audit_dataset(self.root), self.base / "blocked.yaml")

    def test_report_only_does_not_invent_raw_rows(self):
        self.label.write_text(BOX + "\n" + POLYGON)
        path = self.base / "audit.json"
        report = audit_dataset(self.root)
        path.write_text(json.dumps(report))
        diagnosis = diagnose_labels(audit_report=path)
        self.assertEqual(diagnosis["current_audit_issue_count"], len(report["issues"]))
        self.assertIsNone(diagnosis["files"][0]["findings"][0]["raw"])
        self.assertIn("cannot_determine", diagnosis["files"][0]["findings"][0]["interpretation"])

    def test_saved_paths_rebase_and_changed_rows_remain_visible(self):
        self.label.write_text(BOX + "\n" + POLYGON)
        report = audit_dataset(self.root)
        old = str(self.root)
        report["source_yaml"] = report["source_yaml"].replace(old, "/content/previous/export").replace("\\", "/")
        for issue in report["issues"]:
            issue["path"] = issue["path"].replace(old, "/content/previous/export").replace("\\", "/")
        saved = self.base / "saved.json"
        saved.write_text(json.dumps(report))
        result = diagnose_labels(self.root, saved)
        item = result["saved_issue_reconciliation"][0]
        self.assertEqual(item["status"], "still_invalid")
        self.assertEqual(item["field_count"], 7)
        self.assertEqual(item["raw"], POLYGON)
        self.assertIn("tokens=7", write_diagnosis(result, self.base / "diagnosis"))
        self.label.write_text(BOX + "\n0 0.2 0.2 0.1 0.1")
        self.assertEqual(diagnose_labels(self.root, saved)["saved_issue_reconciliation"][0]["status"],
                         "no_longer_rejected_requires_review")
        self.label.rename(self.label.with_suffix(".preserved"))
        self.assertEqual(diagnose_labels(self.root, saved)["saved_issue_reconciliation"][0]["status"], "unresolved")

    def test_saved_path_traversal_is_not_read(self):
        saved = self.base / "saved.json"
        saved.write_text(json.dumps({"source_yaml": "/old/data.yaml", "issues": [
            {"path": "/old/../labels/private.txt", "message": "line 1: bad", "split": "train"}]}))
        item = diagnose_labels(self.root, saved)["saved_issue_reconciliation"][0]
        self.assertEqual(item["status"], "unresolved")
        self.assertIsNone(item["raw"])

    def test_polygon_candidates_do_not_make_malformed_rows_recoverable(self):
        for row in ("0 0 0 0.1 0.1 0.2 0.2", "0 0 0 1 1 0 1 1 0",
                    "0 nan 0 1 0 0 1", "0 0 0 2 0 0 1", "2 0 0 1 0 0 1"):
            self.assertNotIn("segmentation_polygon_like", classify_row(row, ["fixture"])["candidates"])

    def test_copy_with_unrelated_invalid_file_still_blocks_training(self):
        self.label.write_text(BOX + "\n" + POLYGON)
        (self.root / "test/labels/sample.txt").write_text("0 0.5")
        before = inventory(self.root)
        dest = self.base / "incomplete"
        manifest = create_reviewed_copy(self.root, dest, self.plan())
        self.assertFalse(manifest["ready"])
        self.assertFalse(json.loads((dest / "repair_audit.json").read_text())["ready"])
        self.assertEqual(before, inventory(self.root))
        with self.assertRaises(ValueError):
            write_training_yaml(audit_dataset(dest), self.base / "blocked.yaml")

    def test_reviewed_copy_preserves_all_original_bytes_and_images(self):
        self.label.write_bytes(b"\xef\xbb\xbf" + (BOX + "\r\n" + POLYGON + "\r\n").encode())
        before = inventory(self.root)
        dest = self.base / "reviewed"
        manifest = create_reviewed_copy(self.root, dest, self.plan())
        self.assertTrue(manifest["ready"])
        self.assertEqual(before, inventory(self.root))
        copied = (dest / "train/labels/sample.txt").read_bytes()
        self.assertTrue(copied.startswith(b"\xef\xbb\xbf" + BOX.encode() + b"\r\n"))
        self.assertEqual(len(parse_label(dest / "train/labels/sample.txt", ["fixture"])[0]), 2)
        self.assertTrue((dest / "repair_manifest.json").is_file())
        self.assertTrue(audit_dataset(dest)["ready"])

    def test_ambiguous_and_invalid_repairs_are_rejected(self):
        self.label.write_text(BOX + "\n" + POLYGON)
        before = inventory(self.root)
        for replacement, reason in (("", "review"), (POLYGON, "review"), ("0 0.5 0.5 -0.2 0.2", "review"), (BOX, "")):
            with self.assertRaises(ValueError):
                create_reviewed_copy(self.root, self.base / "rejected", self.plan(replacement, reason))
            self.assertFalse((self.base / "rejected").exists())
        plan_path = self.plan()
        plan = json.loads(plan_path.read_text())
        plan["files"][0]["original_sha256"] = "stale"
        plan_path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            create_reviewed_copy(self.root, self.base / "stale", plan_path)
        self.assertEqual(before, inventory(self.root))
        with self.assertRaises(ValueError):
            create_reviewed_copy(self.root, self.root / "nested", self.plan())

    def test_training_guard_rejects_failed_audit_before_any_model_call(self):
        notebook = json.loads((ROOT / "notebooks/01_fire_smoke_detection.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in notebook["cells"] if "training" in c.get("metadata", {}).get("tags", []))
        def forbidden(*args, **kwargs):
            raise AssertionError("Model loading is forbidden")
        scope = {"DATASET_REPORT": {"ready": False}, "DATA_YAML": None, "YOLO": forbidden}
        # Exercise only the rejection path, with no model library imported.
        guarded_test_cell = cell.replace("RUN_TRAINING = False", "RUN_TRAINING = True", 1)
        with self.assertRaisesRegex(RuntimeError, "clean dataset audit"):
            exec(compile(guarded_test_cell, "training-gate-negative-test", "exec"), scope)


if __name__ == "__main__":
    unittest.main(verbosity=2)
