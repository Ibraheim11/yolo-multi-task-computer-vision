"""Synthetic read-only provenance regression checks; no network or model imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.coco_provenance import verify_coco, ring_match
from src.label_diagnostics import diagnose_labels, inventory
from src.polygon_review import _items

ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.base = Path(self.temp.name)
        self.root, self.review, self.coco = [self.base / p for p in ("original", "review", "coco")]
        for folder in (self.root / "train/images", self.root / "train/labels", self.review, self.coco):
            folder.mkdir(parents=True)
        (self.root / "data.yaml").write_text("train: train/images\nnames: [fire]\n")
        Image.new("RGB", (100, 100), "red").save(self.root / "train/images/original.png")
        (self.root / "train/labels/original.txt").write_text("0 0.1 0.1 0.8 0.1 0.2 0.8 0.1 0.1\n")
        packet = {"dataset": str(self.root), "items": _items(self.root, diagnose_labels(self.root))}
        (self.review / "review.json").write_text(json.dumps(packet))
        (self.review / "decisions.json").write_text('{"items": [{"approve": false}]}')
        Image.open(self.root / "train/images/original.png").save(self.coco / "renamed.png")
        self.data = {"images": [{"id": 91, "file_name": "renamed.png", "width": 100, "height": 100}],
                     "categories": [{"id": 42, "name": "fire"}],
                     "annotations": [{"id": 900, "image_id": 91, "category_id": 42, "iscrowd": 0,
                                      "segmentation": [[10, 10, 80, 10, 20, 80]], "bbox": [10, 10, 70, 70]}]}

    def tearDown(self):
        self.temp.cleanup()

    def run_report(self):
        (self.coco / "_annotations.coco.json").write_text(json.dumps(self.data))
        before = [inventory(p) for p in (self.root, self.review, self.coco)]
        report = verify_coco(self.root, self.review, self.coco, self.base / "report.json")
        self.assertEqual(before, [inventory(p) for p in (self.root, self.review, self.coco)])
        self.assertFalse(report["conversion_authorized"])
        self.assertFalse(report["training_enabled"])
        return report["items"][0]

    def test_renamed_image_and_different_category_id(self):
        row = self.run_report()
        self.assertEqual(row["status"], "export_polygon_correspondence")
        self.assertEqual(row["nine_token_investigation"]["unique_vertices"], 3)
        self.assertTrue(row["nine_token_investigation"]["repeated_closing_vertex"])

    def test_ring_winding_start_closure_and_tolerance(self):
        pts = [[.1, .1], [.8, .1], [.2, .8], [.1, .1]]
        self.assertTrue(ring_match(pts, [20, 80, 80, 10, 10, 10, 20, 80], 100, 100, .05))
        self.assertFalse(ring_match(pts, [20, 80, 80, 10, 10.1, 10], 100, 100, .05))

    def test_bbox_only_is_inconclusive(self):
        self.data["annotations"][0]["segmentation"] = []
        self.assertEqual(self.run_report()["status"], "inconclusive_geometry_or_category")

    def test_wrong_category(self):
        self.data["categories"][0]["name"] = "smoke"
        self.assertEqual(self.run_report()["status"], "inconclusive_geometry_or_category")

    def test_bbox_conflict(self):
        self.data["annotations"][0]["bbox"] = [0, 0, 100, 100]
        self.assertEqual(self.run_report()["status"], "inconclusive_geometry_or_category")

    def test_multipart_rle_and_missing_segmentation(self):
        for seg in ([[10, 10, 80, 10, 20, 80]] * 2, {"counts": "fixture", "size": [100, 100]}, None):
            with self.subTest(seg=seg):
                self.data["annotations"][0]["segmentation"] = seg
                self.assertEqual(self.run_report()["status"], "inconclusive_geometry_or_category")
                (self.base / "report.json").unlink()

    def test_filename_is_not_identity(self):
        Image.new("RGB", (100, 100), "blue").save(self.coco / "renamed.png")
        self.assertEqual(self.run_report()["status"], "inconclusive_image_identity")

    def test_duplicate_content_identity_is_ambiguous(self):
        self.data["images"].append(dict(self.data["images"][0], id=92))
        self.assertEqual(self.run_report()["status"], "inconclusive_image_identity")

    def test_duplicate_annotation_geometry_is_ambiguous(self):
        self.data["annotations"].append(dict(self.data["annotations"][0], id=901))
        self.assertEqual(self.run_report()["status"], "inconclusive_geometry_or_category")

    def test_missing_export_is_unavailable(self):
        report = verify_coco(self.root, self.review, self.coco, self.base / "report.json")
        self.assertEqual(report["items"][0]["status"], "unavailable")

    def test_changed_review_and_unsafe_output_blocked(self):
        with self.assertRaises(ValueError):
            verify_coco(self.root, self.review, self.coco, self.review / "new.json")
        (self.root / "train/labels/original.txt").write_text("0 0.5 0.5 0.2 0.2")
        with self.assertRaises(ValueError):
            verify_coco(self.root, self.review, self.coco, self.base / "report.json")

    def test_dimensions_ids_and_traversal_fail_closed(self):
        for change in ("dimensions", "ids", "traversal"):
            with self.subTest(change=change):
                original = json.loads(json.dumps(self.data))
                if change == "dimensions":
                    self.data["images"][0]["width"] = 200
                elif change == "ids":
                    self.data["annotations"][0]["image_id"] = 999
                else:
                    self.data["images"][0]["file_name"] = "../original/train/images/original.png"
                with self.assertRaises(ValueError):
                    self.run_report()
                self.assertFalse((self.base / "report.json").exists())
                self.data = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
