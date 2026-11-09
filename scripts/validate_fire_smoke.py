"""Offline Phase 2A checks. Synthetic fixtures test audits, never claim dataset statistics."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs" / ".matplotlib")
os.environ["MPLBACKEND"] = "Agg"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "outputs" / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
from PIL import Image
import yaml
from src.fire_smoke import audit_dataset, locate_yaml, parse_label, show_samples, write_training_yaml


class AuditChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit-fixture-", dir=ROOT / "outputs")
        self.root = Path(self.temp.name)
        assert self.root.resolve().is_relative_to((ROOT / "outputs").resolve())
        for split, color in (("train", "red"), ("valid", "green"), ("test", "blue")):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir()
            Image.new("RGB", (20, 20), color).save(self.root / split / "images" / "sample.png")
            (self.root / split / "labels" / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        self.config = {"train": "../train/images", "val": "../valid/images", "test": "../test/images", "names": ["fixture-object"], "nc": 1}
        self.save_config()

    def tearDown(self):
        self.temp.cleanup()

    def save_config(self):
        (self.root / "data.yaml").write_text(yaml.safe_dump(self.config))

    def test_clean_export_and_absolute_training_yaml(self):
        report = audit_dataset(self.root)
        self.assertTrue(report["ready"])
        self.assertEqual(report["splits"]["train"]["images"], 1)
        destination = write_training_yaml(report, self.root / "audited.yaml")
        canonical = yaml.safe_load(destination.read_text())
        self.assertTrue(Path(canonical["val"]).is_absolute())
        self.assertNotIn("download", canonical)
        # Exercise plotting with actual fixture annotations, never save a claimed result.
        import matplotlib.pyplot as plt
        with patch.object(plt, "show"):
            show_samples(report, per_split=1)
        plt.close("all")

    def test_invalid_box_rows(self):
        label = self.root / "train" / "labels" / "sample.txt"
        for row in ("0 0.5 0.5 0.4", "2 0.5 0.5 0.4 0.4", "0 nan 0.5 0.4 0.4", "0 0.1 0.5 0.8 0.4", "0 0.5 0.5 -0.2 0.4", "0.5 0.5 0.5 0.4 0.4"):
            label.write_text(row)
            self.assertTrue(parse_label(label, ["fixture-object"])[1], row)
        label.write_text("0 0.5 0.5 0.4 0.4\n" * 2)
        self.assertTrue(parse_label(label, ["fixture-object"])[1])
        self.assertFalse(audit_dataset(self.root)["ready"])

    def test_missing_label_corrupt_image_and_orphan(self):
        (self.root / "train" / "labels" / "sample.txt").unlink()
        (self.root / "valid" / "images" / "sample.png").write_bytes(b"not an image")
        (self.root / "test" / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2")
        report = audit_dataset(self.root)
        self.assertFalse(report["ready"])
        self.assertEqual(report["splits"]["train"]["missing_labels"], 1)
        self.assertEqual(report["splits"]["val"]["corrupt_images"], 1)
        self.assertEqual(report["splits"]["test"]["orphan_labels"], 1)
        with self.assertRaises(ValueError):
            write_training_yaml(report, self.root / "blocked.yaml")

    def test_empty_background_and_leakage(self):
        Image.new("RGB", (20, 20), "white").save(self.root / "train" / "images" / "background.png")
        (self.root / "train" / "labels" / "background.txt").write_text("")
        self.assertEqual(audit_dataset(self.root)["splits"]["train"]["empty_labels"], 1)
        source = self.root / "train" / "images" / "sample.png"
        (self.root / "test" / "images" / "sample.png").write_bytes(source.read_bytes())
        report = audit_dataset(self.root)
        self.assertFalse(report["ready"])
        self.assertTrue(any("duplicate across splits" in i["message"] for i in report["issues"]))

    def test_missing_split_and_yaml_ambiguity(self):
        self.config.pop("test")
        self.save_config()
        self.assertFalse(audit_dataset(self.root)["ready"])
        (self.root / "train" / "data.yaml").write_text("names: [x]")
        with self.assertRaises(ValueError):
            locate_yaml(self.root)

    def test_content_changes_are_recorded(self):
        before = audit_dataset(self.root)
        Image.new("RGB", (20, 20), "yellow").save(self.root / "train" / "images" / "sample.png")
        self.assertNotEqual(before, audit_dataset(self.root))


def smoke_notebook():
    """Execute default local cells with network/model/training calls forbidden."""
    from ultralytics import YOLO
    import requests
    nb_path = ROOT / "notebooks" / "01_fire_smoke_detection.ipynb"
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse(source)
    flags = {"INSTALL_DEPENDENCIES", "DOWNLOAD_DATASET", "RUN_TRAINING", "RUN_VALIDATION", "RUN_VIDEO_INFERENCE", "ALLOW_LOCAL_TRAINING"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in flags:
                    assert isinstance(node.value, ast.Constant) and node.value.value is False, target.id
                    seen.add(target.id)
    assert seen == flags
    # Every model construction/train/val/predict call must be nested under its explicit action flag.
    def visit(node, guards=()):
        if isinstance(node, ast.If):
            positive = (node.test.id,) if isinstance(node.test, ast.Name) else ()
            for child in node.body:
                visit(child, guards + positive)
            for child in node.orelse:
                visit(child, guards)
            return
        if isinstance(node, ast.Call):
            is_model = isinstance(node.func, ast.Name) and node.func.id == "YOLO"
            is_action = isinstance(node.func, ast.Attribute) and node.func.attr in {"train", "val", "predict"}
            if is_model or is_action:
                assert set(guards) & {"RUN_TRAINING", "RUN_VALIDATION", "RUN_VIDEO_INFERENCE"}, ast.unparse(node)
        for child in ast.iter_child_nodes(node):
            visit(child, guards)
    visit(tree)
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output), patch.object(YOLO, "__init__", side_effect=AssertionError("Model loading forbidden")), patch.object(YOLO, "train", side_effect=AssertionError("Training forbidden")), patch.object(YOLO, "val", side_effect=AssertionError("Validation forbidden")), patch.object(YOLO, "predict", side_effect=AssertionError("Inference forbidden")), patch.object(requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden")):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"{nb_path.name}:cell-{index}", "exec"), namespace)
    assert namespace["TRAINING_CONFIG"]["epochs"] == 50
    assert namespace["TRAINING_CONFIG"]["imgsz"] == 640
    assert namespace["MODEL_ID"] == "yolo26n.pt"
    assert namespace["TRAINING_RUN_DIR"] is None
    assert namespace["VIDEO_OUTPUT"] is None
    from ultralytics.cfg import get_cfg
    config = get_cfg(overrides=namespace["TRAINING_CONFIG"])
    assert config.epochs == 50 and config.imgsz == 640 and config.patience == 0
    assert namespace["WORKSPACE"] == "gfo-fc1de"
    assert namespace["PROJECT"] == "fire-and-smoke-detector-yykya"
    assert namespace["DATASET_VERSION"] == 14
    # Verify SDK print/error redaction using an explicitly synthetic token and no network.
    sentinel = "TEST_ONLY_SECRET_SENTINEL"
    def forbidden_sdk(*args, **kwargs):
        print(sentinel)
        raise RuntimeError(sentinel)
    original_sdk = namespace["Roboflow"]
    captured = io.StringIO()
    namespace["Roboflow"] = forbidden_sdk
    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            with patch.dict(os.environ, {"ROBOFLOW_API_KEY": sentinel}), contextlib.redirect_stdout(captured):
                try:
                    namespace["download_roboflow_securely"](Path(temp) / "not-created")
                except RuntimeError as exc:
                    formatted = "".join(traceback.format_exception(exc))
                    assert sentinel not in formatted
                else:
                    raise AssertionError("Expected redacted SDK error")
        assert sentinel not in captured.getvalue()
    finally:
        namespace["Roboflow"] = original_sdk
    # Static credential check: API key must be a runtime variable, never a literal.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "api_key":
                    assert isinstance(keyword.value, ast.Name) and keyword.value.id == "key"
    print("PASS: secure runtime key wiring and SDK output/exception redaction")
    baseline = ROOT / "outputs" / "phase2a" / "preserved_notebooks.json"
    if baseline.is_file():
        for name, digest in json.loads(baseline.read_text()).items():
            assert hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest() == digest, name
    print("PASS: default notebook execution; network/model/training/inference forbidden; 50 epochs / 640 config; other notebooks preserved")


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    os.chdir(ROOT)
    smoke_notebook()
