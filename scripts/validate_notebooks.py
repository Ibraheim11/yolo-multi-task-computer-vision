"""Validate published notebooks offline without downloading data or executing code."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "01_fire_smoke_detection.ipynb",
    "02_object_counting.ipynb",
    "03_visioneye_mapping.ipynb",
    "04_neural_ocean_detection.ipynb",
)


def syntax_for_notebook(source: str) -> str:
    """Mask standalone Colab shell/magic lines before standard Python parsing.

    This is a basic syntax check, not IPython execution or semantic validation.
    """
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def validate_notebook(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4, path
    assert data.get("nbformat_minor", 0) >= 5, path
    assert isinstance(data.get("cells"), list), path
    cells = data["cells"]
    ids = [cell.get("id") for cell in cells]
    assert all(isinstance(value, str) and value for value in ids), f"Missing IDs: {path}"
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {path}"

    code_count = 0
    for index, cell in enumerate(cells):
        assert cell.get("cell_type") in ("markdown", "code"), (path, index)
        assert isinstance(cell.get("metadata"), dict), (path, index)
        assert isinstance(cell.get("source"), list), (path, index)
        if cell["cell_type"] != "code":
            continue

        code_count += 1
        assert cell.get("execution_count") is None, (path, index)
        assert cell.get("outputs") == [], (path, index)
        source = "".join(cell["source"])
        try:
            ast.parse(syntax_for_notebook(source), filename=f"{path.name}:cell-{index}")
        except SyntaxError as exc:
            raise AssertionError(f"Syntax issue: {path.name} cell {index}: {exc}") from exc

    nbformat.validate(nbformat.from_dict(data))
    return len(cells), code_count


def main() -> None:
    folder = ROOT / "notebooks"
    actual = {path.name for path in folder.glob("*.ipynb")}
    assert actual == set(NAMES), f"Expected four notebooks, found: {sorted(actual)}"

    for name in NAMES:
        cell_count, code_count = validate_notebook(folder / name)
        print(f"PASS: {name} — {cell_count} cells, {code_count} code cells")

    print("NOTEBOOK STATIC VALIDATION: PASS")
    print("No notebooks executed; no model training, downloads or inference performed.")


if __name__ == "__main__":
    main()
