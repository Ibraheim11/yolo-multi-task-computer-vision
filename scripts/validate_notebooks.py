"""Validate scaffolds offline; nbformat enables full official schema checking."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = sorted((root / "notebooks").glob("*.ipynb"))
assert len(paths) == 4, "Expected four notebooks"
try:
    import nbformat
except ImportError:
    raise SystemExit("nbformat is required: install requirements.txt before schema validation")
for path in paths:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["source"], list)
        assert isinstance(cell["metadata"], dict)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    if nbformat:
        nbformat.validate(nbformat.from_dict(notebook))
    print(f"PASS: {path.name}: JSON, structure, Python syntax, clean outputs")
print("Official nbformat schema: " + ("PASS" if nbformat else "NOT RUN (install requirements.txt)"))
