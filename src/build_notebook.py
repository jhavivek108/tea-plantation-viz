"""Turn src/analysis.py (cells marked with `# %%`) into an executed notebook.

Usage: python src/build_notebook.py   ->  notebooks/tea_production_analysis.ipynb
"""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "analysis.py"
OUT = ROOT / "notebooks" / "tea_production_analysis.ipynb"


def cells_from_script(text: str) -> list:
    cells = []
    for chunk in text.split("# %%")[1:]:
        if chunk.startswith(" [markdown]"):
            body = chunk.split("\n", 1)[1]
            md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.strip().splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            code = chunk.split("\n", 1)[1].strip("\n")
            if code.strip():
                cells.append(nbformat.v4.new_code_cell(code))
    return cells


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells_from_script(SRC.read_text(encoding="utf-8"))
    nb.cells[1].source = "%matplotlib inline\n" + nb.cells[1].source
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
    nbformat.write(nb, OUT)
    print(f"Wrote executed notebook: {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
