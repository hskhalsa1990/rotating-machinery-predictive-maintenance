"""Convert notebook_source.py (percent format) into a Jupyter notebook JSON file (no nbformat needed).

Usage: python make_notebook.py   -> writes Rotating_Machinery_PdM_Suite.ipynb (without outputs)
For a notebook WITH outputs run: python run_all.py
"""
import json
import re

SOURCE = "notebook_source.py"
NOTEBOOK = "Rotating_Machinery_PdM_Suite.ipynb"


def parse(path=SOURCE):
    cells, cur = [], None
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            if cur:
                cells.append(cur)
            cur = {"type": "markdown" if "[markdown]" in line else "code", "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        cells.append(cur)
    out = []
    for c in cells:
        lines = c["lines"]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if c["type"] == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        out.append((c["type"], "\n".join(lines)))
    return out


def _lines(src):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def to_notebook(cells, outputs=None):
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})
        else:
            outs = outputs[i] if outputs else []
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": outs, "source": _lines(src)})
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    json.dump(to_notebook(parse()), open(NOTEBOOK, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", NOTEBOOK)
