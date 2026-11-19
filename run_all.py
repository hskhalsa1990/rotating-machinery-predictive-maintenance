"""Headless runner: executes every code cell of notebook_source.py in order (no Jupyter needed),
captures printed text and matplotlib figures, and writes Rotating_Machinery_PdM_Suite.ipynb WITH outputs.
All charts are also saved as PNG files in the current folder by the notebook code itself.

Usage: python run_all.py
NOTE: this is a simple custom runner, not a Jupyter kernel.
"""
import ast
import base64
import contextlib
import io
import json
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_notebook import NOTEBOOK, parse, to_notebook  # noqa: E402


def _display(*objs):
    for o in objs:
        print(o.to_string() if hasattr(o, "to_string") else o)


def main():
    cells = parse()
    ns = {"__name__": "__main__", "display": _display}
    outputs = {}
    t_all = time.time()
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        buf, outs, t0 = io.StringIO(), [], time.time()
        try:
            tree = ast.parse(src)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {i}>", "exec"), ns)
                res = eval(compile(last, f"<cell {i}>", "eval"), ns) if last is not None else None
            if res is not None:
                text = res.to_string() if hasattr(res, "to_string") else repr(res)
                outs.append({"output_type": "execute_result", "execution_count": None, "metadata": {},
                             "data": {"text/plain": text.splitlines(True)}})
        except Exception:
            print(f"ERROR in cell {i}:\n{src[:400]}\n", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        text = buf.getvalue()
        if text:
            outs.insert(0, {"output_type": "stream", "name": "stdout", "text": text.splitlines(True)})
        for num in plt.get_fignums():
            b = io.BytesIO()
            plt.figure(num).savefig(b, format="png", dpi=70, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode()}})
        plt.close("all")
        outputs[i] = outs
        print(f"cell {i:3d} ok  {time.time() - t0:5.1f}s")
    json.dump(to_notebook(cells, outputs), open(NOTEBOOK, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"done in {time.time() - t_all:.0f}s -> {NOTEBOOK}")


if __name__ == "__main__":
    main()
