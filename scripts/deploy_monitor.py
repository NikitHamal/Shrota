#!/usr/bin/env python3
"""Push monitor/shrota_ui.py into the live MoLab Marimo session.

Only touches notebook cells (delete stale cells, run the new ones, save the file).
Background GPU jobs are never signalled or restarted.

    python3 scripts/deploy_monitor.py            # deploy
    python3 scripts/deploy_monitor.py --dry-run  # show cell ids + sizes
"""
import argparse
import ast
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import molab_exec as mx  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "monitor" / "shrota_ui.py"
CELL_IDS = ["RZXw", "aexf", "nRTa"]  # stable ids reused across deploys
NOTEBOOK = "/marimo/notebook.py"


def extract_cells(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    cells = []
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.FunctionDef) and any("cell" in ast.unparse(d) for d in node.decorator_list)):
            continue
        body = node.body
        if isinstance(body[-1], ast.Return):
            body = body[:-1]
        first, last = body[0].lineno, body[-1].end_lineno
        cells.append(textwrap.dedent("\n".join(lines[first - 1:last])).strip() + "\n")
    return cells


def current_cell_ids() -> list[str]:
    out = mx.run_bash(
        "python3 -c \"import json;print(' '.join(c['id'] for c in "
        f"json.load(open('/marimo/__marimo__/session/notebook.py.json'))['cells']))\""
    )
    return out.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    codes = extract_cells(SRC)
    assert len(codes) == len(CELL_IDS), f"expected {len(CELL_IDS)} cells, got {len(codes)}"
    if args.dry_run:
        for cid, c in zip(CELL_IDS, codes):
            print(cid, len(c.splitlines()), "lines")
        return
    client = mx.get_client()
    stale = [c for c in current_cell_ids() if c not in CELL_IDS]
    for cid in stale:
        client.api_post("/api/kernel/delete", {"cellId": cid})
        print("deleted stale cell", cid)
    client.api_post("/api/kernel/run", {"cellIds": CELL_IDS, "codes": codes})
    print("ran", CELL_IDS)
    client.api_post("/api/kernel/save", {
        "cellIds": CELL_IDS, "codes": codes, "names": ["_"] * len(codes),
        "configs": [{"column": None, "disabled": False, "hide_code": False}] * len(codes),
        "filename": NOTEBOOK, "persist": True,
    })
    print("saved", NOTEBOOK)


if __name__ == "__main__":
    main()
