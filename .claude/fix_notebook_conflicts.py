"""Resolve merge-conflict markers in two Jupyter notebooks.

For each conflict block of the form::

    <<<<<<< HEAD
    <head-content>
    =======
    <other-content>
    >>>>>>> <some-hash>

we keep the HEAD content and drop the rest, since HEAD reflects the
current main-branch state (post-`model.dualscale` -> `model.model`
refactor). After stripping markers we re-parse the result as JSON to
make sure the notebook is valid, and pretty-print it back so the file
size stays stable across tools that re-save it.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [
    "simulation/parameter calibration/dynamic model/pointwise.ipynb",
    "simulation/parameter calibration/dynamic model/polar.ipynb",
]


def resolve(text):
    """Return text with every conflict block reduced to its HEAD side."""
    out = []
    in_conflict = False
    in_head = False     # True while inside `<<<<<<< HEAD .. =======`
    n_conflicts = 0
    for line in text.splitlines(keepends=True):
        s = line.lstrip()
        if s.startswith("<<<<<<<"):
            in_conflict = True
            in_head = True
            n_conflicts += 1
            continue
        if in_conflict and s.startswith("======="):
            in_head = False
            continue
        if in_conflict and s.startswith(">>>>>>>"):
            in_conflict = False
            in_head = False
            continue
        if not in_conflict:
            out.append(line)
        elif in_head:
            out.append(line)
        # in_conflict and not in_head -> skip
    return "".join(out), n_conflicts


def main():
    for rel in TARGETS:
        path = os.path.join(ROOT, rel)
        print(f"\n=== {rel} ===")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        backup = path + ".pre-fix.bak"
        with open(backup, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  backup written -> {os.path.basename(backup)}")

        new_text, n = resolve(text)
        print(f"  {n} conflict block(s) resolved -> kept HEAD side")

        # Validate by parsing as JSON.
        try:
            nb = json.loads(new_text)
        except json.JSONDecodeError as exc:
            print(f"  [FAIL] still not valid JSON: {exc}")
            print(f"         leaving the backup at {backup}")
            continue

        # Quick sanity: notebook should have cells.
        n_cells = len(nb.get("cells", []))
        kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name", "?")
        print(f"  [ok] valid notebook  ·  {n_cells} cells  ·  kernel={kernel}")

        # Re-serialise with the same formatting Jupyter uses (indent=1).
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"  rewrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
