"""Patch every notebook that imports from model.dualscale or model.dynamic.

Replaces:
  * 'from model.dualscale' -> 'from model.model'
  * 'from model.dynamic'   -> 'from model.model'
  * 'model.dualscale' / 'model.dynamic' bare references in markdown/code
    cells (preserves any other text on the line)
"""
import json, os, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS = [
    # (regex, replacement)
    (re.compile(r"from\s+model\.dualscale\b"),  "from model.model"),
    (re.compile(r"from\s+model\.dynamic\b"),    "from model.model"),
    (re.compile(r"\bmodel\.dualscale\b"),       "model.model"),
    (re.compile(r"\bmodel\.dynamic\b"),         "model.model"),
]

def patch_text(s):
    out = s
    for rx, rep in PATTERNS:
        out = rx.sub(rep, out)
    return out


def patch_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    changed = 0
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, str):
            new = patch_text(src)
            if new != src:
                cell["source"] = new
                changed += 1
        elif isinstance(src, list):
            new_lines = [patch_text(line) for line in src]
            if new_lines != src:
                cell["source"] = new_lines
                changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return changed


def walk():
    targets = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # skip the worktrees and any .ipynb_checkpoints
        if ".claude\\worktrees" in dirpath or ".ipynb_checkpoints" in dirpath:
            continue
        if dirpath.startswith(os.path.join(ROOT, ".claude", "worktrees")):
            continue
        for f in filenames:
            if f.endswith(".ipynb"):
                p = os.path.join(dirpath, f)
                # quick text scan to skip uninteresting notebooks
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    if "model.dualscale" in text or "model.dynamic" in text:
                        targets.append(p)
                except Exception:
                    pass
    return targets


def main():
    targets = walk()
    print(f"Found {len(targets)} notebooks with dualscale/dynamic references:")
    grand_total = 0
    skipped = []
    for p in targets:
        rel = os.path.relpath(p, ROOT)
        try:
            n = patch_notebook(p)
        except json.JSONDecodeError as exc:
            print(f"  [SKIP malformed  ] {rel}    ({exc})")
            skipped.append(rel)
            continue
        if n:
            print(f"  [patched {n:>2d} cells] {rel}")
        else:
            print(f"  [no change       ] {rel}")
        grand_total += n
    if skipped:
        print(f"\nSkipped (malformed JSON): {skipped}")
    print(f"\nTotal cells patched: {grand_total}")


if __name__ == "__main__":
    main()
