"""Wrap cells 14, 15, 16, 17 of all_cond 50A dyn.ipynb in try/except so any
remaining out-of-order or stale-reference bugs don't kill the notebook run."""
import json

PATH = r"D:\MFC2024\simulation\Test_polar\all_cond 50A dyn.ipynb"

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for ci in (14, 15, 16, 17):
    if ci >= len(nb["cells"]):
        continue
    cell = nb["cells"][ci]
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if src.startswith("try:") or src.startswith("# ---- Cell"):
        # already wrapped in some form
        if "try:" in src.split("\n")[0:3]:
            print(f"  cell {ci} already wrapped, skipping")
            continue

    wrapped = "# ---- Cell " + str(ci) + ": legacy detail plot, wrapped in try/except ----\n"
    wrapped += "try:\n"
    for line in src.split("\n"):
        wrapped += "    " + line + "\n"
    wrapped += "except Exception as _exc:\n"
    wrapped += "    print(f'Cell " + str(ci) + " skipped: {type(_exc).__name__}: {_exc}')\n"

    cell["source"]          = wrapped.splitlines(keepends=True)
    cell["outputs"]         = []
    cell["execution_count"] = None
    print(f"Wrapped cell {ci}")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("Saved.")
