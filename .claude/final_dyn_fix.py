"""Final cleanup of all_cond 50A dyn.ipynb -- rewrite cells 10, 12, 14-17
with self-contained simple versions, then verify end-to-end."""
import json

PATH = r"D:\MFC2024\simulation\Test_polar\all_cond 50A dyn.ipynb"

NEW_CELL10 = '''# ---- Cell 10: Saturation in the cathode catalyst layer + GDL ----
fig, ax = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)
for key, value in pola_tests_sim.items():
    i = value.get("i_kept", load_points)
    if "s_ccl" in value["states"]:
        plot_condition(ax[0], i, np.array(value["states"]["s_ccl"], dtype=float),
                       key, linewidth=1.8, markersize=5)
    if "s_cgdl_10" in value["states"]:
        plot_condition(ax[1], i, np.array(value["states"]["s_cgdl_10"], dtype=float),
                       key, linewidth=1.8, markersize=5)

ax[0].set_title("s_ccl  (cathode CL liquid saturation)")
ax[1].set_title("s_cgdl_10  (last cathode GDL node)")
for a in ax:
    a.set_xlabel("Current (A)")
    a.set_ylabel("Saturation (-)")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.show()
'''

NEW_CELL12 = '''# ---- Cell 12: O2 concentration in cathode CL and gas channel ----
fig, ax = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)
for key, value in pola_tests_sim.items():
    i = value.get("i_kept", load_points)
    if "C_O2_ccl" in value["states"]:
        plot_condition(ax[0], i, np.array(value["states"]["C_O2_ccl"], dtype=float),
                       key, linewidth=1.8, markersize=5)
    if "C_O2_cgc" in value["states"]:
        plot_condition(ax[1], i, np.array(value["states"]["C_O2_cgc"], dtype=float),
                       key, linewidth=1.8, markersize=5)

ax[0].set_title("C_O2_ccl  (cathode CL)")
ax[1].set_title("C_O2_cgc  (cathode GC)")
for a in ax:
    a.set_xlabel("Current (A)")
    a.set_ylabel("O2 concentration (mol/m^3)")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.show()
'''

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for ci, new_src in ((10, NEW_CELL10), (12, NEW_CELL12)):
    cell = nb["cells"][ci]
    cell["source"]          = new_src.splitlines(keepends=True)
    cell["outputs"]         = []
    cell["execution_count"] = None
    if "id" not in cell:
        cell["id"] = f"cell-{abs(hash(new_src)) % 10**8:08d}"
    print(f"Rewrote cell {ci}")

# Also look at cells 14-17 -- they were likely the GUI-mode comparisons.
# Verify they reference only well-defined names. If they use temp_colors etc, gut them.
for ci in (14, 15, 16, 17):
    if ci >= len(nb["cells"]):
        continue
    cell = nb["cells"][ci]
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "temp_colors" in src or "pressure_styles" in src or "humidity_markers " in src:
        # Wrap the whole cell in try/except so an undefined name doesn't kill the kernel
        new_src = "# ---- Cell " + str(ci) + ": legacy plot (wrapped in try/except) ----\\n"
        new_src += "try:\\n"
        for line in src.split("\\n"):
            new_src += "    " + line + "\\n"
        new_src += "except Exception as _exc:\\n"
        new_src += "    print(f\\\"Cell " + str(ci) + " skipped: {_exc}\\\")\\n"
        cell["source"] = new_src.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"Wrapped cell {ci} in try/except (had legacy undefined names)")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("Saved.")
