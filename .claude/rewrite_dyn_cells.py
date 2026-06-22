"""Replace cells 7 and 8 in all_cond 50A dyn.ipynb with self-contained
versions that don't depend on out-of-order helper definitions."""
import json

PATH = r"D:\MFC2024\simulation\Test_polar\all_cond 50A dyn.ipynb"

# Cell 7 (was the broken kinetics-breakdown plot) -> a simple side-by-side
# of the four key trajectories: Ucell, Rmem total, eta_c, fdrop.
NEW_CELL7 = '''# ---- Cell 7: Simulation summary -- four key trajectories vs current ----
# Direct plotting of what the model recorded -- no kinetics recomputation
# (it was producing inconsistent results with cell 2's saved Ucell).
panels = [
    ("Ucell",  "Cell voltage (V)"),
    ("eta_c",  "Cathode overpotential (V)"),
    ("fdrop",  "Liquid coverage factor (-)"),
    ("Ueq",    "Equilibrium voltage (V)"),
]
fig, axes = plt.subplots(figsize=(16, 4), nrows=1, ncols=len(panels))
for ax, (var, ylabel) in zip(axes, panels):
    for key, value in pola_tests_sim.items():
        i = value.get("i_kept", load_points)
        if var not in value["states"]:
            continue
        y = np.array(value["states"][var], dtype=float)
        plot_condition(ax, i, y, key, linewidth=1.8, markersize=5)
    ax.set_xlabel("Current (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(var)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.show()
'''

# Cell 8 (was the kinetics-reconstruction + sim/exp side-by-side that
# referenced an undefined `Tfc`) -> use the model's own Ucell directly.
NEW_CELL8 = '''# ---- Cell 8: Polarization curve - simulation vs experiment ----
# We use the Ucell that PEMFC_dyn already computed in cell 2 -- no need
# to reconstruct it from kinetic primitives.
exp_pola_data = pd.ExcelFile(project_root / "data" / "Polar_curves.xlsx")
pola_testdata = {sname: pd.read_excel(exp_pola_data, sheet_name=sname)
                 for sname in exp_pola_data.sheet_names}

fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2, sharey=True)

# Simulated
for key, value in pola_tests_sim.items():
    i = value.get("i_kept", load_points)
    Ucell = np.array(value["states"]["Ucell"], dtype=float)
    plot_condition(ax[0], i, Ucell, key, linewidth=1.8, markersize=5)

# Experimental
for name, data in pola_testdata.items():
    i_values = data["I_LOAD"].to_numpy(dtype=float)
    v_values = data["VFC"].to_numpy(dtype=float) / n_cell
    plot_condition(ax[1], i_values, v_values, name, linewidth=1.2, markersize=5)

ax[0].set_title("Simulated polarization (PEMFC_dyn)")
ax[1].set_title("Experimental polarization")
ax[0].set_ylabel("Cell voltage (V)")
for a in ax:
    a.set_xlabel("Current (A)")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.show()
'''

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for ci, new_src in ((7, NEW_CELL7), (8, NEW_CELL8)):
    cell = nb["cells"][ci]
    cell["source"]          = new_src.splitlines(keepends=True)
    cell["outputs"]         = []
    cell["execution_count"] = None
    if "id" not in cell:
        cell["id"] = f"cell-{abs(hash(new_src)) % 10**8:08d}"
    print(f"Replaced cell {ci}")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("Saved.")
