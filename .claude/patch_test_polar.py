"""One-shot patch script: add GUI-style polarization presentation sections
to simulation/Test_polar/all_cond 50A dyn.ipynb and ...stat.ipynb."""
import json
import uuid
from pathlib import Path


def make_md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


def make_code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


# =================================================================
# DYNAMIC notebook -- 1 header + 4 plot cells
# =================================================================
DYN_HEADER = (
    "## GUI-style polarization presentation\n"
    "Mirrors the result tabs that the Streamlit GUI shows for a polarization "
    "test profile, adapted to a per-current snapshot:\n"
    "1. Polar curve (Cell performance tab)\n"
    "2. Spatial profile across the MEA (Spatial profile tab)\n"
    "3. Manifold pressures and humidities (Manifolds tab)\n"
    "4. Water content (Water content tab)\n\n"
    "Each plot puts current on the x-axis and uses `plot_condition` for the "
    "colour / linestyle / marker mapping across (T, P, RHC) conditions.\n"
)

DYN_PANEL_CELL = r'''# 1. Polar curve  -- analogue of the GUI's "Cell performance" tab.
# Rebuild Ucell = Ueq - eta_c - i * (sum(Rmem) + Re) from the per-current snapshot.
fig, ax = plt.subplots(figsize=(8, 4))
for cond_key, value in pola_tests_sim.items():
    states = value["states"]
    Ucell = []
    for k in range(len(load_points)):
        Rmem_total = float(np.sum(states["Rmem"][k]))
        Ucell.append(states["Ueq"][k]
                     - states["eta_c"][k]
                     - states["i_fc"][k] * (Rmem_total + parameters["Re"]))
    plot_condition(ax, load_points, Ucell, cond_key, linewidth=1.6, markersize=5)
ax.set_title("Polarization curve (simulation)")
ax.set_xlabel("Current (A)")
ax.set_ylabel("Cell voltage (V)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

DYN_SPATIAL_CELL = r'''# 2. Spatial profile across the MEA at the highest tested current --
# analogue of the GUI's "Spatial profile" tab (modules.display.build_profile_figure),
# but draws one curve per (T, P, RHC) condition instead of one time index.
profile_panels = [
    ("v",          nodes_names_vp, "Vapor (mol/m$^3$)"),
    ("O2",         nodes_name_O2,  "Oxygen (mol/m$^3$)"),
    ("H2",         nodes_names_H2, "Hydrogen (mol/m$^3$)"),
    ("saturation", nodes_names_s,  "Liquid saturation (-)"),
    ("lambda",     nodes_lambda,   r"Water content $\lambda$ (-)"),
    ("T",          nodes_T,        "Temperature (K)"),
]
last_idx = len(load_points) - 1
fig, axes = plt.subplots(figsize=(14, 8), nrows=3, ncols=2)
for ax, (key, var_list, ylabel) in zip(axes.flatten(), profile_panels):
    for cond_key, value in pola_tests_sim.items():
        compact = [value["states"][n][last_idx] for n in var_list]
        expanded = expand_profile_on_nodes(key, compact)
        plot_condition(ax, nodes, expanded, cond_key, linewidth=1.3, markersize=3)
    for x in borders:
        ax.axvline(x=x, color="0.7", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
fig.suptitle(f"Spatial profile at I = {load_points[-1]} A", fontsize=11)
plt.tight_layout()
plt.show()
'''

DYN_MANIFOLD_CELL = r'''# 3. Manifold pressures and humidities vs current --
# analogue of the GUI's "Manifolds" tab (2 x 4 = 8 panels).
manifold_vars = [
    ("Phi_asm", "Anode supply RH (-)"),
    ("Pasm",    "Anode supply pressure (Pa)"),
    ("Phi_csm", "Cathode supply RH (-)"),
    ("Pcsm",    "Cathode supply pressure (Pa)"),
    ("Phi_aem", "Anode exhaust RH (-)"),
    ("Paem",    "Anode exhaust pressure (Pa)"),
    ("Phi_cem", "Cathode exhaust RH (-)"),
    ("Pcem",    "Cathode exhaust pressure (Pa)"),
]
fig, axes = plt.subplots(figsize=(14, 6), nrows=2, ncols=4)
for ax, (var, title) in zip(axes.flatten(), manifold_vars):
    for cond_key, value in pola_tests_sim.items():
        if var not in value["states"]:
            continue
        plot_condition(ax, load_points, value["states"][var], cond_key, linewidth=1.4, markersize=4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Current (A)")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

DYN_WATER_CELL = r'''# 4. Water content -- analogue of the GUI's "Water content" tab.
# Lambda in each CL vs current, plus a heatmap of the per-node membrane
# lambda at I = max (one row per condition).
fig, axes = plt.subplots(figsize=(12, 3), nrows=1, ncols=2)
for cond_key, value in pola_tests_sim.items():
    plot_condition(axes[0], load_points, value["states"]["lambda_acl"], cond_key, linewidth=1.4)
    plot_condition(axes[1], load_points, value["states"]["lambda_ccl"], cond_key, linewidth=1.4)
for ax, t in zip(axes, ("Anode CL", "Cathode CL")):
    ax.set_title(f"{t} water content")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel(r"$\lambda$ (-)")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

mem_keys = [f"lambda_mem_{i+1}" for i in range(parameters["n_mem"])]
last_idx = len(load_points) - 1
mat = np.array([[value["states"][k][last_idx] for k in mem_keys]
                for value in pola_tests_sim.values()])
fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(pola_tests_sim))))
im = ax.imshow(mat, aspect="auto", origin="lower", cmap="cividis")
ax.set_yticks(range(len(pola_tests_sim)))
ax.set_yticklabels(list(pola_tests_sim.keys()), fontsize=8)
ax.set_xlabel("Membrane node")
ax.set_title(f"Membrane water content $\\lambda$ at I = {load_points[-1]} A")
fig.colorbar(im, ax=ax, label=r"$\lambda$ (-)")
plt.tight_layout()
plt.show()
'''

# =================================================================
# STATIC notebook -- 1 header + 2 plot cells
# =================================================================
STAT_HEADER = (
    "## GUI-style polarization presentation\n"
    "Mirrors the Streamlit GUI output for a polarization profile run in "
    "Static mode (single polar curve), plus the spatial profile across the "
    "MEA at the highest tested current.\n"
)

STAT_POLAR_CELL = r'''# Polar curve (Ucell vs I) -- the GUI's static-mode output,
# rebuilt from the per-current `model.solve(i)` result dicts.
fig, ax = plt.subplots(figsize=(8, 4))
for cond_key, group in stat_sim_log.items():
    Ucell_curve = []
    for I, sol_i in zip(I_points, group["states"]):
        i_density = I / parameters["Aact"]
        Ucell_curve.append(sol_i["Ueq"] - sol_i["eta_c"]
                           - i_density * (sol_i["Rohm"] + sol_i["Rccl"] + sol_i["Racl"]))
    ax.plot(I_points, Ucell_curve, marker="o", linewidth=1.6, markersize=5, label=cond_key)
ax.set_title("Static-model polarization curves")
ax.set_xlabel("Current (A)")
ax.set_ylabel("Cell voltage (V)")
ax.legend(fontsize=7, loc="best", ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

STAT_SPATIAL_CELL = r'''# Spatial profile across the MEA at the highest tested current.
# Static `model.solve(i)` returns lambda_mem / s_agdl / s_cgdl / C_v_agdl /
# C_v_cgdl as per-node arrays.
import numpy as np

last_idx = len(I_points) - 1
fig, axes = plt.subplots(figsize=(14, 6), nrows=2, ncols=2)

x_mem  = np.linspace(0, parameters["Hmem"], 10)
x_agdl = np.linspace(0, parameters["Hgdl"], 10)
x_cgdl = np.linspace(0, parameters["Hgdl"], 10)

for cond_key, group in stat_sim_log.items():
    sol_i = group["states"][last_idx]
    axes[0, 0].plot(x_mem,  sol_i["lambda_mem"], marker="o", label=cond_key, linewidth=1.3, markersize=3)
    axes[0, 1].plot(x_cgdl, sol_i["s_cgdl"],     marker="o", label=cond_key, linewidth=1.3, markersize=3)
    axes[1, 0].plot(x_agdl, sol_i["C_v_agdl"],   marker="o", label=cond_key, linewidth=1.3, markersize=3)
    axes[1, 1].plot(x_cgdl, sol_i["C_v_cgdl"],   marker="o", label=cond_key, linewidth=1.3, markersize=3)

axes[0, 0].set_title("Membrane water content");          axes[0, 0].set_ylabel(r"$\lambda$ (-)")
axes[0, 1].set_title("Cathode GDL saturation");          axes[0, 1].set_ylabel("s (-)")
axes[1, 0].set_title("Anode GDL vapor concentration");   axes[1, 0].set_ylabel("C$_v$ (mol/m$^3$)")
axes[1, 1].set_title("Cathode GDL vapor concentration"); axes[1, 1].set_ylabel("C$_v$ (mol/m$^3$)")
for ax in axes.flatten():
    ax.set_xlabel("x (m)")
    ax.grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=6, loc="best", ncol=2)
fig.suptitle(f"Spatial profile at I = {I_points[-1]} A", fontsize=11)
plt.tight_layout()
plt.show()
'''


def patch_notebook(nb_path, cells_to_drop, cells_to_append):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if cells_to_drop:
        nb["cells"] = [c for c in nb["cells"] if c.get("id") not in cells_to_drop]
    # Strip trailing empty code cells
    while nb["cells"] and nb["cells"][-1].get("cell_type") == "code" and not "".join(nb["cells"][-1].get("source", [])).strip():
        nb["cells"].pop()
    nb["cells"].extend(cells_to_append)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(nb["cells"])


dyn_path = Path("simulation/Test_polar/all_cond 50A dyn.ipynb")
stat_path = Path("simulation/Test_polar/all_cond 50A stat.ipynb")

dyn_total = patch_notebook(
    dyn_path,
    cells_to_drop=set(),
    cells_to_append=[
        make_md(DYN_HEADER),
        make_code(DYN_PANEL_CELL),
        make_code(DYN_SPATIAL_CELL),
        make_code(DYN_MANIFOLD_CELL),
        make_code(DYN_WATER_CELL),
    ],
)
print(f"dyn.ipynb: total cells = {dyn_total}")

# Static notebook -- drop the duplicate mass-flow cell and the broken polar cell.
stat_total = patch_notebook(
    stat_path,
    cells_to_drop={"d3020bf7", "a12d26ca"},
    cells_to_append=[
        make_md(STAT_HEADER),
        make_code(STAT_POLAR_CELL),
        make_code(STAT_SPATIAL_CELL),
    ],
)
print(f"stat.ipynb: total cells = {stat_total}")

import nbformat
nbformat.validate(nbformat.read(str(dyn_path), as_version=4))
nbformat.validate(nbformat.read(str(stat_path), as_version=4))
print("Both notebooks remain valid.")
