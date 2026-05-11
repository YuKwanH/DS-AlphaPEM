"""Replace the cluttered 2x2 comparison figure with one cell per scenario.
Each per-scenario cell calls a shared helper that produces two figures:
  1. the fit on the conditions USED for calibration
  2. the fit on ALL conditions (used + held out)
Per-condition mini-panels keep the curves readable.

Applies to both ./dynamic model/polar.ipynb and ./static model/polar.ipynb.
"""
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


HELPER_MD = (
    "## Compare best-fit polarization curves across scenarios\n"
    "The helper `plot_scenario_fit(name)` produces two figures for one "
    "scenario:\n"
    "1. **USED for calibration** — one mini-panel per condition that was "
    "actually fitted; if the optimization worked these should match closely.\n"
    "2. **ALL conditions** — the same plot but over every operating "
    "condition. Held-out conditions reveal how well the calibrated parameters "
    "generalize. Panel titles tagged `[CAL]` mark conditions that were used "
    "during calibration.\n"
)

HELPER_CELL = r'''import math

def plot_scenario_fit(name):
    """Produce two per-condition grids for a calibration scenario:
    1. only the conditions used in the fit
    2. every condition in the experimental dataset
    Blue solid markers = experiment, red dotted squares = simulation.
    """
    info = results[name]
    cal_set = set(info["calibrated_keys"])

    panels_to_show = [
        ("USED for calibration", info["calibrated_keys"]),
        ("ALL conditions",       list(polardata_exp.keys())),
    ]
    for title_suffix, keys in panels_to_show:
        n = len(keys)
        if n == 0:
            print(f"({name}: no conditions for '{title_suffix}')")
            continue
        ncols = min(4, max(1, n))
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, cond_key in zip(axes.flatten(), keys):
            ucell_exp = _experimental_ucell(cond_key)
            ucell_sim = info["sim_all"].get(cond_key)
            ax.plot(I_tested, ucell_exp, "o-", color="tab:blue",
                    linewidth=1.4, label="experiment")
            if ucell_sim is not None:
                ax.plot(I_tested, ucell_sim, "s:", color="tab:red",
                        linewidth=1.2, label="simulated")
            in_cal = cond_key in cal_set
            tag = "[CAL] " if in_cal else ""
            ax.set_title(f"{tag}{cond_key}", fontsize=8,
                         color="tab:green" if in_cal else "black")
            ax.grid(True, alpha=0.3)
        for ax in axes.flatten()[n:]:
            ax.set_visible(False)
        axes.flatten()[0].legend(fontsize=8, loc="best")
        fig.suptitle(f"{info['label']}  --  {title_suffix}\n"
                     f"objective on calibrated subset = {info['best_value']:.4f}",
                     fontsize=10)
        fig.tight_layout()
        plt.show()
'''

def scenario_md(scenario_name):
    return (
        f"### Scenario: `{scenario_name}`\n"
        "Two figures — fit on the calibrated subset first, then on every "
        "experimental condition.\n"
    )


def scenario_cell(scenario_name):
    return f'plot_scenario_fit("{scenario_name}")\n'


def patch_notebook(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Find and remove the existing "Compare best-fit" markdown + plot cell.
    # The 2x2 plot cell is the code cell after the "Compare best-fit" markdown.
    new_cells = []
    skip_next_code = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown" and "Compare best-fit polarization curves across scenarios" in src:
            skip_next_code = True
            continue  # drop the old markdown header
        if skip_next_code and c.get("cell_type") == "code":
            skip_next_code = False
            continue  # drop the old 2x2 plot cell
        new_cells.append(c)
    nb["cells"] = new_cells

    # Locate where to insert the new per-scenario block. We want it right
    # after the "## Best parameters side-by-side" table cell. Find that cell;
    # if missing, insert before the residual heatmap section. Otherwise append.
    insert_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "markdown" and "Best parameters side-by-side" in "".join(c["source"]):
            # The corresponding code cell follows immediately.
            insert_idx = i + 2
            break
    if insert_idx is None:
        # Fallback: before "## Residual heatmap" md
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") == "markdown" and "Residual heatmap" in "".join(c["source"]):
                insert_idx = i
                break
    if insert_idx is None:
        insert_idx = len(nb["cells"])

    new_block = [
        make_md(HELPER_MD),
        make_code(HELPER_CELL),
        make_md(scenario_md("one_condition")),
        make_code(scenario_cell("one_condition")),
        make_md(scenario_md("fixed_pressure")),
        make_code(scenario_cell("fixed_pressure")),
        make_md(scenario_md("fixed_temperature")),
        make_code(scenario_cell("fixed_temperature")),
        make_md(scenario_md("all_data")),
        make_code(scenario_cell("all_data")),
    ]
    nb["cells"][insert_idx:insert_idx] = new_block

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(nb["cells"])


for path_str in [
    "simulation/parameter calibration/dynamic model/polar.ipynb",
    "simulation/parameter calibration/static model/polar.ipynb",
]:
    n = patch_notebook(Path(path_str))
    print(f"{path_str}: total cells = {n}")

import nbformat
for path_str in [
    "simulation/parameter calibration/dynamic model/polar.ipynb",
    "simulation/parameter calibration/static model/polar.ipynb",
]:
    nbformat.validate(nbformat.read(path_str, as_version=4))
print("Both notebooks valid.")
