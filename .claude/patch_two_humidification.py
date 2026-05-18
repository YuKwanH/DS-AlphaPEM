"""Add a 'two_humidification' calibration scenario between 'one_condition'
and 'fixed_pressure' in the 4 calibration notebooks:
  * dynamic model / polar.ipynb
  * static  model / polar.ipynb
  * dynamic model / HFR.ipynb
  * static  model / HFR.ipynb

The new scenario fits the model on two conditions that share P and T but
differ in RHC, so the resulting parameters are sensitive to humidification.
Defaults: P = 1.3 bar, T = 50 C, RHC in {0, 50}.
"""
import json
import uuid
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


NEW_SCENARIO_DEFS = r'''# Anchor conditions for the partial calibration scenarios -- edit if you
# want different subsets. Defaults are picked so each scenario maps to at
# least one condition present in the experimental dataset.
SINGLE_T   = 60       # degrees C
SINGLE_P   = 1.4e5    # Pa
SINGLE_RHC = 50       # %         (RHC=0 is only available at P=1.3 bar)
TWO_HUM_T  = 50       # degrees C -- scenario "two humidifications"
TWO_HUM_P  = 1.3e5    # Pa        -- (RHC=0 and RHC=50 are both available at P=1.3 bar)
FIXED_P    = 1.3e5    # Pa  -- scenario "fixed pressure"
FIXED_T    = 60       # degrees C -- scenario "fixed temperature"

scenarios = {
    "one_condition":      {
        "label":  f"One condition (T={SINGLE_T}, P={SINGLE_P/1e5:.1f} bar, RHC={SINGLE_RHC})",
        "filter": lambda T, P, RHC: (T == SINGLE_T) and (P == SINGLE_P) and (RHC == SINGLE_RHC),
    },
    "two_humidification": {
        "label":  f"Two humidifications (P={TWO_HUM_P/1e5:.1f} bar, T={TWO_HUM_T}, RHC in {{0, 50}})",
        "filter": lambda T, P, RHC: (T == TWO_HUM_T) and (P == TWO_HUM_P) and (RHC in (0, 50)),
    },
    "fixed_pressure":     {
        "label":  f"Fixed pressure (P={FIXED_P/1e5:.1f} bar, all T, all RHC)",
        "filter": lambda T, P, RHC: P == FIXED_P,
    },
    "fixed_temperature":  {
        "label":  f"Fixed temperature (T={FIXED_T}, all P, all RHC)",
        "filter": lambda T, P, RHC: T == FIXED_T,
    },
    "all_data":           {
        "label":  "All conditions",
        "filter": lambda T, P, RHC: True,
    },
}


def conditions_in_scenario(filter_fn):
    """Return the cond_keys present in the experimental data that pass the filter."""
    keys = []
    for RHC in RHC_tested:
        for P_des in PAC_tested:
            for T_des in TFC_tested:
                cond_key = "T" + str(T_des) + "_P" + str(int(P_des/1e2 - 1e3)) + "_HRC" + str(RHC)
                # Look up against whichever experimental-data dict this notebook uses.
                exp_data = globals().get("polardata_exp") or globals().get("hfrdata_exp") or {}
                if cond_key in exp_data and filter_fn(T_des, P_des, RHC):
                    keys.append(cond_key)
    return keys


for name, info in scenarios.items():
    matches = conditions_in_scenario(info["filter"])
    print(f"  {name:20s}  {len(matches):2d} conditions  -> {matches}")
'''


def patch(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # 1. Replace the scenario-defs code cell.
    found_defs = False
    for c in nb["cells"]:
        if c.get("cell_type") == "code" and "Anchor conditions" in "".join(c.get("source", [])):
            c["source"] = NEW_SCENARIO_DEFS.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            found_defs = True
            break
    if not found_defs:
        raise SystemExit(f"!! no scenario-defs cell found in {nb_path}")

    # 2. Insert the new per-scenario plot cell pair right after the
    # 'one_condition' plot cell. The pattern is:
    #     md("### Scenario: `one_condition`")
    #     code('plot_scenario_fit("one_condition")')
    #     md("### Scenario: `fixed_pressure`")
    #     ...
    new_md   = md("### Scenario: `two_humidification`\n"
                  "Fit on the two operating conditions that share P and T but "
                  "differ in cathode humidification (RHC = 0 and 50).\n")
    new_code = code('plot_scenario_fit("two_humidification")\n')

    inserted = False
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" \
                and 'plot_scenario_fit("one_condition")' in "".join(c.get("source", [])):
            # Skip past this code cell to insert immediately after it.
            nb["cells"][i + 1:i + 1] = [new_md, new_code]
            inserted = True
            break
    if not inserted:
        raise SystemExit(f"!! no plot_scenario_fit(\"one_condition\") cell found in {nb_path}")

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(nb["cells"])


notebooks = [
    "simulation/parameter calibration/dynamic model/polar.ipynb",
    "simulation/parameter calibration/static model/polar.ipynb",
    "simulation/parameter calibration/dynamic model/HFR.ipynb",
    "simulation/parameter calibration/static model/HFR.ipynb",
]
for p in notebooks:
    n = patch(Path(p))
    print(f"  {p}: total cells = {n}")

import nbformat
for p in notebooks:
    nbformat.validate(nbformat.read(p, as_version=4))
print("All four notebooks valid.")
