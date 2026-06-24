"""Drill into which profile key triggers the IndexError."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from copy import deepcopy
from config.initialize import parameters as _PARAMS_DEFAULT
from config.initialize import operating_inputs as _OP_DEFAULT
from config.settings import (nodes_postfix, nodes_names_vp, nodes_names_H2,
                             nodes_name_O2, nodes_names_s, nodes_lambda, nodes_T,
                             expand_profile_on_nodes, has_species_value)
from modules.display import extract_profile_dyn, DYN_NAME_GROUPS
from gui import profiles
from gui.runner import run as run_simulation

params    = deepcopy(_PARAMS_DEFAULT)
op_inputs = deepcopy({k: v for k, v in _OP_DEFAULT.items() if k != "current_density"})
profile_func = profiles.step(0.0, 6.0, 20.0, 12000.0, 1.0, 3.0)

model, sol, status = run_simulation(
    params=params, op_inputs=op_inputs,
    model_variant="Dual-scale", profile_func=profile_func,
    t_span=(0.0, 20.0), max_step=0.1, method="BDF",
    polar_sweep=None, aux_system=False,
)

profile = extract_profile_dyn(model, t_index=-1)

print(f"{'key':<12s} {'extracted':>10s}  {'expected':>9s}  {'first 3 vals'}")
print("-" * 60)
for key, names in DYN_NAME_GROUPS:
    vals = profile.get(key, [])
    expected_count = sum(1 for p in nodes_postfix if has_species_value(key, p))
    head = vals[:3] if hasattr(vals, "__len__") else vals
    flag = "" if (len(vals) == 0 or len(vals) == expected_count) else "  <-- MISMATCH"
    print(f"{key:<12s} {len(vals):>10d}  {expected_count:>9d}  {head}{flag}")
print()

for key, names in DYN_NAME_GROUPS:
    vals = profile.get(key, [])
    if not vals:
        print(f"  [{key}] empty -> skipped (expand would no-op via _plot_panels continue)")
        continue
    try:
        y = expand_profile_on_nodes(key, vals)
        print(f"  [{key}] expand OK  -> len {len(y)}")
    except Exception as exc:
        print(f"  [{key}] expand FAILED -> {type(exc).__name__}: {exc}")
        print(f"          len(vals)={len(vals)}  expected={sum(1 for p in nodes_postfix if has_species_value(key, p))}")
