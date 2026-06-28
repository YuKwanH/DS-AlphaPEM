"""Verify the new discrete `e` parameter end-to-end across all 3 optimizers."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from copy import deepcopy
from config.initialize import parameters as P
from data.export import export_experiment_data
from gui import calibration as cal
from gui import calib_backend as cb


# 1. Schema and detection helpers
print("=== Block 1: schema ===")
assert "e" in cal.CALIB_PARAMS
spec = cal.CALIB_PARAMS["e"]
assert len(spec) == 6 and spec[5] == [3, 4, 5]
print(f'  [ok] CALIB_PARAMS["e"] = {spec}')
assert cal._is_discrete("e") is True
assert cal._is_discrete("OCV") is False
print(f'  [ok] _is_discrete("e")    = True')
print(f'  [ok] _is_discrete("OCV")  = False')

assert cal._default_bounds("e") == [3, 4, 5]
assert cal._default_bounds("OCV") == (0.90, 1.00)
print(f'  [ok] _default_bounds("e")    = {cal._default_bounds("e")}')
print(f'  [ok] _default_bounds("OCV")  = {cal._default_bounds("OCV")}')

assert "e" in cal.DEFAULT_PARAMS
print(f'  [ok] "e" appears in DEFAULT_PARAMS: {cal.DEFAULT_PARAMS}')


# 2. Backend _is_categorical helper
print()
print("=== Block 2: backend categorical detection ===")
assert cb._is_categorical([3, 4, 5]) is True
assert cb._is_categorical((0.1, 0.5)) is False
print(f'  [ok] _is_categorical([3,4,5])  = True')
print(f'  [ok] _is_categorical((0.1,.5)) = False')


# 3. Drive each optimizer with a tiny budget and confirm `e` lands on {3,4,5}
print()
print("=== Block 3: optimizers honour the discrete set ===")
pola = export_experiment_data("pola")
cond_keys = sorted(pola.keys())[:1]
baseline = deepcopy(P)

base_request = {
    "target":     "Polarization",
    "model":      "Static",
    "n_trials":   12,
    "seed":       7,
    "params":     ["i0_c_ref", "e"],
    "bounds":     {"i0_c_ref": (1e-7, 1e-3), "e": [3, 4, 5]},
    "conditions": cond_keys,
}

for optimizer in ("TPE (Optuna)", "Genetic algorithm", "Grid search"):
    req = deepcopy(base_request)
    req["optimizer"] = optimizer
    print(f"\n  >>> {optimizer}")
    res = cb.run_calibration(req, baseline_params=baseline, data=pola)
    e_best = res["best_params"]["e"]
    i0_best = res["best_params"]["i0_c_ref"]
    print(f"     best e         : {e_best!r}  (must be one of [3, 4, 5])")
    print(f"     best i0_c_ref  : {i0_best:.4g}")
    print(f"     best_loss      : {res['best_loss']:.4g}  ({res['n_evals']} evals)")
    assert e_best in (3, 4, 5), f"{optimizer} returned e={e_best!r}, not in [3,4,5]"


print()
print("=== ALL DISCRETE-PARAMETER CHECKS PASS ===")
