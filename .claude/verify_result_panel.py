"""Reproduce the GUI crash: run a small calibration that mixes discrete
and continuous params, then render the result panel exactly as the GUI
does. Anything that throws is a real defect.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import streamlit as st
import numpy as np
from copy import deepcopy

from config.initialize import parameters as P
from data.export       import export_experiment_data
from gui import calib_backend as cb
from gui import calibration   as cal


# ---- Streamlit mocks so render() can run headless --------------------------
class _C:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __getattr__(self, n): return getattr(st, n)
st.container   = lambda *a, **k: _C()
st.columns     = lambda spec, **k: [_C() for _ in range(
    len(spec) if hasattr(spec, "__len__") else int(spec))]
st.expander    = lambda *a, **k: _C()
st.tabs        = lambda labels, **k: [_C() for _ in labels]
st.selectbox   = lambda label, options, index=0, **k: list(options)[index]
st.multiselect = lambda label, options, default=None, **k: list(default) if default else []
st.number_input = lambda label, value=0, **k: value
st.text_input  = lambda label, value="", **k: value
st.button      = lambda label, **k: False
st.checkbox    = lambda label, value=False, **k: value
st.pyplot      = lambda *a, **k: None
st.dataframe   = lambda df, **k: print(f"  [df rendered] columns={list(df.columns)}  rows={len(df)}")
st.slider      = lambda label, **k: k.get("value", k.get("max_value", 0))
st.progress    = lambda *a, **k: _C()
st.spinner     = lambda *a, **k: _C()
for n in ("markdown", "divider", "caption", "info", "error", "success",
          "warning", "json", "toast"):
    setattr(st, n, lambda *a, **k: None)


# ---- Build a request with BOTH discrete and continuous params -------------
data = export_experiment_data("pola")
cond_keys = sorted(data.keys())[:2]   # two conditions, ~14 currents total
request = {
    "target":     "Polarization",
    "model":      "Static",            # algebraic — fast
    "aux_system": False,
    "optimizer":  "TPE (Optuna)",
    "n_trials":   8,
    "seed":       42,
    "params":     ["i0_c_ref", "kappa_c", "e"],
    "bounds":     {
        "i0_c_ref": (0.01, 15.0),       # continuous (log range)
        "kappa_c":  (0.5, 3.0),         # continuous
        "e":        [3, 4, 5],          # DISCRETE — the case the GUI crashed on
    },
    "conditions": cond_keys,
}

print("=" * 72)
print(" Running Static calibration with mixed continuous + discrete bounds")
print("=" * 72)
print(f"   params      : {request['params']}")
print(f"   bounds      : {request['bounds']}")
print(f"   conditions  : {cond_keys}")
print()

import time
t0 = time.perf_counter()
res = cb.run_calibration(request, baseline_params=deepcopy(P), data=data)
print(f"   best_loss   : {res['best_loss']:.4g}")
print(f"   best_params : {res['best_params']}")
print(f"   wall        : {time.perf_counter()-t0:.1f}s")
print()

# ---- Render the result panel using the SAME code path the GUI uses --------
print("=" * 72)
print(" Rendering result panel (this is where the GUI crashed earlier)")
print("=" * 72)
state = {"calib_result": res, "calib_request": request}
try:
    cal._render_results_panel(state)
    print("\n   [PASS] result panel rendered without exception")
except Exception as exc:
    import traceback; traceback.print_exc()
    print(f"\n   [FAIL] {type(exc).__name__}: {exc}")
    sys.exit(1)


# ---- Also exercise a request with NO discrete params (regression check) ---
print()
print("=" * 72)
print(" Regression: same render with continuous-only bounds")
print("=" * 72)
request2 = deepcopy(request); request2["params"] = ["i0_c_ref", "kappa_c"]
request2["bounds"] = {k: request["bounds"][k] for k in request2["params"]}
res2 = cb.run_calibration(request2, baseline_params=deepcopy(P), data=data)
state2 = {"calib_result": res2, "calib_request": request2}
try:
    cal._render_results_panel(state2)
    print("\n   [PASS] result panel renders for continuous-only too")
except Exception as exc:
    import traceback; traceback.print_exc()
    print(f"\n   [FAIL] {type(exc).__name__}: {exc}")
    sys.exit(1)


# ---- All-discrete edge case ----------------------------------------------
print()
print("=" * 72)
print(" Edge case: all-discrete bounds")
print("=" * 72)
request3 = deepcopy(request)
request3["params"] = ["e"]
request3["bounds"] = {"e": [3, 4, 5]}
res3 = cb.run_calibration(request3, baseline_params=deepcopy(P), data=data)
state3 = {"calib_result": res3, "calib_request": request3}
try:
    cal._render_results_panel(state3)
    print("\n   [PASS] result panel renders for all-discrete params")
except Exception as exc:
    import traceback; traceback.print_exc()
    print(f"\n   [FAIL] {type(exc).__name__}: {exc}")
    sys.exit(1)


print()
print("=" * 72)
print(" ALL RESULT-PANEL CHECKS PASSED")
print("=" * 72)
