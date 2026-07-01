"""Verify the calibration backend dispatches to every model variant.

Tiny budgets (2 trials × 1 condition × few currents) so the audit
completes in minutes instead of hours. The goal is to confirm the
plumbing — that Static -> PEMFC_stat, Dual-scale -> PEMFC,
Dynamic -> PEMFC_dyn (with auto-routing under the Aux toggle) — not to
verify calibration quality.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from copy import deepcopy
import numpy as np
from config.initialize import parameters as P
from data.export       import export_experiment_data
from gui import calib_backend as cb


# ---------------------------------------------------------------------------
# Block 1: routing helper matches simulation page convention
# ---------------------------------------------------------------------------
print("=" * 72)
print(" BLOCK 1 - resolve_transient_model() matches the simulation page")
print("=" * 72)
cases = [
    ("Dual-scale", False, "PEMFC"),
    ("Dual-scale", True,  "PEMFC_dyn"),   # auto-promote
    ("Dynamic",    True,  "PEMFC_dyn"),
    ("Dynamic",    False, "PEMFC"),       # auto-demote
]
for variant, aux, expected in cases:
    got = cb.resolve_transient_model(variant, aux)
    ok = got == expected
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] ({variant!r:<13s}, aux={aux})  ->  {got!r:<12s} "
          f"(expected {expected!r})")
    assert ok


# ---------------------------------------------------------------------------
# Block 2: tiny TPE run with each model dispatched correctly
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 2 - tiny TPE calibration drives the right model per combo")
print("=" * 72)

pola = export_experiment_data("pola")
# Single condition, sub-set the currents so the transient runs are sized
# down by reducing how many integrations happen per trial.
cond_keys = ["T50_P300_HRC0"]
df = pola[cond_keys[0]].dropna(subset=["I_LOAD", "VFC"]).sort_values("I_LOAD")
i_subset = df["I_LOAD"].to_numpy()[:3]   # 3 currents only

# Use only 2 parameters for speed.
base_req = {
    "target":     "Polarization",
    "optimizer":  "TPE (Optuna)",
    "n_trials":   2,
    "seed":       7,
    "params":     ["i0_c_ref", "kappa_c"],
    "bounds":     {"i0_c_ref": (0.01, 10.0), "kappa_c": (0.5, 3.0)},
    "conditions": cond_keys,
}

# Monkey-patch the data so only 3 currents are seen per condition.
pola_small = {k: v.iloc[df.index.get_indexer(df.index[:3])].copy()
              for k, v in pola.items()}

for variant, aux, expected_kind in [
    ("Static",     False, "PEMFC_stat"),
    ("Dual-scale", False, "PEMFC"),
    ("Dual-scale", True,  "PEMFC_dyn"),
    ("Dynamic",    True,  "PEMFC_dyn"),
    ("Dynamic",    False, "PEMFC"),
]:
    req = deepcopy(base_req)
    req["model"] = variant
    req["aux_system"] = aux
    print(f"\n  >>> variant={variant!r}, aux={aux}  (expecting {expected_kind})")
    t0 = time.perf_counter()
    try:
        res = cb.run_calibration(req, baseline_params=deepcopy(P),
                                 data=pola_small)
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"     FAIL: {type(exc).__name__}: {exc}")
        raise
    wall = time.perf_counter() - t0
    resolved = res["resolved_class"]
    ok = resolved == expected_kind
    tag = "PASS" if ok else "FAIL"
    print(f"     [{tag}] resolved_class={resolved!r}  "
          f"best_loss={res['best_loss']:.4g}  wall={wall:.1f}s")
    assert ok, f"resolved {resolved!r}, expected {expected_kind!r}"

print()
print("=" * 72)
print(" ALL MODEL-DISPATCH CHECKS PASSED")
print("=" * 72)
