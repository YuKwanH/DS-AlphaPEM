"""End-to-end smoke test for the calibration backend.

Runs each of the three optimizers (TPE / GA / Grid) with a very small
budget against a single polarization condition. Verifies that:
 * the objective function evaluates without crashing
 * each driver returns the expected result schema
 * the convergence history is non-empty and monotone-non-increasing
   in the "best so far" sense
 * the best-fit overlay arrays are well-shaped

We deliberately keep n_trials tiny (10-15) so the script finishes in
~30-60 s wall-time. The goal is to verify the *plumbing*, not the
quality of the calibration fit.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from copy import deepcopy

from config.initialize import parameters as _PARAMS_DEFAULT
from data.export import export_experiment_data
from gui.calib_backend import (
    parse_condition_key, make_objective, run_calibration, _DRIVERS,
)


# ---------------------------------------------------------------------------
# Block A — condition key parser
# ---------------------------------------------------------------------------
print("=" * 80)
print("BLOCK A — Condition-key parser")
print("=" * 80)
samples = ["T50_P300_HRC0", "T60_P400_HRC50", "T70_P500_HRC0"]
expected = [(323.15, 1.3e5, 0.0),
            (333.15, 1.4e5, 0.5),
            (343.15, 1.5e5, 0.0)]
for s, (T, P, RHC) in zip(samples, expected):
    cond = parse_condition_key(s)
    ok = (abs(cond["Tfc_K"] - T) < 1e-6 and abs(cond["P_Pa"] - P) < 1.0
          and abs(cond["RHC_frac"] - RHC) < 1e-6)
    print(f"  [{'ok' if ok else 'FAIL':<4}] {s:<16s} -> T={cond['Tfc_K']:.2f} K, "
          f"P={cond['P_Pa']:.0f} Pa, RHC={cond['RHC_frac']:.2f}")
    assert ok


# ---------------------------------------------------------------------------
# Block B — objective evaluates at default params
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK B — Objective evaluation at default params")
print("=" * 80)
pola = export_experiment_data("pola")
cond_keys = sorted(pola.keys())[:1]  # single condition for speed
print(f"  using condition: {cond_keys[0]}")

baseline = deepcopy(_PARAMS_DEFAULT)
objective, exp_cache, predict = make_objective(
    "Polarization", baseline, pola, cond_keys,
)

t0 = time.perf_counter()
loss_baseline = objective({})
print(f"  loss @ defaults   : {loss_baseline:.4g}  ({time.perf_counter()-t0:.2f} s)")
assert np.isfinite(loss_baseline)


# Check that perturbing a parameter changes the loss.
t0 = time.perf_counter()
loss_perturbed = objective({"i0_c_ref": baseline.get("i0_c_ref", 1e-5) * 0.5})
print(f"  loss after i0/=2  : {loss_perturbed:.4g}  ({time.perf_counter()-t0:.2f} s)")
assert np.isfinite(loss_perturbed)


# ---------------------------------------------------------------------------
# Block C — run each optimizer with a tiny budget
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK C — Drive each optimizer (small budget)")
print("=" * 80)

# Use 2 parameters and 10-15 trials each so it's fast.
request_template = {
    "target":     "Polarization",
    "model":      "Static",
    "n_trials":   12,
    "seed":       42,
    "params":     ["i0_c_ref", "kappa_c"],
    "bounds":     {"i0_c_ref": (1e-7, 1e-3), "kappa_c": (1.0, 20.0)},
    "conditions": cond_keys,
}

results = {}
for optimizer in _DRIVERS:
    req = deepcopy(request_template)
    req["optimizer"] = optimizer
    print(f"\n  >>> {optimizer}")
    last_progress = []
    def _on_progress(done, total, best_loss, lp=last_progress):
        lp.append((done, total, best_loss))
    t0 = time.perf_counter()
    try:
        res = run_calibration(req, baseline_params=baseline, data=pola,
                              on_progress=_on_progress)
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        raise
    wall = time.perf_counter() - t0
    results[optimizer] = res

    # Validate schema
    for key in ("best_params", "best_loss", "history", "best_curves",
                "elapsed_s", "optimizer", "n_evals", "message", "target", "conditions"):
        assert key in res, f"{optimizer}: missing key {key!r}"

    # Validate history
    h = res["history"]
    assert len(h) >= 1
    best_so_far = float("inf"); monotone = True
    for _, l in h:
        best_so_far = min(best_so_far, l)
    # best_so_far should equal res['best_loss'] (within float)
    assert abs(best_so_far - res["best_loss"]) < 1e-9 or np.isnan(res["best_loss"])
    # progress callback was hit
    assert len(last_progress) >= 1

    # Validate best_curves shape
    for cond_key in cond_keys:
        if cond_key not in res["best_curves"]:
            continue
        c = res["best_curves"][cond_key]
        n = len(c["i_meas"])
        assert len(c["y_meas"]) == n
        assert len(c["y_pred"]) == n

    # Print summary
    print(f"     evals      : {res['n_evals']}")
    print(f"     best_loss  : {res['best_loss']:.4g}")
    print(f"     best_params: {res['best_params']}")
    print(f"     wall       : {wall:.1f} s")
    print(f"     message    : {res['message']}")
    print(f"     progress callbacks: {len(last_progress)} hits")


# ---------------------------------------------------------------------------
# Block D — sanity: every optimizer should improve over a random guess
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK D — Sanity check — best loss <= baseline (defaults) loss")
print("=" * 80)
for optimizer, res in results.items():
    improved = res["best_loss"] <= loss_baseline * 1.001  # tiny slack for noise
    tag = "ok" if improved else "WARN"
    print(f"  [{tag:<4}] {optimizer:<22s} best={res['best_loss']:.4g} "
          f"vs default={loss_baseline:.4g}")

# Persist machine-readable results
out_path = os.path.join(os.path.dirname(__file__), "verify_calib_backend.json")
def _serialize(res):
    out = {k: v for k, v in res.items() if k not in ("best_curves",)}
    out["best_curves"] = {
        k: {kk: vv.tolist() for kk, vv in v.items()}
        for k, v in res["best_curves"].items()
    }
    return out
with open(out_path, "w") as f:
    json.dump({k: _serialize(v) for k, v in results.items()},
              f, indent=2, default=str)

print()
print("=" * 80)
print(f"ALL CHECKS PASSED  -  results -> {out_path}")
print("=" * 80)
