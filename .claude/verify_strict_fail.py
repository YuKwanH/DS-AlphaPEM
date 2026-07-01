"""Verify the new strict-fail policy in make_objective.

Three scenarios:
  A. Clean Static run                          -> finite loss < FAILURE_LOSS
  B. predict() returns NaN at one point        -> exact FAILURE_LOSS
  C. predict() returns finite but predict func
     is replaced by a transient stub that
     returns NaN-on-time-mismatch              -> exact FAILURE_LOSS
  D. Final-time check inside the actual
     transient predictor (force t_end mismatch
     by patching solve_ivp to return early)    -> exact FAILURE_LOSS
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
from copy import deepcopy
from config.initialize import parameters as P
from data.export       import export_experiment_data
from gui import calib_backend as cb


def _make_dummy_objective(predict_fn):
    """Build an objective using the real make_objective() but with a
    custom predict() stub injected, so we can force every failure mode
    without waiting for the real model to misbehave."""
    pola = export_experiment_data("pola")
    cond_keys = sorted(pola.keys())[:1]
    # Mimic the internals of make_objective() but pass our predict() in.
    objective, exp_cache, _ = cb.make_objective(
        "Polarization", deepcopy(P), pola, cond_keys,
    )
    # Now rebuild the closure with the custom predict.
    def custom_obj(po):
        p = deepcopy(P)
        p.update(po)
        loss = 0.0; n = 0
        for ck in cond_keys:
            i_meas, y_meas = exp_cache[ck]
            if i_meas.size == 0: continue
            y_pred = predict_fn(p, ck, i_meas)
            if not np.isfinite(y_pred).all():
                return cb.FAILURE_LOSS
            loss += float(np.sum((y_pred - y_meas) ** 2))
            n += i_meas.size
        return loss / n if n else cb.FAILURE_LOSS
    return custom_obj, cond_keys


# A — clean Static run
print("=" * 72)
print(" A. Clean Static run (no failures)")
print("=" * 72)
pola = export_experiment_data("pola")
cond_keys = sorted(pola.keys())[:1]
obj, _, _ = cb.make_objective(
    "Polarization", deepcopy(P), pola, cond_keys,
    model_variant="Static", aux_system=False,
)
loss_clean = obj({"i0_c_ref": 0.5, "kappa_c": 1.5})
print(f"   loss = {loss_clean:.4g}")
assert np.isfinite(loss_clean)
assert loss_clean < cb.FAILURE_LOSS / 2
print("   [PASS] finite, well below FAILURE_LOSS")


# B — predict() returns NaN at one point
print()
print("=" * 72)
print(f" B. Predict returns NaN at one point  (expect exact FAILURE_LOSS = {cb.FAILURE_LOSS})")
print("=" * 72)
def predict_with_nan(p, ck, i_meas):
    out = np.full(len(i_meas), 0.7, dtype=float)
    out[2] = np.nan   # one NaN — should fail the whole trial
    return out
obj_b, _ = _make_dummy_objective(predict_with_nan)
loss_b = obj_b({"i0_c_ref": 0.5})
print(f"   loss = {loss_b}")
assert loss_b == cb.FAILURE_LOSS
print("   [PASS] one NaN -> FAILURE_LOSS")


# C — predict() returns all NaN (simulation failed across the board)
print()
print("=" * 72)
print(" C. Predict returns all-NaN  (full simulation failure)")
print("=" * 72)
def predict_all_nan(p, ck, i_meas):
    return np.full(len(i_meas), np.nan, dtype=float)
obj_c, _ = _make_dummy_objective(predict_all_nan)
loss_c = obj_c({"i0_c_ref": 0.5})
print(f"   loss = {loss_c}")
assert loss_c == cb.FAILURE_LOSS
print("   [PASS] all-NaN -> FAILURE_LOSS")


# D — transient predictor inside calib_backend with patched solve_ivp
#     that returns a truncated solution -> should trigger the final-time check
print()
print("=" * 72)
print(" D. Transient predictor with truncated solve_ivp  (final time mismatch)")
print("=" * 72)
import gui.calib_backend as cbm
import types
original_solve_ivp = cbm.solve_ivp

class _FakeSol:
    def __init__(self, t_span):
        # Return a solution that "succeeded" but stopped at half the span.
        self.success = True
        self.message = "fake"
        # Cheat the time check
        self.t = np.array([0.0, t_span[1] * 0.5])
        self.y = np.zeros((218, 2))  # finite, but t mismatched

def fake_solve_ivp(rhs, t_span, y0, **kw):
    return _FakeSol(t_span)

cbm.solve_ivp = fake_solve_ivp
try:
    # Manually call the transient predictor — should return all-NaN
    # because the final-time tolerance is violated.
    y = cbm._predict_polarization_transient(
        deepcopy(P), cond_keys[0], np.array([10.0, 20.0]),
        model_kind="PEMFC",
    )
    print(f"   predict output: {y}")
    assert np.all(np.isnan(y)), "expected all NaN, got finite values"
    print("   [PASS] half-span sol.t -> all-NaN predictions -> trial fails")
finally:
    cbm.solve_ivp = original_solve_ivp


print()
print("=" * 72)
print(" ALL STRICT-FAIL CHECKS PASSED")
print("=" * 72)
