"""Simulation dispatcher.

Wraps `solve_ivp` + `model._recovery` for the dual-scale and dynamic models;
provides a polarization-sweep adapter for the static model. All inputs are
defensive copies so the global dicts in `config.initialize` stay clean across
repeated runs from the GUI.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp

from config.initialize import init_x
from config.settings import solver_variable_names, solver_flux_names
from model.dualscale import PEMFC
from model.dynamic import PEMFC_dyn
from model.static import PEMFC_stat


MODEL_VARIANTS = ("Dual-scale", "Dynamic", "Static")


def _solve_with_fallback(dxdt, t_span, y0, method, max_step):
    """Run solve_ivp; on transient-NaN failure (newer scipy is strict),
    fall back to LSODA which tolerates the same intermediate NaNs that
    older scipy silently survives. Returns ``(sol, fallback_used)``.
    """
    try:
        sol = solve_ivp(dxdt, t_span, y0, method=method, max_step=max_step)
        return sol, False
    except ValueError as exc:
        msg = str(exc).lower()
        if "inf" not in msg and "nan" not in msg:
            raise
        if method.upper() == "LSODA":
            raise
    sol = solve_ivp(dxdt, t_span, y0, method="LSODA", max_step=max_step)
    return sol, True


def run(params, op_inputs, model_variant, profile_func, t_span,
        max_step=0.1, method="BDF", polar_sweep=None):
    if model_variant == "Static":
        return _run_static(params, op_inputs, polar_sweep or {})

    # Defensive copies — never mutate the dicts owned by st.session_state.
    params    = dict(params)
    op_inputs = dict(op_inputs)
    op_inputs["current_density"] = profile_func

    t0 = time.perf_counter()
    if model_variant == "Dynamic":
        y0 = init_x(op_inputs, params)
        model = PEMFC_dyn(parameters=params, operating_inputs=op_inputs,
                          initial_variable_values=y0, time_interval=t_span)
        sol, fallback = _solve_with_fallback(model.dxdt, t_span, y0, method, max_step)
        try:
            model._recovery(sol)
        except AttributeError:
            pass
    else:
        model = PEMFC(param=params, operating_inputs=op_inputs,
                      variable_names=solver_variable_names,
                      flux_names=solver_flux_names)
        y0 = init_x(op_inputs, params)
        sol, fallback = _solve_with_fallback(model.dxdt, t_span, y0, method, max_step)
        model._recovery(sol)

    runtime = time.perf_counter() - t0
    msg = getattr(sol, "message", "")
    if fallback:
        msg = (f"BDF failed on a transient NaN; auto-retried with LSODA. "
               f"Solver message: {msg}").strip()
    status = {
        "runtime_s": runtime,
        "n_states": len(y0),
        "n_steps": len(sol.t),
        "success": bool(sol.success),
        "message": msg,
        "model_variant": model_variant + (" → LSODA fallback" if fallback else ""),
        "kind": "transient",
    }
    return model, sol, status


def _run_static(params, op_inputs, polar_sweep):
    n_points = int(polar_sweep.get("n_points", 30))
    i_max_A_cm2 = float(polar_sweep.get("i_max_A_cm2", params.get("i_max_pola", 1.65e4) / 1e4))
    Aact = params.get("Aact", 31e-4)
    Sa = op_inputs.get("Sa", 1.2)
    Sc = op_inputs.get("Sc", 2.5)

    # Defensive copies + derived inlet/outlet flows the static solver needs.
    params    = dict(params)
    op_inputs = dict(op_inputs)
    F = 96485.0
    I_ref = i_max_A_cm2 * 1e4 * Aact
    op_inputs.setdefault("Win_a",  Sa * I_ref / (2.0 * F))
    op_inputs.setdefault("Win_c",  Sc * I_ref / (4.0 * F))
    op_inputs.setdefault("Wout_a", op_inputs["Win_a"])
    op_inputs.setdefault("Wout_c", op_inputs["Win_c"])

    model = PEMFC_stat(parameters=params, operating_inputs=op_inputs)

    i_grid = np.linspace(0.05e4, i_max_A_cm2 * 1e4, n_points)
    Ucell, i_keep = [], []
    t0 = time.perf_counter()
    for i in i_grid:
        try:
            res = model.solve(float(i))
            v = (res["Ueq"] - res["eta_c"]
                 - i * (res["Rohm"] + res["Rccl"] + res["Racl"]))
            if np.isfinite(v):
                Ucell.append(float(v))
                i_keep.append(float(i))
        except Exception:
            continue
    runtime = time.perf_counter() - t0

    polar = {"i_A_m2": np.array(i_keep), "Ucell_V": np.array(Ucell)}
    status = {
        "runtime_s": runtime,
        "n_points": len(i_keep),
        "success": len(i_keep) > 0,
        "message": "" if i_keep else "static solver failed at every i",
        "model_variant": "Static",
        "kind": "polar",
    }
    return model, polar, status
