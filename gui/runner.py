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


def run(params, op_inputs, model_variant, profile_func, t_span,
        max_step=0.1, method="BDF", polar_sweep=None):
    if model_variant == "Static":
        return _run_static(params, op_inputs, polar_sweep or {})

    p = dict(params)
    op = dict(op_inputs)
    op["current_density"] = profile_func

    t0 = time.perf_counter()
    if model_variant == "Dynamic":
        y0 = init_x(op, p)
        model = PEMFC_dyn(parameters=p, operating_inputs=op,
                          initial_variable_values=y0, time_interval=t_span)
        sol = solve_ivp(model.dxdt, t_span, y0, method=method, max_step=max_step)
        try:
            model._recovery(sol)
        except AttributeError:
            pass
    else:
        model = PEMFC(param=p, operating_inputs=op,
                      variable_names=solver_variable_names,
                      flux_names=solver_flux_names)
        y0 = init_x(op, p)
        sol = solve_ivp(model.dxdt, t_span, y0, method=method, max_step=max_step)
        model._recovery(sol)

    runtime = time.perf_counter() - t0
    status = {
        "runtime_s": runtime,
        "n_states": len(y0),
        "n_steps": len(sol.t),
        "success": bool(sol.success),
        "message": getattr(sol, "message", ""),
        "model_variant": model_variant,
        "kind": "transient",
    }
    return model, sol, status


def _run_static(params, op_inputs, polar_sweep):
    n_points = int(polar_sweep.get("n_points", 30))
    i_max_A_cm2 = float(polar_sweep.get("i_max_A_cm2", params.get("i_max_pola", 1.65e4) / 1e4))
    Aact = params.get("Aact", 31e-4)
    Sa = op_inputs.get("Sa", 1.2)
    Sc = op_inputs.get("Sc", 2.5)

    op = dict(op_inputs)
    F = 96485.0
    I_ref = i_max_A_cm2 * 1e4 * Aact
    op.setdefault("Win_a", Sa * I_ref / (2.0 * F))
    op.setdefault("Win_c", Sc * I_ref / (4.0 * F))
    op.setdefault("Wout_a", op["Win_a"])
    op.setdefault("Wout_c", op["Win_c"])

    p = dict(params)
    model = PEMFC_stat(parameters=p, operating_inputs=op)

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
