"""Simulation dispatcher.

Wraps `solve_ivp` + `model._recovery` for the dual-scale and dynamic models;
provides a polarization-sweep adapter for the static model. All inputs are
defensive copies so the global dicts in `config.initialize` stay clean across
repeated runs from the GUI.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp

from config.initialize import init_x, init_x_for
from config.settings import solver_variable_names, solver_flux_names
from model.coefficients import F, R, yO2_ext, Kshape, Psat, lambda_eq
from model.model import PEMFC, PEMFC_0D, PEMFC_dyn
from model.static import PEMFC_stat


MODEL_VARIANTS = ("Dual-scale", "Dynamic", "Static")


def _build_dyn_initial_state(params, op_inputs):
    """Thin wrapper kept for backwards-compatibility -- delegates to the
    canonical builder in ``config.initialize.init_x_for('dynamic', ...)``.

    The actual logic now lives in ``config/initialize.py::init_x_dyn`` so
    any callsite (notebooks, scripts, calibration loops) can get the
    correctly-sized initial state without duplicating the seed formulas
    here in the GUI layer.
    """
    return init_x_for('dynamic', op_inputs, params)


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


def _resolve_transient_model(model_variant, aux_system):
    """Pick the actual transient model class to run.

    All three dual-scale variants live in :mod:`model.model`:
      * ``PEMFC``      -- 1-D, WITHOUT balance-of-plant (no aux).
      * ``PEMFC_dyn``  -- 1-D, WITH compressor / BoP (with aux).
      * ``PEMFC_0D``   -- 0-D lumped (used as a comparison companion).

    The auxiliary-system toggle (not the "Model variant" radio) is what
    really selects the class. If the user picks a combination that's
    physically inconsistent, we silently route to the matching class and
    return a note for the status strip:

      Dual-scale + With aux   -> Dynamic   (PEMFC has no BoP code)
      Dynamic    + Without aux-> Dual-scale (PEMFC_dyn requires BoP;
                                             else solver hits a
                                             (181,) vs (218,) broadcast)
      Dual-scale + Without aux-> Dual-scale (no change)
      Dynamic    + With aux   -> Dynamic    (no change)
    """
    if aux_system and model_variant == "Dual-scale":
        return "Dynamic", "auto-promoted: PEMFC has no BoP, ran PEMFC_dyn instead"
    if (not aux_system) and model_variant == "Dynamic":
        return "Dual-scale", "auto-demoted: PEMFC_dyn requires BoP, ran PEMFC instead"
    return model_variant, None


def run(params, op_inputs, model_variant, profile_func, t_span,
        max_step=0.1, method="BDF", polar_sweep=None, aux_system=True):
    if model_variant == "Static":
        return _run_static(params, op_inputs, polar_sweep or {})

    requested_variant = model_variant
    model_variant, route_note = _resolve_transient_model(model_variant, aux_system)

    # Defensive copies — never mutate the dicts owned by st.session_state.
    params    = dict(params)
    op_inputs = dict(op_inputs)
    op_inputs["current_density"] = profile_func
    # Auxiliary-system toggle: the model's dxdt reads parameters["aux_system"]
    # and includes (True) or skips (False) the compressor / BoP equations.
    params["aux_system"] = bool(aux_system)

    t0 = time.perf_counter()
    if model_variant == "Dynamic":
        # ``init_x`` produces a dual-scale-shaped (218-element) state vector
        # that PEMFC_dyn cannot consume — its dxdt expects 181 elements.
        # Use the GUI-side builder so the model file stays untouched.
        y0 = _build_dyn_initial_state(params, op_inputs)
        model = PEMFC_dyn(parameters=params, operating_inputs=op_inputs,
                          initial_variable_values=y0, time_interval=t_span)
        sol, fallback = _solve_with_fallback(model.dxdt, t_span, y0, method, max_step)
        try:
            model._recovery(sol)
        except AttributeError:
            pass
        # Normalize PEMFC_dyn's data layout so the GUI plotting code (which
        # was written for PEMFC / PEMFC_0D) sees a uniform interface:
        #   * PEMFC_dyn stores derived electrochem in ``self.ec_kinetics``
        #     -- alias it as ``echem_traj`` for the results renderer.
        #   * PEMFC_dyn puts ``Ucell`` in ``self.variables`` -- also expose
        #     it in echem_traj so the Cell-performance tab can find it.
        if not hasattr(model, "echem_traj"):
            model.echem_traj = dict(getattr(model, "ec_kinetics", {}))
            if "Ucell" not in model.echem_traj and "Ucell" in model.variables:
                model.echem_traj["Ucell"] = model.variables["Ucell"]
            # Ensure a time axis exists in echem_traj for symmetry.
            if "t" not in model.echem_traj and "t" in model.variables:
                model.echem_traj["t"] = model.variables["t"]
    else:
        model = PEMFC(param=params, operating_inputs=op_inputs,
                      variable_names=solver_variable_names,
                      flux_names=solver_flux_names)
        y0 = init_x(op_inputs, params)
        sol, fallback = _solve_with_fallback(model.dxdt, t_span, y0, method, max_step)
        model._recovery(sol)

    runtime = time.perf_counter() - t0
    msg = getattr(sol, "message", "")
    if route_note:
        msg = (route_note + ". " + msg).strip(". ").strip() + ("." if msg else "")
    if fallback:
        msg = (f"BDF failed on a transient NaN; auto-retried with LSODA. "
               f"Solver message: {msg}").strip()
    aux_label = "aux: on" if aux_system else "aux: off"
    label = model_variant
    if model_variant != requested_variant:
        label = f"{model_variant} (from {requested_variant})"
    status = {
        "runtime_s": runtime,
        "n_states": len(y0),
        "n_steps": len(sol.t),
        "success": bool(sol.success),
        "message": msg,
        "model_variant": (label + f" ({aux_label})"
                          + (" → LSODA fallback" if fallback else "")),
        "kind": "transient",
        "aux_system": bool(aux_system),
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


# ===========================================================================
# 0D benchmark companion runs
# ===========================================================================
# When the user ticks "Compare with 0D benchmark" in the GUI, we run the
# 0D lumped-parameter model (``PEMFC_0D`` in model/model.py) alongside
# the main simulation so the results can be overlaid for comparison.
#
# Two modes:
#   * Transient: same params/op_inputs/profile_func/t_span as the main run.
#   * Polar:     sweep a constant-current profile through the 0D ODE for
#                each polarisation point and take the steady-state Ucell.
#
# The model file is NOT modified -- we only call its public API.
# ===========================================================================

def run_0d_companion(params, op_inputs, profile_func, t_span,
                     max_step=0.1, method="BDF", polar_sweep=None):
    """Return ``(variables, echem_traj, polar, status)`` from a 0D run.

    The status dict has the same shape as the main ``run()``'s status so
    the GUI can show it next to the 1D status. Failures are caught and
    reported -- a failing 0D companion never aborts the main result.
    """
    if polar_sweep is not None:
        return _run_0d_polar(params, op_inputs, polar_sweep)
    return _run_0d_transient(params, op_inputs, profile_func, t_span,
                             max_step, method)


def _run_0d_transient(params, op_inputs, profile_func, t_span, max_step, method):
    params    = dict(params)
    op_inputs = dict(op_inputs)
    op_inputs["current_density"] = profile_func

    t0 = time.perf_counter()
    try:
        model = PEMFC_0D(parameters=params, operating_inputs=op_inputs)
        y0 = model.default_initial_state(params, op_inputs)
        info = model.solve(t_span=t_span, y0=y0, method=method,
                           max_step=max_step, verbose=False, sparsity=False)
        sol = info["sol"] if isinstance(info, dict) and "sol" in info else info
        model._recovery(sol)
        runtime = time.perf_counter() - t0
        return {
            "variables":  {k: np.asarray(v).copy()
                           for k, v in model.variables.items()
                           if hasattr(v, "__len__") and not isinstance(v, (str, dict))},
            "echem_traj": {k: np.asarray(v).copy()
                           for k, v in model.echem_traj.items()
                           if hasattr(v, "__len__") and not isinstance(v, (str, dict))},
            "polar": None,
            "status": {
                "runtime_s": runtime,
                "n_steps":   len(sol.t),
                "success":   bool(sol.success),
                "message":   getattr(sol, "message", ""),
                "kind":      "transient",
            },
        }
    except Exception as exc:
        return {
            "variables": {}, "echem_traj": {}, "polar": None,
            "status": {"runtime_s": time.perf_counter() - t0,
                       "n_steps": 0, "success": False,
                       "message": f"0D companion failed: {exc}",
                       "kind": "transient"},
        }


def _run_0d_polar(params, op_inputs, polar_sweep):
    """Sweep the 0D model through constant-current points, take final Ucell.

    A short transient (t_span = (0, 30)) is plenty for the 0D electrochem
    to settle; the BoP dynamics aren't relevant to the steady polarisation
    curve, so we keep n_group_pt low for speed (matches the polar notebook).
    """
    n_points    = int(polar_sweep.get("n_points", 30))
    i_max_A_cm2 = float(polar_sweep.get("i_max_A_cm2",
                                        params.get("i_max_pola", 1.65e4) / 1e4))

    # Lighter Pt PSD for the polar sweep -- micro-scale details don't
    # affect steady-state Ucell and the runtime drops by ~10x.
    params = dict(params)
    params.setdefault("n_group_pt", 10)
    if params["n_group_pt"] > 10:
        params["n_group_pt"] = 10

    i_grid = np.linspace(0.05e4, i_max_A_cm2 * 1e4, n_points)
    Ucell_pts, i_keep = [], []
    msg_parts = []
    t0 = time.perf_counter()

    for i_val in i_grid:
        op = dict(op_inputs)
        op["current_density"] = (lambda t, _i=float(i_val): _i)
        try:
            model = PEMFC_0D(parameters=params, operating_inputs=op)
            y0 = model.default_initial_state(params, op)
            info = model.solve(t_span=(0.0, 30.0), y0=y0, method="BDF",
                               max_step=0.1, verbose=False, sparsity=False)
            sol = info["sol"] if isinstance(info, dict) and "sol" in info else info
            model._recovery(sol)
            u_last = float(model.echem_traj["Ucell"][-1])
            if np.isfinite(u_last):
                Ucell_pts.append(u_last)
                i_keep.append(float(i_val))
        except Exception as exc:
            msg_parts.append(f"i={i_val:.0f}: {exc}")
            continue

    runtime = time.perf_counter() - t0
    polar = {"i_A_m2": np.array(i_keep), "Ucell_V": np.array(Ucell_pts)}
    return {
        "variables": {}, "echem_traj": {}, "polar": polar,
        "status": {
            "runtime_s": runtime,
            "n_points":  len(i_keep),
            "success":   len(i_keep) > 0,
            "message":   ("; ".join(msg_parts) if msg_parts
                          else f"swept {len(i_keep)} of {n_points} points"),
            "kind":      "polar",
        },
    }
