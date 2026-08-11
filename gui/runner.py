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


# Names editable through the GUI's "Micro-scale CL" parameter group.
_KINETIC_CONST_NAMES = ("k1", "k1_ref", "k2", "k2_ref", "k3",
                        "krdp", "k4", "k5", "kdet_ref")


def apply_kinetic_consts(kinetic_consts):
    """Push the GUI-edited Pt-surface rate constants into the model modules.

    The constants are module-level globals in ``model/coefficients.py``,
    and every consumer (`kinetic_eq`, `state_eq`, `model`) star-imports
    them -- meaning each module holds its own binding. Setting the value
    on all four modules makes the edit effective regardless of which
    module's function reads it at solve time. Model files themselves are
    never modified.
    """
    if not kinetic_consts:
        return
    import model.coefficients as _coeffs
    import model.kinetic_eq as _kin
    import model.state_eq as _steq
    import model.model as _mdl
    for name in _KINETIC_CONST_NAMES:
        if name not in kinetic_consts:
            continue
        val = float(kinetic_consts[name])
        for mod in (_coeffs, _kin, _steq, _mdl):
            if hasattr(mod, name):
                setattr(mod, name, val)


def _guarded_dxdt(model):
    """Wrap ``model.dxdt`` with a positive floor on physically non-negative
    states (saturations ``s_*``, concentrations ``C_*``, water content
    ``lambda_*``) for the DERIVATIVE EVALUATION only.

    Implicit solvers routinely probe trial states slightly outside the
    physical domain; a marginally negative saturation or O2 concentration
    then hits fractional powers ((C_ref/C)**kappa_c) or divisions in the
    kinetics and produces NaN, killing the whole integration in one step.
    Flooring the *read* values at a tiny epsilon keeps every evaluation
    finite without modifying the model files or the solver state itself.
    """
    names = getattr(model, "variable_names", None) or getattr(
        model, "solver_variable_names", [])
    guard_idx = np.array([i for i, n in enumerate(names)
                          if n.startswith(("s_", "C_", "lambda_"))], dtype=int)
    if guard_idx.size == 0:
        return model.dxdt

    def dxdt(t, y):
        yc = np.array(y, dtype=float, copy=True)
        yc[guard_idx] = np.maximum(yc[guard_idx], 1e-6)
        return model.dxdt(t, yc)

    return dxdt


def _progress_rhs(dxdt, t_span, progress_callback):
    """Return an RHS wrapper that reports monotonically increasing time.

    The ODE solvers can evaluate earlier trial times while constructing an
    implicit step, so reporting every call would make a progress bar move
    backwards.  Emit at most 1,001 updates (0.1% increments) instead and
    leave the numerical state and derivative untouched.
    """
    if progress_callback is None:
        return dxdt

    start, end = (float(t_span[0]), float(t_span[1]))
    duration = end - start
    last_bucket = [-1]

    def wrapped(t, y):
        if duration > 0.0:
            fraction = float(np.clip((float(t) - start) / duration, 0.0, 1.0))
        else:
            fraction = 1.0
        bucket = min(1000, int(fraction * 1000.0))
        if bucket > last_bucket[0]:
            last_bucket[0] = bucket
            try:
                progress_callback(fraction, float(t))
            except Exception:
                # A display problem must never abort a physical simulation.
                pass
        return dxdt(t, y)

    return wrapped


def _finish_progress(progress_callback, sol):
    """Complete an attached progress display after a successful solve."""
    if (progress_callback is None or sol is None
            or not bool(getattr(sol, "success", False))):
        return
    final_time = float(sol.t[-1]) if len(sol.t) else 0.0
    try:
        progress_callback(1.0, final_time)
    except Exception:
        pass


def _solve_with_fallback(dxdt, t_span, y0, method, max_step,
                         solve_kwargs=None, progress_callback=None):
    """Run ``solve_ivp`` with the automatic BDF -> LSODA solver chain.

    Tries ``method`` first (BDF by default). If that attempt fails for
    ANY reason -- Python exception raised, ``sol.success == False``
    (LSODA's ``Unexpected istate`` message, BDF's Jacobian NaN, etc.),
    zero steps taken, or a non-finite final state -- the routine
    automatically retries the integration with LSODA. If both methods
    fail, returns the LSODA result (with ``success=False``) so the
    caller can surface the solver message to the user.

    Returns ``(sol, fallback_used)``. ``fallback_used = True`` means
    LSODA finished the integration after the primary method could not.
    """
    solve_kwargs = dict(solve_kwargs or {})

    def _attempt(m):
        attempt_kwargs = dict(solve_kwargs)
        if m not in ("BDF", "RADAU"):
            attempt_kwargs.pop("jac_sparsity", None)
        try:
            sol = solve_ivp(
                _progress_rhs(dxdt, t_span, progress_callback),
                t_span, y0, method=m, max_step=max_step,
                **attempt_kwargs,
            )
        except Exception:
            return None
        return sol

    def _is_valid(sol):
        return (sol is not None
                and bool(sol.success)
                and sol.y.shape[1] > 0
                and bool(np.all(np.isfinite(sol.y[:, -1]))))

    primary = (method or "BDF").upper()
    sol_primary = _attempt(primary)
    if _is_valid(sol_primary):
        return sol_primary, False

    # Primary was already LSODA -- no second-order solver in the chain.
    if primary == "LSODA":
        return sol_primary, False

    sol_lsoda = _attempt("LSODA")
    if _is_valid(sol_lsoda):
        return sol_lsoda, True

    # Both failed: prefer the LSODA sol (has the more informative message).
    return sol_lsoda if sol_lsoda is not None else sol_primary, True


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
        max_step=0.1, method="BDF", polar_sweep=None, aux_system=True,
        kinetic_consts=None, progress_callback=None):
    # Apply the GUI-edited Pt-surface rate constants ("Micro-scale CL"
    # group) before any model object is built.
    apply_kinetic_consts(kinetic_consts)

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
        sol, fallback = _solve_with_fallback(
            _guarded_dxdt(model), t_span, y0, method, max_step,
            progress_callback=progress_callback,
        )
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
        solve_kwargs = {"atol": 1e-4}
        if (method or "BDF").upper() in ("BDF", "RADAU"):
            solve_kwargs["jac_sparsity"] = model.jac_sparsity(y0)
        sol, fallback = _solve_with_fallback(
            model.dxdt, t_span, y0, method, max_step,
            solve_kwargs=solve_kwargs,
            progress_callback=progress_callback,
        )
        model._recovery(sol)

    _finish_progress(progress_callback, sol)

    runtime = time.perf_counter() - t0
    msg = getattr(sol, "message", "")
    if route_note:
        msg = (route_note + ". " + msg).strip(". ").strip() + ("." if msg else "")
    if fallback:
        msg = (f"{method} could not finish the integration; auto-switched "
               f"to LSODA. Solver message: {msg}").strip()
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
        # Dedicated flag so the results panel can surface a prominent
        # notice when the auto-switch fired. The calibration path never
        # sets this — it uses solve_ivp directly with the chosen method.
        "fallback_used": bool(fallback),
        "method_requested": method,
        "method_actual":    "LSODA" if fallback else method,
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

    # ---- Input validation (no silent overrides) ----------------------------
    # The algebraic water balance in PEMFC_stat needs at least a trace of
    # humidification — a perfectly dry anode/cathode boundary makes a_w go
    # to zero and the closed-form lambda(a_w) expression diverges. Tell the
    # user *exactly* which input is incompatible instead of silently
    # bumping their value (the GUI rule is: never run a different setup
    # than the one the user typed in).
    phi_a = float(op_inputs.get("Phi_a_des", 0.0))
    phi_c = float(op_inputs.get("Phi_c_des", 0.0))
    if phi_a < 0.05 or phi_c < 0.05:
        return None, {"i_A_m2": np.array([]), "Ucell_V": np.array([])}, {
            "runtime_s": 0.0, "n_points": 0,
            "success": False,
            "message": (
                f"Static (algebraic) solver requires Phi_a_des and Phi_c_des "
                f"both ≥ 0.05 — currently Phi_a_des={phi_a:.3f}, "
                f"Phi_c_des={phi_c:.3f}. Edit the Operating parameters in "
                f"section §1 and re-run."
            ),
            "model_variant": "Static",
            "kind": "polar",
        }

    # ---- Fix the missing module-level globals (code defect, not a UI lie).
    # PEMFC_stat internally reads Pa_des / Pc_des from `model.static` module
    # globals (see model/static.py — line `Pa_des, Pc_des` defaults at import).
    # The runner never synced them, so any non-default pressure in the GUI
    # was effectively ignored by the static solver. Sync them now so the
    # GUI's pressure inputs are actually honored.
    import model.static as _static_module
    _static_module.Pa_des = op_inputs["Pa_des"]
    _static_module.Pc_des = op_inputs["Pc_des"]

    model = PEMFC_stat(parameters=params, operating_inputs=op_inputs)

    i_grid = np.linspace(0.05e4, i_max_A_cm2 * 1e4, n_points)
    Ucell, i_keep = [], []
    failures = 0
    t0 = time.perf_counter()
    for i in i_grid:
        try:
            res = model.solve(float(i))
            v = (res["Ueq"] - res["eta_c"]
                 - i * (res["Rohm"] + res["Rccl"] + res["Racl"]))
            if np.isfinite(v):
                Ucell.append(float(v))
                i_keep.append(float(i))
            else:
                failures += 1
        except Exception:
            failures += 1
            continue
    runtime = time.perf_counter() - t0

    polar = {"i_A_m2": np.array(i_keep), "Ucell_V": np.array(Ucell)}
    msg = ""
    if not i_keep:
        msg = "static solver failed at every i"
    elif failures:
        msg = f"{failures} of {n_points} points did not converge"
    status = {
        "runtime_s": runtime,
        "n_points": len(i_keep),
        "success": len(i_keep) > 0,
        "message": msg,
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
                     max_step=0.1, method="BDF", polar_sweep=None,
                     progress_callback=None):
    """Return ``(variables, echem_traj, polar, status)`` from a 0D run.

    The status dict has the same shape as the main ``run()``'s status so
    the GUI can show it next to the 1D status. Failures are caught and
    reported -- a failing 0D companion never aborts the main result.
    """
    if polar_sweep is not None:
        return _run_0d_polar(params, op_inputs, polar_sweep)
    return _run_0d_transient(params, op_inputs, profile_func, t_span,
                             max_step, method, progress_callback)


def _run_0d_transient(params, op_inputs, profile_func, t_span, max_step, method,
                      progress_callback=None):
    params    = dict(params)
    op_inputs = dict(op_inputs)
    op_inputs["current_density"] = profile_func

    t0 = time.perf_counter()
    try:
        model = PEMFC_0D(parameters=params, operating_inputs=op_inputs)
        y0 = model.default_initial_state(params, op_inputs)
        original_dxdt = model.dxdt
        model.dxdt = _progress_rhs(original_dxdt, t_span, progress_callback)
        try:
            info = model.solve(t_span=t_span, y0=y0, method=method,
                               max_step=max_step, verbose=False, sparsity=False)
        finally:
            model.dxdt = original_dxdt
        sol = info["sol"] if isinstance(info, dict) and "sol" in info else info
        _finish_progress(progress_callback, sol)
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
