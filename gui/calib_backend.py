"""Backend optimizers for the calibration GUI.

Three optimizer drivers (TPE / Genetic / Grid) and a small objective
library that fits the static PEMFC model against experimental
Polarization and HFR data.

Calibration model
-----------------
The fitting runs against ``PEMFC_stat`` (algebraic, no transient) for
speed — every objective evaluation is sub-second per condition. The
condition key encodes (T_C, P_dPa, RHC_pct); we recover the operating
inputs the static module needs (Phi_a/c, Pa/c, Tfc, Win/Wout) using the
recipe from `.claude/patch_hfr.py` (LEV-200 commercial stack).

Each objective returns the *sum of squared residuals* between simulated
and measured points across the selected conditions, summed over the
common test currents. Lower is better.

Backends
--------
* **TPE (Optuna)** — Bayesian, sample-efficient (recommended).
* **Genetic algorithm** — ``scipy.optimize.differential_evolution`` with
  ``init="sobol"``, which is GA-family (population + mutation +
  crossover + selection); chosen so it is single-package and well
  tested.
* **Grid search** — uniform sweep, ``floor(n_trials ** (1/k))`` points
  per axis; capped to the trial budget.

Each driver returns the same shape::

    {
        "best_params":   {name: value, ...},
        "best_loss":     float,
        "history":       [(trial_idx, loss), ...],   # convergence curve
        "best_curves":   {cond_key: {"i_meas": np.ndarray,
                                       "v_meas": np.ndarray,
                                       "v_pred": np.ndarray}},
        "elapsed_s":     float,
        "optimizer":     "TPE (Optuna)" | "Genetic algorithm" | "Grid search",
        "n_evals":       int,
        "message":       str,
    }
"""
import math
import re
import time
import warnings
from copy import deepcopy
from itertools import product

import numpy as np

from model.static import PEMFC_stat
import model.static as static_module


# ---------------------------------------------------------------------------
# Condition-key parser  (e.g. "T50_P400_HRC50" -> 50 °C, 1.4e5 Pa, 50 % RH)
# ---------------------------------------------------------------------------
_COND_RX = re.compile(r"^T(\d+)_P(\d+)_HRC(\d+)$")


def parse_condition_key(key):
    """Return dict ``{Tfc_K, P_Pa, RHC_frac}`` from a key like ``T50_P400_HRC0``.

    ``P{NNN}`` encodes pressure as ``(NNN + 1000) * 100 Pa`` per the
    convention in ``data/Polar_curves.xlsx`` — so ``P300 == 1.3e5 Pa``.
    """
    m = _COND_RX.match(key)
    if not m:
        raise ValueError(f"Cannot parse condition key {key!r}")
    T_C, P_code, RHC_pct = (int(g) for g in m.groups())
    return dict(
        Tfc_K=float(T_C + 273.15),
        P_Pa=float((P_code + 1000) * 100),
        RHC_frac=float(RHC_pct) / 100.0,
        _label=key,
    )


def _make_op_stat(cond):
    """Build the ``operating_inputs`` dict the static solver needs.

    Mirrors the recipe used in `.claude/patch_hfr.py` so the static
    algebraic water balance converges (the GUI's `_run_static` defaults
    are too dry — see the audit report for details).
    """
    P = cond["P_Pa"]
    # Tell static-module globals which pressure to use (read internally).
    static_module.Pa_des = P
    static_module.Pc_des = P
    return {
        "Tfc":       cond["Tfc_K"],
        "Phi_a_des": cond["RHC_frac"],
        "Phi_c_des": cond["RHC_frac"],
        "Pa_des":    P,
        "Pc_des":    P,
        "Win_c":     4.8,
        "Wout_c":   21.0,
        "Win_a":     4.8,
        "Wout_a":    4.8,
    }


# ---------------------------------------------------------------------------
# Predict-and-residual machinery
# ---------------------------------------------------------------------------
N_CELL_DEFAULT = 20  # LEV-200 stack default


def _predict_polarization(params, cond_key, i_meas_A):
    """Run ``PEMFC_stat`` for each measured current; return per-cell Ucell array.

    Any non-finite result is replaced with NaN; the caller's loss
    function should treat NaN as "this trial failed".
    """
    cond  = parse_condition_key(cond_key)
    op    = _make_op_stat(cond)
    Aact  = params.get("Aact", 31e-4)
    model = PEMFC_stat(params, op)

    out = np.full(len(i_meas_A), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, I in enumerate(i_meas_A):
            i = float(I) / Aact
            try:
                sol = model.solve(i)
            except Exception:
                continue
            ucell = sol["Ueq"] - sol["eta_c"] - i * (sol["Rohm"] + sol["Rccl"] + sol["Racl"])
            if math.isfinite(ucell):
                out[k] = ucell
    return out


def _predict_hfr(params, cond_key, i_meas_A):
    """Run ``PEMFC_stat`` for each measured current; return Rohm array."""
    cond  = parse_condition_key(cond_key)
    op    = _make_op_stat(cond)
    Aact  = params.get("Aact", 31e-4)
    model = PEMFC_stat(params, op)

    out = np.full(len(i_meas_A), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, I in enumerate(i_meas_A):
            i = float(I) / Aact
            try:
                sol = model.solve(i)
            except Exception:
                continue
            r = sol.get("Rohm")
            if r is not None and math.isfinite(r):
                out[k] = float(r)
    return out


def _experimental_polarization(data, cond_key, n_cell):
    df = data[cond_key].dropna(subset=["I_LOAD", "VFC"]).sort_values("I_LOAD")
    return (df["I_LOAD"].to_numpy(dtype=float),
            df["VFC"].to_numpy(dtype=float) / n_cell)


def _experimental_hfr(data, cond_key):
    df = data[cond_key].copy()
    import ast as _ast

    def _parse(cell):
        if isinstance(cell, (int, float)):
            return float(cell)
        try:
            val = _ast.literal_eval(str(cell))
        except (ValueError, SyntaxError):
            return float("nan")
        return float(val[0]) if isinstance(val, tuple) else float(val)

    df["R_real"] = df["R"].apply(_parse)
    df = df.dropna(subset=["I_LOAD", "R_real"]).sort_values("I_LOAD")
    return (df["I_LOAD"].to_numpy(dtype=float),
            df["R_real"].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------
FAILURE_LOSS = 1e6


def make_objective(target, baseline_params, data, conditions, *,
                   n_cell=N_CELL_DEFAULT):
    """Build an objective ``f(param_dict) -> loss``.

    Only the keys in ``param_dict`` override the baseline; every other
    parameter stays at its default. ``data`` is the dict returned by
    ``data.export.export_experiment_data``.
    """
    target = str(target).lower()
    if target.startswith("polar"):
        predict = _predict_polarization
        getexp  = lambda k: _experimental_polarization(data, k, n_cell)
    elif target.startswith("hfr"):
        predict = _predict_hfr
        getexp  = lambda k: _experimental_hfr(data, k)
    else:
        raise ValueError(f"target {target!r} not supported (use 'Polarization' or 'HFR')")

    # Cache experimental arrays so we only pay the I/O once.
    exp_cache = {k: getexp(k) for k in conditions}

    def objective(param_overrides):
        p = deepcopy(baseline_params)
        for k, v in param_overrides.items():
            p[k] = v
        loss = 0.0
        n = 0
        for cond_key in conditions:
            i_meas, y_meas = exp_cache[cond_key]
            if i_meas.size == 0:
                continue
            y_pred = predict(p, cond_key, i_meas)
            if not np.isfinite(y_pred).all():
                return FAILURE_LOSS
            loss += float(np.sum((y_pred - y_meas) ** 2))
            n += i_meas.size
        if n == 0:
            return FAILURE_LOSS
        return loss / n  # mean squared error

    return objective, exp_cache, predict


# ---------------------------------------------------------------------------
# Optimizer drivers
# ---------------------------------------------------------------------------
def _is_log_param(name):
    """Default to log-scale for parameters whose canonical range spans
    multiple decades. Falls back to linear otherwise."""
    # Names mirror gui/calibration.py::CALIB_PARAMS (the entries where the
    # 4th tuple field is True).
    log_set = {"i0_c_ref", "Re", "Hgdl", "Hcl", "a_slim", "Hmem"}
    return name in log_set


def _run_tpe(objective, params, bounds, n_trials, seed, on_progress=None):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "TPE requires the `optuna` package. Install it with "
            "`pip install optuna`, then restart Streamlit."
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    history = []

    def _opt(trial):
        po = {}
        for k in params:
            lo, hi = bounds[k]
            po[k] = trial.suggest_float(k, lo, hi, log=_is_log_param(k))
        loss = objective(po)
        history.append((len(history) + 1, loss))
        if on_progress is not None:
            on_progress(len(history), n_trials,
                        min(l for _, l in history))
        return loss

    study.optimize(_opt, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": dict(study.best_params),
        "best_loss":   float(study.best_value),
        "history":     history,
        "n_evals":     len(history),
        "message":     f"TPE completed {len(history)} trials.",
    }


def _run_ga(objective, params, bounds, n_trials, seed, on_progress=None):
    """Single-objective evolutionary search via ``scipy.optimize.differential_evolution``.

    DE is GA-family (population + mutation + crossover + selection); we
    pick it so the optimizer is single-package and well tested.
    """
    from scipy.optimize import differential_evolution

    history = []

    def _f(x):
        po = {k: float(v) for k, v in zip(params, x)}
        loss = float(objective(po))
        history.append((len(history) + 1, loss))
        if on_progress is not None:
            on_progress(len(history), n_trials,
                        min(l for _, l in history))
        return loss

    rng = np.random.default_rng(seed)
    popsize = max(5, min(15, n_trials // 5))
    maxiter = max(1, n_trials // popsize)
    boundlist = [bounds[k] for k in params]
    result = differential_evolution(
        _f, boundlist, popsize=popsize, maxiter=maxiter, seed=int(seed),
        init="sobol", tol=1e-6, mutation=(0.5, 1.5), recombination=0.7,
        polish=False, updating="deferred",
    )
    best_params = {k: float(v) for k, v in zip(params, result.x)}
    return {
        "best_params": best_params,
        "best_loss":   float(result.fun),
        "history":     history,
        "n_evals":     len(history),
        "message":     f"GA finished after {len(history)} evals (popsize={popsize}, maxiter={maxiter}).",
    }


def _run_grid(objective, params, bounds, n_trials, seed, on_progress=None):
    """Uniform grid; ``n_per_axis = floor(n_trials ** (1/k))`` (≥ 2)."""
    k = len(params)
    n_per_axis = max(2, int(n_trials ** (1.0 / max(1, k))))
    # Build axes (linear for now; could log-space for known log params).
    axes = {}
    for name in params:
        lo, hi = bounds[name]
        if _is_log_param(name) and lo > 0:
            axes[name] = np.geomspace(lo, hi, n_per_axis)
        else:
            axes[name] = np.linspace(lo, hi, n_per_axis)
    grid = list(product(*[axes[name] for name in params]))
    if len(grid) > n_trials:
        # Even subsample so we don't blow the budget.
        idx = np.linspace(0, len(grid) - 1, n_trials, dtype=int)
        grid = [grid[i] for i in idx]

    history = []
    best_loss = float("inf")
    best_params = None
    for vals in grid:
        po = {k_: float(v) for k_, v in zip(params, vals)}
        loss = float(objective(po))
        history.append((len(history) + 1, loss))
        if loss < best_loss:
            best_loss = loss
            best_params = po
        if on_progress is not None:
            on_progress(len(history), len(grid),
                        min(l for _, l in history))
    return {
        "best_params": best_params or {},
        "best_loss":   best_loss,
        "history":     history,
        "n_evals":     len(history),
        "message":     f"Grid swept {len(history)} of {n_per_axis ** k} candidate points "
                       f"({n_per_axis} per axis).",
    }


_DRIVERS = {
    "TPE (Optuna)":      _run_tpe,
    "Genetic algorithm": _run_ga,
    "Grid search":       _run_grid,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_calibration(request, *, baseline_params, data, on_progress=None):
    """Drive the requested optimizer to completion.

    ``request`` is the dict the Start button posts to ``state["calib_request"]``.
    ``data`` is the experimental-data dict from
    ``data.export.export_experiment_data`` matching ``request['target']``.
    ``on_progress(done, total, best_loss)`` is called once per evaluation
    so the GUI can update a progress bar.

    Returns the unified result dict described in the module docstring.
    """
    target     = request["target"]
    params     = list(request["params"])
    bounds     = {k: tuple(request["bounds"][k]) for k in params}
    optimizer  = request["optimizer"]
    n_trials   = int(request["n_trials"])
    seed       = int(request["seed"])
    conditions = list(request.get("conditions") or [])

    if not conditions:
        # Fall back to every available condition.
        conditions = list(data.keys())
    if not params:
        raise ValueError("Calibration request has no parameters to fit.")

    objective, exp_cache, predict = make_objective(
        target, baseline_params, data, conditions
    )

    driver = _DRIVERS.get(optimizer)
    if driver is None:
        raise ValueError(f"Unknown optimizer {optimizer!r} "
                         f"(expected one of {list(_DRIVERS)})")

    t0 = time.perf_counter()
    res = driver(objective, params, bounds, n_trials, seed, on_progress=on_progress)
    elapsed = time.perf_counter() - t0

    # Build the best-fit curves for the result panel: per condition,
    # report (i_meas, y_meas, y_pred-using-best-params).
    p_best = deepcopy(baseline_params)
    for k, v in res["best_params"].items():
        p_best[k] = v
    best_curves = {}
    for cond_key in conditions:
        i_meas, y_meas = exp_cache[cond_key]
        if i_meas.size == 0:
            continue
        y_pred = predict(p_best, cond_key, i_meas)
        best_curves[cond_key] = {
            "i_meas": i_meas, "y_meas": y_meas, "y_pred": y_pred,
        }

    return {
        "best_params":   res["best_params"],
        "best_loss":     res["best_loss"],
        "history":       res["history"],
        "best_curves":   best_curves,
        "elapsed_s":     elapsed,
        "optimizer":     optimizer,
        "target":        target,
        "n_evals":       res["n_evals"],
        "message":       res["message"],
        "conditions":    conditions,
    }
