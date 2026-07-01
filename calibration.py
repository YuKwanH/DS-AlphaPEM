"""PEMFC simulator - raw-code parameter-calibration entry point.

Fits model parameters against experimental polarization or HFR data using
one of three optimizers (TPE / Genetic algorithm / Grid search). Edit
the USER SETTINGS block below, then run::

    python calibration.py

For a single forward simulation use ``main.py``. For the interactive
calibration GUI use ``run_gui.bat`` (or ``streamlit run gui/app.py``) and
open the Calibration page.
"""
from copy import deepcopy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from config.initialize import parameters as DEFAULT_PARAMETERS
from data.export       import export_experiment_data
from gui.calib_backend import run_calibration


# =============================================================================
# USER SETTINGS  --  edit these, save, then re-run `python calibration.py`
# =============================================================================

# Experimental dataset to fit against: "Polarization" or "HFR".
TARGET = "Polarization"

# Model variant the calibration backend evaluates:
#   * "Static"     -> PEMFC_stat (algebraic, ~50 ms/point — recommended default)
#   * "Dual-scale" -> PEMFC, transient-to-steady at each measurement current
#                     (aux=off; or auto-promoted to PEMFC_dyn if AUX_SYSTEM=True)
#   * "Dynamic"    -> PEMFC_dyn (full BoP); demoted to PEMFC if AUX_SYSTEM=False
# Transient modes are 50-200x slower per trial than Static — set N_TRIALS
# small (~10-20) and use a short CONDITIONS list when picking them.
MODEL_VARIANT = "Dynamic"

# Balance-of-plant equations (used only when MODEL_VARIANT != "Static"):
#   True  -> transient PEMFC_dyn (compressor / manifolds integrated)
#   False -> transient PEMFC (ideal fixed supply)
# The routing matches the simulation page exactly.
AUX_SYSTEM = False

# PDE solver passed to `scipy.integrate.solve_ivp` for transient
# calibration ("BDF", "Radau", "LSODA", "RK45"). The backend does NOT
# silently switch to a different solver mid-run — a trial that fails
# with this solver stays failed. Ignored for MODEL_VARIANT="Static".
METHOD = "BDF"

# Optimizer: "TPE (Optuna)", "Genetic algorithm", or "Grid search".
OPTIMIZER = "TPE (Optuna)"

# Number of trials (TPE/Grid) or function evaluations (GA derives popsize
# and maxiter from this).
N_TRIALS = 50

# Reproducibility seed for the sampler / population.
SEED = 42

# Operating conditions to include in the loss. Each key encodes
# T{degC}_P{P*1e5 Pa offset}_HRC{percent}, e.g. "T50_P300_HRC0" means
# Tfc=50 C, P=1.3 bar, RHc=0 %. Leave empty to use every condition the
# experimental file contains.
CONDITIONS = ["T50_P300_HRC0", "T50_P300_HRC50", "T50_P400_HRC50"]

# Parameter set to fit.
#   * Continuous : value is the (low, high) tuple.
#   * Discrete   : value is the list of allowed choices.
PARAMS_TO_FIT = {
    "OCV":         (0.90, 1.00),
    "i0_c_ref":    (0.01, 15.0),     # log scale auto-detected by backend
    "kappa_c":     (0.5,  3.0),
    "epsilon_mc":  (0.15, 0.5),
    "epsilon_c":   (0.05, 0.35),
    "epsilon_cl":  (0.1,  0.5),
    "Hcl":         (1e-5, 3e-5),     # log scale
    "Hgdl":        (2e-4, 4e-4),     # log scale
    "e":           [3, 4, 5],        # discrete (categorical)
}

# When True, write the result bundle (best params + convergence history +
# best-fit curves) to `calibration_result.json` next to this script.
SAVE_RESULT = True

# When True, open matplotlib windows with the convergence curve and the
# best-fit overlay vs experimental data.
SHOW_PLOTS = True


# =============================================================================
# Run --  no editing needed below this line for a normal calibration
# =============================================================================

def _print_progress(done, total, best_loss):
    """Single-line progress indicator printed during the optimizer loop."""
    bar_width = 28
    frac = min(1.0, done / max(1, total))
    filled = int(round(bar_width * frac))
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r   [{bar}] {done:4d}/{total}  best loss = {best_loss:.4g}",
          end="", flush=True)


def _build_request():
    return {
        "target":     TARGET,
        "model":      MODEL_VARIANT,
        "aux_system": AUX_SYSTEM,
        "method":     METHOD,
        "optimizer":  OPTIMIZER,
        "n_trials":   N_TRIALS,
        "seed":       SEED,
        "params":     list(PARAMS_TO_FIT.keys()),
        "bounds":     PARAMS_TO_FIT,
        "conditions": CONDITIONS,
    }


def _dataset_key(target):
    return {"Polarization": "pola", "HFR": "hfr"}[target]


def _print_header(req, data, conditions):
    print("=" * 70)
    print(" PEMFC simulator - raw-code parameter calibration")
    print("=" * 70)
    print(f"   Target         : {req['target']}")
    if req["model"] == "Static":
        resolved = "PEMFC_stat (algebraic, ~50 ms/point)"
    else:
        from gui.calib_backend import resolve_transient_model
        cls = resolve_transient_model(req["model"], req["aux_system"])
        resolved = f"{cls} (transient settle, aux_system={req['aux_system']})"
    print(f"   Model variant  : {req['model']}   -> {resolved}")
    print(f"   Optimizer      : {req['optimizer']}")
    print(f"   Trials / seed  : {req['n_trials']}  /  seed={req['seed']}")
    print(f"   Conditions     : {len(conditions)} (of {len(data)} available)")
    print(f"                    {conditions}")
    print(f"   Parameters     : {len(req['params'])} ({', '.join(req['params'])})")
    print("=" * 70)


def _print_best(res):
    print("\n\n" + "=" * 70)
    print(f"   RESULT  best loss = {res['best_loss']:.6g}   "
          f"evals = {res['n_evals']}   wall = {res['elapsed_s']:.1f} s")
    print(f"   {res['message']}")
    print("=" * 70)
    print("   Best parameters:")
    for k, v in res["best_params"].items():
        bounds = PARAMS_TO_FIT.get(k)
        if isinstance(bounds, list):
            extra = f"   (choices: {bounds})"
        elif bounds is not None:
            extra = f"   (range: {bounds[0]:.3g} .. {bounds[1]:.3g})"
        else:
            extra = ""
        if isinstance(v, float):
            print(f"     {k:<12s} = {v:.6g}{extra}")
        else:
            print(f"     {k:<12s} = {v}{extra}")
    print("=" * 70)


def _plot_convergence(res, ax):
    trials = np.asarray([t for t, _ in res["history"]])
    losses = np.asarray([l for _, l in res["history"]])
    best_so_far = np.minimum.accumulate(losses)
    ax.plot(trials, losses, marker=".", linestyle="", alpha=0.55,
            color="tab:gray", label="trial loss")
    ax.plot(trials, best_so_far, linewidth=1.6, color="tab:blue",
            label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")
    ax.set_title(f"Convergence -- {res['optimizer']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_overlay(res, ax):
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
    for i, (cond, c) in enumerate(res["best_curves"].items()):
        col = palette[i % len(palette)]
        ax.plot(c["i_meas"], c["y_meas"], marker="o", markersize=5,
                linewidth=0, label=f"{cond} (exp)", color=col)
        ax.plot(c["i_meas"], c["y_pred"], linestyle="--", linewidth=1.5,
                label=f"{cond} (fit)", color=col)
    ax.set_xlabel("Load current $I_{LOAD}$ (A)")
    if res["target"] == "Polarization":
        ax.set_ylabel("Per-cell voltage $U_{cell}$ (V)")
        ax.set_title("Polarization -- fit vs measurement")
    else:
        ax.set_ylabel("HFR (m$\\Omega$)")
        ax.set_title("HFR -- fit vs measurement")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)


def _save_result(res):
    """Write the result bundle next to this script as JSON."""
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration_result.json")
    payload = {
        "optimizer":   res["optimizer"],
        "target":      res["target"],
        "best_params": res["best_params"],
        "best_loss":   res["best_loss"],
        "n_evals":     res["n_evals"],
        "elapsed_s":   res["elapsed_s"],
        "conditions":  res["conditions"],
        "history":     [(int(t), float(l)) for t, l in res["history"]],
        "best_curves": {
            k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            for k, v in res["best_curves"].items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"   Result saved   : {out_path}")


def main():
    req = _build_request()
    data = export_experiment_data(_dataset_key(req["target"]))

    # Resolve / validate the requested conditions against the dataset.
    if not req["conditions"]:
        req["conditions"] = sorted(data.keys())
    missing = [c for c in req["conditions"] if c not in data]
    if missing:
        raise ValueError(f"Conditions not found in {req['target']} data: {missing}\n"
                         f"   Available: {sorted(data.keys())}")

    _print_header(req, data, req["conditions"])

    baseline = deepcopy(DEFAULT_PARAMETERS)
    t0 = time.perf_counter()
    res = run_calibration(req, baseline_params=baseline, data=data,
                          on_progress=_print_progress)
    wall = time.perf_counter() - t0

    _print_best(res)

    if SAVE_RESULT:
        _save_result(res)

    if SHOW_PLOTS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        _plot_convergence(res, axes[0])
        _plot_overlay(res, axes[1])
        fig.suptitle(f"PEMFC calibration -- {req['target']}  "
                     f"{req['optimizer']}  best loss = {res['best_loss']:.4g}",
                     fontsize=11)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
