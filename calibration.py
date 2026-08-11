"""Run one of the PEMFC project's three calibration workflows.

Choose exactly one calibration with the three switches immediately below,
then run ``python calibration.py``.  ``python calibration.py --verify`` runs
short, non-optimizing checks for all three workflows.

The ECSA and Hmem workflows are executable versions of the two calibration
notebooks in the repository.  The dual-scale workflow fits PEMFC parameters
against polarization or HFR measurements using the shared calibration
backend.
"""

# =============================================================================
# CALIBRATION SELECTORS -- set exactly one of these three variables to True
# =============================================================================
RUN_ECSA_CALIBRATION = False
RUN_HMEM_CALIBRATION = False
RUN_DUAL_SCALE_CALIBRATION = True


import argparse
import csv
from copy import deepcopy
import io
import math
import os
from threading import Lock
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize_scalar

from config.initialize import parameters as DEFAULT_PARAMETERS
from data.export import export_experiment_data
from gui.calib_backend import run_calibration


# =============================================================================
# COMMON OUTPUT SETTINGS
# =============================================================================

# Write the selected workflow's result to calibration_result.csv.
SAVE_RESULT = True

# Display the selected workflow's result figure.
SHOW_PLOTS = True


# =============================================================================
# ECSA CALIBRATION SETTINGS
# Ported from ECSA calibration.ipynb.
# =============================================================================

# Normalized ECSA measurements at 0, 5k, ..., 30k voltage cycles.  This
# matches ``exp_data`` in the updated ECSA calibration notebook exactly.
ECSA_CYCLES = np.arange(0, 35_000, 5_000, dtype=float)
ECSA_EXPERIMENTAL = np.array([1.00, 0.71, 0.69, 0.595, 0.61, 0.58, 0.39])

# One model segment represents 5,000 six-second square-wave cycles.  The
# notebook runs one warm-up segment followed by six reported segments.
ECSA_SEGMENT_SECONDS = 5_000 * 6.0
ECSA_WARMUP_SEGMENTS = 1
ECSA_REPORTED_SEGMENTS = 6
ECSA_ODE_METHOD = "BDF"

# The notebook's Optuna budget and parameter search ranges.
ECSA_N_TRIALS = 1_000
ECSA_TIMEOUT_SECONDS = 8 * 3_600
ECSA_N_JOBS = 4
ECSA_SEED = 42
ECSA_PARAMETER_BOUNDS = {
    "krdp": (1e-13, 1e-7),
    "k1": (1e-17, 1e-11),
    "k2": (1e-17, 1e-11),
    "k3": (1e-17, 1e-13),
    "k4_ref": (1e-24, 1e-18),
}


# =============================================================================
# HMEM CALIBRATION SETTINGS
# Ported from Hmem calibration.ipynb.
# =============================================================================

HMEM_TIME_HOURS = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
HMEM_EXPERIMENTAL_M = np.array([2.5e-5, 2.39e-5, 2.25e-5, 2.05e-5, 1.8e-5])
HMEM_INITIAL_M = 2.5e-5
# The notebook uses this fixed value in its forward simulation.  The Python
# workflow treats it as a reference and calibrates A_1 inside the bounds below.
HMEM_NOTEBOOK_A1 = 5.5e-13
HMEM_A1_BOUNDS = (1e-15, 1e-11)
HMEM_ODE_METHOD = "BDF"
HMEM_MAX_STEP_SECONDS = 10.0


# =============================================================================
# DUAL-SCALE MODEL CALIBRATION SETTINGS
# =============================================================================

# Experimental target: "Polarization" or "HFR".
DUAL_SCALE_TARGET = "Polarization"

# This workflow calibrates the transient dual-scale PEMFC model.
DUAL_SCALE_MODEL = "Dual-scale"

# The dual-scale model uses ideal fixed gas supplies without the auxiliary
# balance-of-plant model.  Choose its transient integrator here.
DUAL_SCALE_AUX_SYSTEM = False
DUAL_SCALE_METHOD = "BDF"

# Optimizer: "TPE (Optuna)", "Genetic algorithm", or "Grid search".
DUAL_SCALE_OPTIMIZER = "TPE (Optuna)"
DUAL_SCALE_N_TRIALS = 500
DUAL_SCALE_SEED = 42

# Leave empty to use every condition in the selected experimental dataset.
DUAL_SCALE_CONDITIONS = [
    "T50_P300_HRC0",
    "T50_P300_HRC50",
    "T50_P500_HRC50",
]

# Parameters and bounds used by the dual-scale calibration.
DUAL_SCALE_PARAMS_TO_FIT = {
    "OCV": (0.90, 1.00),
    "i0_c_ref": (0.01, 15.0),
    "kappa_c": (0.5, 3.0),
    "epsilon_mc": (0.15, 0.5),
    "epsilon_c": (0.05, 0.35),
    "epsilon_cl": (0.1, 0.5),
    "Hcl": (1e-5, 3e-5),
    "Hgdl": (2e-4, 4e-4),
    "e": [3, 4, 5],
    "tau": (1.0, 4.0),
    "Re": (5e-7, 1e-5),
    "epsilon_gdl": (0.5, 0.8),
    "a_slim": (5e-3, 0.3),
    "b_slim": (0.1, 0.9),
    "a_switch": (0.05, 0.5),
}


# =============================================================================
# SHARED HELPERS
# =============================================================================

RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "calibration_result.csv")


def _selected_calibration(selectors=None):
    """Return the selected workflow name, requiring exactly one switch."""
    if selectors is None:
        selectors = {
            "ECSA": RUN_ECSA_CALIBRATION,
            "Hmem": RUN_HMEM_CALIBRATION,
            "Dual-scale": RUN_DUAL_SCALE_CALIBRATION,
        }
    selected = [name for name, enabled in selectors.items() if enabled]
    if len(selected) != 1:
        raise ValueError(
            "Set exactly one of RUN_ECSA_CALIBRATION, "
            "RUN_HMEM_CALIBRATION, or RUN_DUAL_SCALE_CALIBRATION to True "
            f"(currently selected: {selected or 'none'})."
        )
    return selected[0]


def _csv_result_rows(value, field=(), index=()):
    """Yield flattened CSV rows while preserving nested result structures."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _csv_result_rows(item, field + (str(key),), index)
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            yield from _csv_result_rows(value.item(), field, index)
            return
        for item_index in np.ndindex(value.shape):
            yield from _csv_result_rows(
                value[item_index], field,
                index + tuple(str(part) for part in item_index),
            )
        return
    if isinstance(value, (list, tuple)):
        for item_index, item in enumerate(value):
            yield from _csv_result_rows(
                item, field, index + (str(item_index),)
            )
        return
    if isinstance(value, np.generic):
        value = value.item()
    yield {
        "field": ".".join(field),
        "index": ".".join(index),
        "value": value,
    }


def _write_result_csv(stream, result):
    """Write every calibration result value in a portable long-form CSV."""
    workflow = str(result.get("calibration", ""))
    writer = csv.DictWriter(
        stream,
        fieldnames=("calibration", "field", "index", "value"),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in _csv_result_rows(
        {key: value for key, value in result.items()
         if key != "calibration"}
    ):
        writer.writerow({"calibration": workflow, **row})


def _save_result(result):
    with open(RESULT_PATH, "w", encoding="utf-8", newline="") as stream:
        _write_result_csv(stream, result)
    print(f"Result saved: {RESULT_PATH}")


def _print_progress(done, total, best_loss=None, *, label="Calibration"):
    """Render a reusable single-line terminal progress bar."""
    width = 28
    fraction = min(1.0, done / max(1, total))
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    detail = (
        "" if best_loss is None
        else f"  best loss = {float(best_loss):.4g}"
    )
    print(
        f"\r   {label:<18s} [{bar}] {done:4d}/{total}{detail}",
        end="",
        flush=True,
    )


# =============================================================================
# ECSA MODEL AND CALIBRATION
# =============================================================================

FARADAY = 96_485.0
GAS_CONSTANT = 8.314
WATER_DENSITY = 997.0
WATER_MOLAR_MASS = 18.02e-3
PT_MOLAR_VOLUME = 9.09
PT_MOLAR_MASS = 195.0849
PT_DENSITY = 21.45
CARBON_DENSITY = 2.26
CARBON_MOLAR_MASS = 12.01
PT2_REFERENCE_CONCENTRATION = 1e-3
PT_SITE_DENSITY = 2.18e-9

ECSA_N_PARTICLE_GROUPS = 50
ECSA_R_MIN = 1e-8
ECSA_R_MAX = 1e-6
_ECSA_DR = 1e-6 / ECSA_N_PARTICLE_GROUPS
ECSA_RADII = (
    np.linspace(ECSA_R_MIN, ECSA_R_MAX, ECSA_N_PARTICLE_GROUPS + 1)
    + _ECSA_DR / 2.0
)[1:]

ECSA_BASE_PARAMETERS = {
    "krdp": 1e-10,
    "k1": 3e-9,
    "k1_ref": 1e-18,
    "k2": 1e-13,
    "k2_ref": 1e-13,
    "k3": 1e-15,
    "k4": 0.0,
    "k5": 0.0,
    "k4_ref": 1.7e-21,
}


def proton_concentration_ccl(lambda_w, ew=1.1, rho_mem=0.002):
    return 1.0 / ((ew / rho_mem) + lambda_w *
                  (WATER_MOLAR_MASS / WATER_DENSITY))


def pt_dissolution(ucell, temperature, c_pt2, theta, *, k1, k1_ref,
                   **_unused):
    """Pt dissolution rate with the repository model's canonical argument order."""
    alpha = 0.5
    equilibrium_voltage = 1.15
    electrons = 2.0
    forward = np.exp(
        alpha * FARADAY * electrons / (GAS_CONSTANT * temperature)
        * (ucell - equilibrium_voltage)
    )
    reverse = (
        c_pt2 / PT2_REFERENCE_CONCENTRATION
        * np.exp(
            -(1.0 - alpha) * FARADAY * electrons
            / (GAS_CONSTANT * temperature)
            * (ucell - equilibrium_voltage)
        )
    )
    return k1 * (1.0 - theta) * forward - k1_ref * reverse


def pt_oxidation(ucell, temperature, proton_concentration, theta, *, k2,
                 k2_ref, **_unused):
    equilibrium_voltage = 0.97
    electrons = 2.0
    alpha = 0.5
    omega = 27e3
    proton_reference = 1e-3
    forward = (
        k2 * np.exp(-omega * theta / (GAS_CONSTANT * temperature))
        * np.exp(
            alpha * FARADAY * electrons / (GAS_CONSTANT * temperature)
            * (ucell - equilibrium_voltage)
        )
    )
    reverse = (
        k2_ref * theta * (proton_concentration / proton_reference) ** 2
        * np.exp(
            -alpha * FARADAY * electrons / (GAS_CONSTANT * temperature)
            * (ucell - equilibrium_voltage)
        )
    )
    return forward - reverse


def pt_oxide_dissolution(theta, proton_concentration, *, k3, **_unused):
    return k3 * theta * proton_concentration ** 2


def pt_detachment(ucell, temperature, radii, *, k4_ref, **_unused):
    electrons = 2.0
    alpha = 0.5
    equilibrium_voltage = 0.2
    return (
        k4_ref * CARBON_MOLAR_MASS / CARBON_DENSITY
        * np.exp(
            alpha * FARADAY * electrons / (GAS_CONSTANT * temperature)
            * (ucell - equilibrium_voltage)
        )
        / radii
    )


def initial_particle_radius_distribution(resolution=100, rmin=ECSA_R_MIN,
                                         rmax=ECSA_R_MAX, std=0.549,
                                         mu=0.538):
    radius = np.linspace(rmin, rmax, resolution)
    return (
        1.0 / (std * np.sqrt(4.0 * np.pi))
        * np.exp(-(np.log(radius * 1e7) - mu) ** 2 / (2.0 * std ** 2))
    )


def ecsa_square_wave(t):
    """Six-second square-wave voltage protocol from the ECSA notebook."""
    t = np.asarray(t) % 6.0
    rise = 0.2 * np.tanh(8.0 * (t - 1.5)) + 0.3
    drop = -0.2 * np.tanh(8.0 * (t - 4.5)) + 0.3
    return rise + drop


class ECSACatalystLayer:
    """Cathode catalyst-layer degradation model used by the ECSA fit."""

    def __init__(self, parameters=None):
        self.temperature = 353.15
        self.parameters = dict(ECSA_BASE_PARAMETERS)
        if parameters:
            self.parameters.update(parameters)
        self.prd0 = initial_particle_radius_distribution(
            resolution=ECSA_N_PARTICLE_GROUPS
        )

    def initial_state(self):
        return np.concatenate((
            [0.0],
            np.zeros(ECSA_N_PARTICLE_GROUPS),
            self.prd0,
        ))

    def derivative(self, t, state):
        c_pt2 = state[0]
        theta = state[1:1 + ECSA_N_PARTICLE_GROUPS]
        prd = state[-ECSA_N_PARTICLE_GROUPS:]
        voltage = float(ecsa_square_wave(t))

        proton_concentration = proton_concentration_ccl(lambda_w=8.5)
        dissolution = pt_dissolution(
            voltage, self.temperature, c_pt2, theta, **self.parameters
        )
        oxidation = pt_oxidation(
            voltage, self.temperature, proton_concentration, theta,
            **self.parameters
        )
        oxide_dissolution = pt_oxide_dissolution(
            theta, proton_concentration, **self.parameters
        )
        detachment = pt_detachment(
            voltage, self.temperature, ECSA_RADII, **self.parameters
        )

        radius_factor = (
            2.0 * PT_MOLAR_VOLUME * 0.2e-4
            / (GAS_CONSTANT * 353.0)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            dr_dt = (
                PT_MOLAR_VOLUME * self.parameters["krdp"] * c_pt2
                * np.exp(-radius_factor / ECSA_RADII)
                - PT_MOLAR_VOLUME * (dissolution + oxidation)
                * PT2_REFERENCE_CONCENTRATION
                * np.exp(radius_factor / ECSA_RADII)
            )

            initial_pt_mass = (
                4.0 / 3.0 * np.pi * PT_DENSITY
                * trapezoid(self.prd0 * ECSA_RADII ** 3, x=ECSA_RADII)
            )
            dissolved_mass_rate = (
                4.0 * np.pi * PT_DENSITY
                * trapezoid(prd * ECSA_RADII ** 2 * dr_dt, x=ECSA_RADII)
            )
            oxide_mass_rate = (
                4.0 * np.pi * PT_DENSITY
                * trapezoid(
                    prd * ECSA_RADII ** 2 * oxide_dissolution,
                    x=ECSA_RADII,
                )
            )
            d_c_pt2 = (
                -3.33 / PT_MOLAR_MASS
                * (dissolved_mass_rate - oxide_mass_rate) / initial_pt_mass
            )
            d_prd = -np.gradient(prd * dr_dt, ECSA_RADII) - detachment * prd
            d_theta = (
                (oxidation - oxide_dissolution) / PT_SITE_DENSITY
                - (2.0 * theta / ECSA_RADII) * dr_dt
            )

        return np.concatenate(([d_c_pt2], d_theta, d_prd))


def simulate_ecsa(parameters, *, segment_seconds=ECSA_SEGMENT_SECONDS,
                  warmup_segments=ECSA_WARMUP_SEGMENTS,
                  reported_segments=ECSA_REPORTED_SEGMENTS,
                  max_step=np.inf, on_progress=None):
    """Return normalized ECSA at the seven notebook measurement points."""
    model = ECSACatalystLayer(parameters)
    state = model.initial_state()
    segment_ratios = []
    denominator = trapezoid(
        ECSA_RADII ** 2 * model.prd0, x=ECSA_RADII
    )

    total_segments = warmup_segments + reported_segments
    for segment_index in range(total_segments):
        solution = solve_ivp(
            model.derivative,
            (0.0, float(segment_seconds)),
            state,
            method=ECSA_ODE_METHOD,
            max_step=max_step,
        )
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError(
                "ECSA integration failed in segment "
                f"{segment_index + 1}: {solution.message}"
            )
        state = solution.y[:, -1]
        ratio = trapezoid(
            ECSA_RADII ** 2 * state[-ECSA_N_PARTICLE_GROUPS:],
            x=ECSA_RADII,
        ) / denominator
        segment_ratios.append(float(ratio))
        if on_progress is not None:
            on_progress(segment_index + 1, total_segments)

    # Match the notebook: the first segment establishes the cycle-zero model
    # state and the six following segment tails are compared with 5k..30k.
    return np.asarray([1.0] + segment_ratios[warmup_segments:], dtype=float)


def run_ecsa_calibration():
    """Fit the five catalyst-degradation constants with Optuna TPE."""
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError as exc:
        raise RuntimeError(
            "ECSA calibration requires Optuna. Install the packages in "
            "requirements.txt and retry."
        ) from exc

    if ECSA_EXPERIMENTAL.shape != ECSA_CYCLES.shape:
        raise ValueError("ECSA_CYCLES and ECSA_EXPERIMENTAL must have equal length.")

    def objective(trial):
        trial_parameters = {
            name: trial.suggest_float(name, low, high, log=True)
            for name, (low, high) in ECSA_PARAMETER_BOUNDS.items()
        }
        try:
            simulated = simulate_ecsa(trial_parameters)
        except (RuntimeError, FloatingPointError, ValueError):
            return 1e6
        if simulated.shape != ECSA_EXPERIMENTAL.shape:
            return 1e6
        return float(np.sum((ECSA_EXPERIMENTAL - simulated) ** 2))

    print("Starting ECSA calibration. This is the long notebook workflow; "
          "each trial performs seven stiff integrations.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=ECSA_SEED)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=MedianPruner(),
    )

    progress_lock = Lock()

    def report_trial_progress(current_study, _finished_trial):
        with progress_lock:
            completed = len(current_study.trials)
            try:
                best_value = current_study.best_value
            except ValueError:
                best_value = None
            _print_progress(
                completed,
                ECSA_N_TRIALS,
                best_value,
                label="ECSA trials",
            )

    started = time.perf_counter()
    study.optimize(
        objective,
        n_trials=ECSA_N_TRIALS,
        timeout=ECSA_TIMEOUT_SECONDS,
        n_jobs=ECSA_N_JOBS,
        callbacks=[report_trial_progress],
    )
    # A timeout may stop before the requested trial count.  Finish the visible
    # bar against the number that actually ran so the terminal never appears
    # stuck on an incomplete line.
    completed_trials = len(study.trials)
    _print_progress(
        completed_trials,
        max(1, completed_trials),
        study.best_value,
        label="ECSA completed",
    )
    print()
    elapsed = time.perf_counter() - started

    print("Validating the best ECSA parameters:")
    simulated = simulate_ecsa(
        study.best_params,
        on_progress=lambda done, total: _print_progress(
            done,
            total,
            study.best_value,
            label="ECSA validation",
        ),
    )
    print()
    history = [
        (trial.number + 1, float(trial.value))
        for trial in study.trials
        if trial.value is not None and math.isfinite(float(trial.value))
    ]
    result = {
        "calibration": "ECSA",
        "best_params": dict(study.best_params),
        "best_loss_sse": float(study.best_value),
        "elapsed_s": elapsed,
        "n_evals": len(study.trials),
        "cycles": ECSA_CYCLES,
        "experimental": ECSA_EXPERIMENTAL,
        "simulated": simulated,
        "history": history,
    }

    print(f"ECSA best SSE: {result['best_loss_sse']:.6g}")
    print("ECSA best parameters:")
    for name, value in result["best_params"].items():
        print(f"  {name:<8s} = {value:.8g}")

    if SAVE_RESULT:
        _save_result(result)
    if SHOW_PLOTS:
        _plot_ecsa_result(result)
    return result


def _plot_ecsa_result(result):
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    history = np.asarray(result["history"], dtype=float)
    if history.size:
        axes[0].scatter(history[:, 0], history[:, 1], s=14, alpha=0.45,
                        color="tab:gray", label="trial SSE")
        axes[0].plot(
            history[:, 0], np.minimum.accumulate(history[:, 1]),
            color="tab:blue", label="best so far",
        )
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Sum of squared errors")
    axes[0].set_title("ECSA optimizer convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result["cycles"], result["experimental"], "--o",
                 color="gray", label="experimental")
    axes[1].plot(result["cycles"], result["simulated"], "-^",
                 color="tab:blue", label="simulation")
    axes[1].set_xlabel("Voltage cycles")
    axes[1].set_ylabel("Normalized ECSA")
    axes[1].set_title("ECSA fit")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    plt.show()


# =============================================================================
# HMEM MODEL AND CALIBRATION
# =============================================================================

def saturation_pressure(temperature):
    """Saturated water-vapour pressure in Pa at ``temperature`` in K."""
    celsius = temperature - 273.15
    return 101_325.0 * 10.0 ** (
        -2.1794 + 0.02953 * celsius - 9.1837e-5 * celsius ** 2
        + 1.4454e-7 * celsius ** 3
    )


def fluoride_release_rate(membrane_thickness, voltage, temperature,
                          oxygen_pressure, a1,
                          initial_thickness=HMEM_INITIAL_M):
    """Fluoride release-rate model from the Hmem notebook."""
    alpha_equivalent = 0.52
    activation_energy = 75e3
    reference_temperature = 273.15 + 95.0
    reference_pressure = 1e5
    return (
        a1 * (oxygen_pressure / reference_pressure)
        * (initial_thickness / membrane_thickness)
        * np.exp(
            alpha_equivalent * FARADAY * voltage
            / (GAS_CONSTANT * temperature)
        )
        * np.exp(
            -activation_energy / GAS_CONSTANT
            * (1.0 / temperature - 1.0 / reference_temperature)
        )
    )


def _hmem_rate_factor():
    temperature = 368.0
    cathode_pressure = 1.5e5
    relative_humidity = 0.3
    oxygen_pressure = (
        0.21 * cathode_pressure
        * (1.0 - relative_humidity * saturation_pressure(temperature)
           / cathode_pressure)
    )
    # dH/dt = -factor * A1 * H0/H.  Keeping this factor separate provides
    # a fast exact objective while the final curve is still checked with BDF.
    release_per_a1_at_h0 = fluoride_release_rate(
        HMEM_INITIAL_M, 0.93, temperature, oxygen_pressure, 1.0
    )
    return 20.8 / (0.82 * 1980e3) * release_per_a1_at_h0


def _predict_hmem_exact(a1, times_seconds):
    """Closed-form solution of the notebook's scalar dH/dt equation."""
    times_seconds = np.asarray(times_seconds, dtype=float)
    radicand = (
        HMEM_INITIAL_M ** 2
        - 2.0 * _hmem_rate_factor() * a1 * HMEM_INITIAL_M * times_seconds
    )
    if np.any(radicand <= 0.0):
        return np.full(times_seconds.shape, np.nan)
    return np.sqrt(radicand)


def solve_hmem(a1, times_seconds, *, max_step=HMEM_MAX_STEP_SECONDS):
    """Integrate the Hmem notebook equation with its BDF solver."""
    times_seconds = np.asarray(times_seconds, dtype=float)
    if times_seconds.ndim != 1 or times_seconds.size == 0:
        raise ValueError("times_seconds must be a non-empty one-dimensional array.")
    if np.any(np.diff(times_seconds) < 0.0) or times_seconds[0] < 0.0:
        raise ValueError("times_seconds must be sorted and non-negative.")

    factor = _hmem_rate_factor()

    def derivative(_t, state):
        thickness = float(state[0])
        if thickness <= 0.0:
            return [np.nan]
        return [-factor * a1 * HMEM_INITIAL_M / thickness]

    solution = solve_ivp(
        derivative,
        (0.0, float(times_seconds[-1])),
        [HMEM_INITIAL_M],
        method=HMEM_ODE_METHOD,
        max_step=max_step,
        t_eval=times_seconds,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f"Hmem integration failed: {solution.message}")
    return solution.y[0]


def fit_hmem_a1(on_progress=None):
    """Calibrate A_1 in log space and return optimizer diagnostics.

    The Hmem notebook only forward-simulates its hard-coded A_1 value.  Since
    A_1 is the sole unknown in that degradation equation, a bounded scalar
    minimization is the complete calibration problem.  The exact solution of
    the same scalar ODE makes objective evaluations fast; ``run_hmem_calibration``
    validates the optimum with the notebook's BDF integration afterward.
    """
    times_seconds = HMEM_TIME_HOURS * 3_600.0
    low, high = HMEM_A1_BOUNDS
    if not (0.0 < low < high):
        raise ValueError("HMEM_A1_BOUNDS must be positive and increasing.")
    if not low <= HMEM_NOTEBOOK_A1 <= high:
        raise ValueError("HMEM_NOTEBOOK_A1 must lie inside HMEM_A1_BOUNDS.")

    evaluation_count = 0
    best_loss = float("inf")

    def loss_for(log10_a1):
        prediction = _predict_hmem_exact(10.0 ** log10_a1, times_seconds)
        if not np.isfinite(prediction).all():
            return 1e6
        return float(np.mean((prediction - HMEM_EXPERIMENTAL_M) ** 2))

    def objective(log10_a1):
        nonlocal evaluation_count, best_loss
        loss = loss_for(log10_a1)
        evaluation_count += 1
        best_loss = min(best_loss, loss)
        if on_progress is not None:
            # The current Hmem problem converges in about 15 evaluations.  The
            # denominator grows if user-edited bounds require more work.
            on_progress(
                evaluation_count,
                max(15, evaluation_count),
                best_loss,
            )
        return loss

    optimum = minimize_scalar(
        objective,
        bounds=(math.log10(low), math.log10(high)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not optimum.success:
        raise RuntimeError(f"Hmem optimizer failed: {optimum.message}")

    best_a1 = 10.0 ** float(optimum.x)
    reference_loss = loss_for(math.log10(HMEM_NOTEBOOK_A1))
    return {
        "best_a1": best_a1,
        "best_loss_exact_mse_m2": float(optimum.fun),
        "reference_a1": HMEM_NOTEBOOK_A1,
        "reference_loss_exact_mse_m2": float(reference_loss),
        "n_evals": int(optimum.nfev),
        "optimizer": "bounded scalar minimization of log10(A_1)",
    }


def run_hmem_calibration():
    """Fit A_1 to the Hmem measurements, then verify the fit with BDF."""
    times_seconds = HMEM_TIME_HOURS * 3_600.0
    started = time.perf_counter()
    fit = fit_hmem_a1(
        on_progress=lambda done, total, best: _print_progress(
            done, total, best, label="Hmem optimization"
        )
    )
    _print_progress(
        fit["n_evals"],
        fit["n_evals"],
        fit["best_loss_exact_mse_m2"],
        label="Hmem optimization",
    )
    print()
    best_a1 = fit["best_a1"]
    print("Validating the fitted Hmem parameter with BDF:")
    _print_progress(0, 1, label="Hmem BDF")
    simulated = solve_hmem(best_a1, times_seconds)
    mse = float(np.mean((simulated - HMEM_EXPERIMENTAL_M) ** 2))
    _print_progress(1, 1, mse, label="Hmem BDF")
    print()
    improvement_percent = 100.0 * (
        1.0 - mse / fit["reference_loss_exact_mse_m2"]
    )
    result = {
        "calibration": "Hmem",
        "optimizer": fit["optimizer"],
        "best_params": {"A_1": best_a1},
        "best_loss_mse_m2": mse,
        "objective_loss_exact_mse_m2": fit["best_loss_exact_mse_m2"],
        "rmse_um": math.sqrt(mse) * 1e6,
        "n_evals": fit["n_evals"],
        "elapsed_s": time.perf_counter() - started,
        "notebook_reference_A_1": fit["reference_a1"],
        "notebook_reference_loss_mse_m2": fit["reference_loss_exact_mse_m2"],
        "loss_improvement_percent": improvement_percent,
        "time_hours": HMEM_TIME_HOURS,
        "experimental_m": HMEM_EXPERIMENTAL_M,
        "simulated_m": simulated,
    }

    print(f"Hmem fitted A_1: {best_a1:.8g} ug/(h m^2)")
    print(f"Hmem RMSE: {result['rmse_um']:.6g} um")
    print(f"Hmem optimizer evaluations: {result['n_evals']}")
    print(f"Hmem loss improvement vs notebook A_1: {improvement_percent:.3f}%")
    if SAVE_RESULT:
        _save_result(result)
    if SHOW_PLOTS:
        _plot_hmem_result(result)
    return result


def _plot_hmem_result(result):
    dense_hours = np.linspace(0.0, float(HMEM_TIME_HOURS[-1]), 401)
    dense_thickness = _predict_hmem_exact(
        result["best_params"]["A_1"], dense_hours * 3_600.0
    )
    reference_thickness = _predict_hmem_exact(
        result["notebook_reference_A_1"], dense_hours * 3_600.0
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    axis.plot(dense_hours, dense_thickness, label="fitted simulation")
    axis.plot(dense_hours, reference_thickness, ":", color="tab:orange",
              label="notebook fixed A_1")
    axis.plot(result["time_hours"], result["experimental_m"], "--o",
              color="gray", label="experimental")
    axis.set_xlabel("Time (hours)")
    axis.set_ylabel("Membrane thickness (m)")
    axis.set_title("Hmem calibration")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()


# =============================================================================
# DUAL-SCALE MODEL CALIBRATION
# =============================================================================

def _dataset_key(target):
    try:
        return {"Polarization": "pola", "HFR": "hfr"}[target]
    except KeyError as exc:
        raise ValueError(
            "DUAL_SCALE_TARGET must be 'Polarization' or 'HFR', "
            f"not {target!r}."
        ) from exc


def _build_dual_scale_request():
    return {
        "target": DUAL_SCALE_TARGET,
        "model": DUAL_SCALE_MODEL,
        "aux_system": DUAL_SCALE_AUX_SYSTEM,
        "method": DUAL_SCALE_METHOD,
        "optimizer": DUAL_SCALE_OPTIMIZER,
        "n_trials": DUAL_SCALE_N_TRIALS,
        "seed": DUAL_SCALE_SEED,
        "params": list(DUAL_SCALE_PARAMS_TO_FIT),
        "bounds": DUAL_SCALE_PARAMS_TO_FIT,
        "conditions": list(DUAL_SCALE_CONDITIONS),
    }


def run_dual_scale_calibration():
    """Fit parameters of the transient dual-scale PEMFC model."""
    request = _build_dual_scale_request()
    if request["model"] != "Dual-scale" or request["aux_system"]:
        raise ValueError(
            "Dual-scale calibration requires DUAL_SCALE_MODEL='Dual-scale' "
            "and DUAL_SCALE_AUX_SYSTEM=False."
        )
    data = export_experiment_data(_dataset_key(request["target"]))
    if not request["conditions"]:
        request["conditions"] = sorted(data)
    missing = [key for key in request["conditions"] if key not in data]
    if missing:
        raise ValueError(
            f"Conditions not found in {request['target']} data: {missing}. "
            f"Available conditions: {sorted(data)}"
        )

    print("=" * 70)
    print("Dual-scale PEMFC parameter calibration")
    print(f"Target: {request['target']}")
    print(f"Model: {request['model']} (aux_system={request['aux_system']})")
    print(f"Optimizer: {request['optimizer']}")
    print(f"Conditions: {request['conditions']}")
    print(f"Parameters: {request['params']}")
    print("=" * 70)

    result = run_calibration(
        request,
        baseline_params=deepcopy(DEFAULT_PARAMETERS),
        data=data,
        on_progress=lambda done, total, best: _print_progress(
            done, total, best, label="Dual-scale trials"
        ),
    )
    result = {"calibration": "Dual-scale", **result}
    print("\n")
    print(f"Best loss: {result['best_loss']:.6g}")
    print("Best parameters:")
    for name, value in result["best_params"].items():
        print(f"  {name:<12s} = {value:.8g}")

    if SAVE_RESULT:
        _save_result(result)
    if SHOW_PLOTS:
        _plot_dual_scale_result(result)
    return result


def _plot_dual_scale_result(result):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    history = np.asarray(result["history"], dtype=float)
    if history.size:
        axes[0].scatter(history[:, 0], history[:, 1], marker=".",
                        alpha=0.55, color="tab:gray", label="trial loss")
        axes[0].plot(
            history[:, 0], np.minimum.accumulate(history[:, 1]),
            color="tab:blue", label="best so far",
        )
        if np.all(history[:, 1] > 0.0):
            axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Convergence - {result['optimizer']}")
    axes[0].grid(True, alpha=0.3)

    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
    for index, (condition, curve) in enumerate(result["best_curves"].items()):
        color = palette[index % len(palette)]
        axes[1].plot(curve["i_meas"], curve["y_meas"], "o", color=color,
                     label=f"{condition} (experimental)")
        axes[1].plot(curve["i_meas"], curve["y_pred"], "--", color=color,
                     label=f"{condition} (fit)")
    axes[1].set_xlabel("Load current (A)")
    axes[1].set_ylabel(
        "Per-cell voltage (V)" if result["target"] == "Polarization"
        else "HFR (mOhm)"
    )
    axes[1].set_title(f"{result['target']} fit")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    plt.show()


# =============================================================================
# VERIFICATION AND COMMAND-LINE ENTRY POINT
# =============================================================================

def verify_calibration_file():
    """Run short checks without launching any optimizer or opening plots."""
    print("Verifying calibration.py ...")

    # Confirm that nested calibration results are preserved in CSV rows.
    csv_buffer = io.StringIO(newline="")
    _write_result_csv(csv_buffer, {
        "calibration": "verification",
        "best_params": {"parameter_a": 1.25},
        "history": [(1, 0.5), (2, 0.25)],
        "curve": np.array([3.0, 4.0]),
    })
    csv_buffer.seek(0)
    csv_rows = list(csv.DictReader(csv_buffer))
    assert csv_rows
    assert set(csv_rows[0]) == {"calibration", "field", "index", "value"}
    assert all(row["calibration"] == "verification" for row in csv_rows)
    assert any(
        row["field"] == "best_params.parameter_a"
        and float(row["value"]) == 1.25
        for row in csv_rows
    )
    assert sum(row["field"] == "history" for row in csv_rows) == 4
    assert sum(row["field"] == "curve" for row in csv_rows) == 2
    print("  [ok] nested calibration results serialize to CSV")

    # Selector behavior: all three modes are independently selectable and an
    # invalid zero/multiple selection is rejected before any long computation.
    calibration_names = ("ECSA", "Hmem", "Dual-scale")
    for expected in calibration_names:
        selection = {name: name == expected for name in calibration_names}
        assert _selected_calibration(selection) == expected
    for invalid in (
        {"ECSA": False, "Hmem": False, "Dual-scale": False},
        {"ECSA": True, "Hmem": True, "Dual-scale": False},
    ):
        try:
            _selected_calibration(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid calibration-selector combination was accepted.")
    print("  [ok] three exclusive calibration selectors")

    # ECSA derivative plus a very short seven-segment integration exercises
    # the complete state layout without running the multi-hour calibration.
    ecsa_model = ECSACatalystLayer()
    derivative = ecsa_model.derivative(0.0, ecsa_model.initial_state())
    assert derivative.shape == ecsa_model.initial_state().shape
    assert np.isfinite(derivative).all()
    ecsa_smoke = simulate_ecsa(
        {}, segment_seconds=0.02, warmup_segments=1,
        reported_segments=6, max_step=0.01,
    )
    assert ecsa_smoke.shape == ECSA_EXPERIMENTAL.shape
    assert np.isfinite(ecsa_smoke).all()
    assert np.array_equal(
        ECSA_EXPERIMENTAL,
        np.array([1.0, 0.71, 0.69, 0.595, 0.61, 0.58, 0.39]),
    )
    print("  [ok] updated ECSA data and short BDF integration")

    # Exercise the actual Hmem optimizer, require an interior solution that
    # improves on the notebook's fixed A_1, then check the BDF forward model.
    hmem_fit = fit_hmem_a1()
    low, high = HMEM_A1_BOUNDS
    assert low < hmem_fit["best_a1"] < high
    assert hmem_fit["n_evals"] > 1
    assert (
        hmem_fit["best_loss_exact_mse_m2"]
        < hmem_fit["reference_loss_exact_mse_m2"]
    )
    smoke_times = np.array([0.0, 300.0, 600.0])
    smoke_a1 = hmem_fit["best_a1"]
    hmem_bdf = solve_hmem(smoke_a1, smoke_times, max_step=10.0)
    hmem_exact = _predict_hmem_exact(smoke_a1, smoke_times)
    assert np.allclose(hmem_bdf, hmem_exact, rtol=1e-7, atol=1e-12)
    print(
        "  [ok] Hmem A_1 optimizer and BDF validation "
        f"(A_1={hmem_fit['best_a1']:.6g}, evals={hmem_fit['n_evals']})"
    )

    # Confirm that the third workflow is a no-auxiliary dual-scale request.
    # A short Static objective check exercises the shared data/objective path
    # without spending minutes on transient settling during verification.
    from gui.calib_backend import (
        FAILURE_LOSS,
        make_objective,
        resolve_transient_model,
    )

    dual_scale_request = _build_dual_scale_request()
    assert dual_scale_request["model"] == "Dual-scale"
    assert dual_scale_request["aux_system"] is False
    assert resolve_transient_model(
        dual_scale_request["model"], dual_scale_request["aux_system"]
    ) == "PEMFC"

    data = export_experiment_data("pola")
    condition = sorted(data)[0]
    objective, _experimental, _predict = make_objective(
        "Polarization",
        deepcopy(DEFAULT_PARAMETERS),
        data,
        [condition],
        model_variant="Static",
        aux_system=False,
        method="BDF",
    )
    backend_loss = float(objective({}))
    assert math.isfinite(backend_loss) and backend_loss < FAILURE_LOSS
    print(
        "  [ok] dual-scale request resolves to PEMFC and shared objective "
        f"loads data (smoke loss={backend_loss:.6g})"
    )

    print("All calibration.py verification checks passed.")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run short checks for all workflows without optimizing",
    )
    arguments = parser.parse_args(argv)
    if arguments.verify:
        verify_calibration_file()
        return 0

    selected = _selected_calibration()
    if selected == "ECSA":
        run_ecsa_calibration()
    elif selected == "Hmem":
        run_hmem_calibration()
    elif selected == "Dual-scale":
        run_dual_scale_calibration()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
