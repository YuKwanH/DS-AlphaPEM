"""Standalone accelerated stress test (AST) example for the PEMFC model.

Edit the user settings immediately below, then run::

    python AST_example.py

The default configuration reproduces ``AST example.ipynb``: a 5,000-cycle
dual-scale simulation with an optional 0D companion.  Use ``--verify`` for a
short end-to-end check that does not change the configured AST duration.
"""

# =============================================================================
# USER-ACCESSIBLE MODEL PARAMETERS
# =============================================================================

PARAMETERS = {
    # Current-profile and analysis parameters retained from the main model.
    "t_step": [0, 3600, 100, 1.5],
    "i_step": [5000.0, 15000.0],
    "delta_pola": [30, 30, 1000.0, 60],
    "i_max_pola": 16500.0,
    "i_EIS": 10000.0,
    "ratio_EIS": 0.05,
    "t_EIS": 15,
    "f_EIS": [-3, 5, 90, 50],

    # Accessible geometry.
    "Aact": 0.0031,          # Active area (m^2)
    "Hmem": 1.2e-5,         # Initial membrane thickness (m)
    "Hgc": 8.0e-4,          # Gas-channel height (m)
    "Wgc": 4.0e-4,          # Gas-channel width (m)
    "Lgc": 1.287,           # Gas-channel length (m)

    # Porous-media and electrochemical parameters.
    "epsilon_gdl": 0.7504,
    "epsilon_cl": 0.252336,
    "epsilon_mc": 0.3312,
    "epsilon_c": 0.257928,
    "e": 4,
    "kappa_co": 37.2,
    "Re": 2.2e-7,
    "tau": 1.12,
    "i0_c_ref": 7.39,
    "kappa_c": 3.4653,
    "C_scl": 1.0e8,
    "a_slim": 0.2172,
    "b_slim": 0.1431,
    "a_switch": 0.5,
    "Hcl": 1.31e-5,
    "Hgdl": 3.79e-4,
    "OCV": 0.98,

    # Numerical discretization and model choices.
    "max_step": 0.1,
    "n_gdl": 10,
    "n_mem": 10,
    "n_group_pt": 50,
    "t_purge": [2.4, 15],
    "type_fuel_cell": "LEV-200",
    "type_control": "Phi_des",
    "type_purge": "constant_purge",
    "aux_system": False,
}


# =============================================================================
# USER-ACCESSIBLE OPERATING CONDITIONS
# =============================================================================

OPERATING_CONDITIONS = {
    "Tfc": 323.15,          # Fuel-cell temperature (K)
    "Pa_des": 1.30e5,      # Desired anode pressure (Pa)
    "Pc_des": 1.30e5,      # Desired cathode pressure (Pa)
    "Phi_a_des": 0.0,      # Desired anode relative humidity (-)
    "Phi_c_des": 0.0,      # Desired cathode relative humidity (-)
    "Sa": 1.2,             # Anode stoichiometry (-)
    "Sc": 2.5,             # Cathode stoichiometry (-)
    "Imin_aux": 10.0,      # Minimum auxiliary-system current (A)
}


# =============================================================================
# USER-ACCESSIBLE PT DEGRADATION CONSTANTS
# =============================================================================

KINETIC_CONSTANTS = {
    "k1": 1.735e-17,
    "k1_ref": 1.0e-18,
    "k2": 6.335e-15,
    "k2_ref": 1.0e-13,
    "k3": 1.13e-14,
    "krdp": 1.77e-11,
    "k4": 0.0,
    "k5": 0.0,
    "kdet_ref": 1.32e-23,
}


# =============================================================================
# USER-ACCESSIBLE AST, SOLVER, AND OUTPUT OPTIONS
# =============================================================================

AST_OPTIONS = {
    # "Dual-scale" without auxiliaries reproduces the notebook.  "Dynamic"
    # requires aux_system=True; the shared GUI runner reconciles mismatches in
    # the same way as the Simulation page.
    "model_variant": "Dual-scale",
    "aux_system": False,

    # AST square-wave current-density profile.
    "profile_kind": "Step",
    "profile_cfg": {
        "step_tstart": 0.0,
        "step_tend": 6.0,
        "i_low": 1000.0,       # A/m^2
        "i_high": 14500.0,     # A/m^2
        "tau_switch": 1.5,     # Transition smoothing (s)
        "t_switch": 1.0,       # Switching time (s)
    },

    # A 30,000 s span with a six-second period executes 5,000 AST cycles.
    "t_start": 0.0,            # s
    "t_end": 30000.0,          # s
    "max_step": 0.1,           # s
    "method": "BDF",          # BDF, Radau, LSODA, or RK45
    "compare_with_0d": True,
}

OUTPUT_OPTIONS = {
    "show_progress": True,
    "show_plots": True,
    "save_summary": True,
    "save_figures": False,
    "suppress_model_warnings": True,
}

# Duration used only by ``python AST_example.py --verify``.
VERIFY_DURATION_SECONDS = 0.2


import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np


# Locate the project from the script path, not from the caller's current
# directory, so ``python D:/MFC2024/AST_example.py`` works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (
    (PROJECT_ROOT / "config" / "initialize.py").exists()
    and (PROJECT_ROOT / "gui" / "runner.py").exists()
):
    raise RuntimeError(
        f"Could not locate the MFC2024 project beside {Path(__file__).name}."
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.options import build_profile_func
from gui.runner import run as run_simulation, run_0d_companion


SUMMARY_PATH = PROJECT_ROOT / "AST_result.json"
RESPONSE_FIGURE_PATH = PROJECT_ROOT / "AST_response.png"
DEGRADATION_FIGURE_PATH = PROJECT_ROOT / "AST_degradation.png"


def _validate_settings(parameters, operating_conditions, options):
    """Reject invalid user settings before starting a long integration."""
    if options["profile_kind"] != "Step":
        raise ValueError("This AST example requires profile_kind='Step'.")
    if options["model_variant"] not in ("Dual-scale", "Dynamic"):
        raise ValueError(
            "AST model_variant must be 'Dual-scale' or 'Dynamic'; the Static "
            "model has no degradation trajectory."
        )
    if float(options["t_end"]) <= float(options["t_start"]):
        raise ValueError("AST_OPTIONS['t_end'] must be greater than t_start.")
    if float(options["max_step"]) <= 0.0:
        raise ValueError("AST_OPTIONS['max_step'] must be positive.")
    if options["method"] not in ("BDF", "Radau", "LSODA", "RK45"):
        raise ValueError("Unsupported ODE method in AST_OPTIONS['method'].")

    profile = options["profile_cfg"]
    if float(profile["step_tend"]) <= float(profile["step_tstart"]):
        raise ValueError("The Step-profile period must be positive.")
    if float(profile["i_low"]) < 0.0 or float(profile["i_high"]) < 0.0:
        raise ValueError("AST current densities must be non-negative.")

    for name in ("Aact", "Hmem", "Hcl", "Hgdl"):
        if float(parameters[name]) <= 0.0:
            raise ValueError(f"PARAMETERS[{name!r}] must be positive.")
    for name in ("Tfc", "Pa_des", "Pc_des", "Sa", "Sc"):
        if float(operating_conditions[name]) <= 0.0:
            raise ValueError(
                f"OPERATING_CONDITIONS[{name!r}] must be positive."
            )
    for name in ("Phi_a_des", "Phi_c_des"):
        humidity = float(operating_conditions[name])
        if not 0.0 <= humidity <= 1.0:
            raise ValueError(
                f"OPERATING_CONDITIONS[{name!r}] must be between 0 and 1."
            )


def _requested_duration_override(cli_duration=None):
    """Resolve an optional short-run duration from CLI or the notebook env var."""
    if cli_duration is not None:
        duration = float(cli_duration)
    else:
        environment_value = os.environ.get("MFC_AST_VERIFY_SECONDS")
        duration = None if environment_value is None else float(environment_value)
    if duration is not None and duration <= 0.0:
        raise ValueError("The AST duration override must be positive.")
    return duration


def _prepare_run(duration_override=None):
    """Copy settings, build the GUI-equivalent profile, and calculate spans."""
    parameters = deepcopy(PARAMETERS)
    operating_conditions = deepcopy(OPERATING_CONDITIONS)
    kinetic_constants = deepcopy(KINETIC_CONSTANTS)
    options = deepcopy(AST_OPTIONS)
    _validate_settings(parameters, operating_conditions, options)

    # Keep the parameter read by the ODE consistent with the runner routing
    # choice, without requiring users to edit the same switch twice.
    parameters["aux_system"] = bool(options["aux_system"])

    profile_state = {
        "profile_kind": options["profile_kind"],
        "profile_cfg": options["profile_cfg"],
        "params": parameters,
    }
    current_density = build_profile_func(profile_state)

    configured_start = float(options["t_start"])
    configured_end = float(options["t_end"])
    simulation_end = configured_end
    if duration_override is not None:
        simulation_end = min(
            configured_end,
            configured_start + float(duration_override),
        )
    simulation_span = (configured_start, simulation_end)

    profile = options["profile_cfg"]
    cycle_period = float(profile["step_tend"] - profile["step_tstart"])
    configured_cycles = (configured_end - configured_start) / cycle_period
    executed_cycles = (simulation_end - configured_start) / cycle_period

    return {
        "parameters": parameters,
        "operating_conditions": operating_conditions,
        "kinetic_constants": kinetic_constants,
        "options": options,
        "current_density": current_density,
        "configured_start": configured_start,
        "configured_end": configured_end,
        "simulation_end": simulation_end,
        "simulation_span": simulation_span,
        "cycle_period": cycle_period,
        "configured_cycles": configured_cycles,
        "executed_cycles": executed_cycles,
    }


def _print_run_header(run):
    options = run["options"]
    profile = options["profile_cfg"]
    print("=" * 72)
    print("PEMFC accelerated stress test")
    print(f"Model: {options['model_variant']}")
    print(f"Auxiliary system: {options['aux_system']}")
    print(
        "Step profile: "
        f"{profile['i_low']:.0f} to {profile['i_high']:.0f} A/m^2, "
        f"period {run['cycle_period']:g} s"
    )
    print(
        f"Configured span: {run['configured_start']:g} to "
        f"{run['configured_end']:g} s "
        f"({run['configured_cycles']:g} cycles)"
    )
    print(
        f"Executed span: {run['simulation_span'][0]:g} to "
        f"{run['simulation_span'][1]:g} s "
        f"({run['executed_cycles']:g} cycles)"
    )
    print(
        f"Solver: {options['method']}, "
        f"max_step={options['max_step']:g} s"
    )
    print(f"0D comparison: {options['compare_with_0d']}")
    print("=" * 72)


class _TerminalProgressBar:
    """Single-line terminal progress display driven by simulated time."""

    def __init__(self, label, t_span, enabled=True, width=30):
        self.label = str(label)
        self.start = float(t_span[0])
        self.end = float(t_span[1])
        self.enabled = bool(enabled)
        self.width = int(width)
        self.last_draw = 0.0
        self.last_fraction = -1.0
        self.open = False

    def update(self, fraction, simulated_time):
        if not self.enabled:
            return
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if fraction <= self.last_fraction:
            return
        now = time.perf_counter()
        # Limit terminal traffic while always drawing start and completion.
        if (0.0 < fraction < 1.0
                and now - self.last_draw < 0.15
                and fraction > self.last_fraction):
            return
        completed = int(round(self.width * fraction))
        bar = "#" * completed + "-" * (self.width - completed)
        line = (
            f"\r{self.label:<15} [{bar}] {fraction * 100:6.1f}%  "
            f"simulated {float(simulated_time):.3g}/{self.end:.3g} s"
        )
        print(line, end="", flush=True)
        self.last_draw = now
        self.last_fraction = fraction
        if fraction >= 1.0:
            print(flush=True)
            self.open = False
        else:
            self.open = True

    def close(self):
        if self.enabled and self.open:
            print(flush=True)
            self.open = False


def _run_main_model(run, progress_callback=None):
    options = run["options"]
    model, solution, status = run_simulation(
        params=run["parameters"],
        op_inputs=run["operating_conditions"],
        model_variant=options["model_variant"],
        profile_func=run["current_density"],
        t_span=run["simulation_span"],
        max_step=float(options["max_step"]),
        method=options["method"],
        aux_system=bool(options["aux_system"]),
        kinetic_consts=run["kinetic_constants"],
        progress_callback=progress_callback,
    )
    if not status["success"]:
        raise RuntimeError(f"Simulation failed: {status['message']}")

    print(f"Resolved model: {status['model_variant']}")
    print(f"States: {status['n_states']}")
    print(f"Accepted time points: {status['n_steps']}")
    print(f"Runtime: {status['runtime_s']:.3f} s")
    print(f"Solver method: {status['method_actual']}")
    print(f"Solver message: {status['message']}")
    return model, solution, status


def _empty_benchmark():
    return {
        "status": {
            "success": False,
            "runtime_s": 0.0,
            "n_steps": 0,
            "message": "0D comparison disabled.",
        },
        "time_s": np.array([], dtype=float),
        "cycles": np.array([], dtype=float),
        "hmem_um": np.array([], dtype=float),
        "ecsa_ratio": np.array([], dtype=float),
    }


def _run_benchmark(run, progress_callback=None):
    benchmark = _empty_benchmark()
    options = run["options"]
    if not options["compare_with_0d"]:
        return benchmark

    raw = run_0d_companion(
        params=run["parameters"],
        op_inputs=run["operating_conditions"],
        profile_func=run["current_density"],
        t_span=run["simulation_span"],
        max_step=float(options["max_step"]),
        method=options["method"],
        progress_callback=progress_callback,
    )
    status = raw["status"]
    benchmark["status"] = status
    if not status["success"]:
        warnings.warn(f"0D benchmark failed: {status['message']}")
        return benchmark

    variables = raw["variables"]
    electrochemistry = raw["echem_traj"]
    time_s = np.asarray(
        electrochemistry.get("t", variables.get("t", [])), dtype=float
    )
    hmem_um = np.asarray(variables.get("Hmem", []), dtype=float) * 1e6
    ecsa_ratio = np.asarray(electrochemistry.get("S_N", []), dtype=float)
    if not (time_s.size and hmem_um.size and ecsa_ratio.size):
        raise RuntimeError(
            "0D benchmark did not recover Hmem and S_N trajectories."
        )

    benchmark.update({
        "time_s": time_s,
        "cycles": (
            (time_s - run["configured_start"]) / run["cycle_period"]
        ),
        "hmem_um": hmem_um,
        "ecsa_ratio": ecsa_ratio,
    })
    print("Resolved model: 0D benchmark")
    print(f"Accepted time points: {status['n_steps']}")
    print(f"Runtime: {status['runtime_s']:.3f} s")
    print(f"Solver message: {status['message']}")
    return benchmark


def _extract_main_trajectories(model, run):
    time_s = np.asarray(model.variables["t"], dtype=float)
    voltage = np.asarray(model.echem_traj["Ucell"], dtype=float)
    ecsa_ratio = np.asarray(model.echem_traj["S_N"], dtype=float)
    membrane_um = np.asarray(model.variables["delta_mem"], dtype=float) * 1e6

    plottable_count = min(time_s.size, voltage.size, ecsa_ratio.size)
    if plottable_count == 0 or membrane_um.size == 0:
        raise RuntimeError("The simulation returned no plottable AST data.")
    return {
        "time_s": time_s,
        "voltage_V": voltage,
        "ecsa_ratio": ecsa_ratio,
        "membrane_um": membrane_um,
        "cycles": (
            (time_s - run["configured_start"]) / run["cycle_period"]
        ),
        "plottable_count": plottable_count,
    }


def _build_response_figure(trajectories, run):
    count = trajectories["plottable_count"]
    response_time = trajectories["time_s"][:count]
    response_end = min(
        float(response_time[-1]),
        run["configured_start"] + 3.0 * run["cycle_period"],
    )
    mask = response_time <= response_end
    current_A = np.asarray(
        [run["current_density"](t) for t in response_time]
    ) * run["parameters"]["Aact"]

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 6.5), sharex=True)
    axes[0].plot(response_time[mask], current_A[mask], color="#1f5a94")
    axes[0].set_ylabel("Current (A)")
    axes[1].plot(
        response_time[mask],
        trajectories["voltage_V"][:count][mask],
        color="#b23a2b",
    )
    axes[1].set_ylabel("Cell voltage (V)")
    axes[2].plot(
        response_time[mask],
        trajectories["ecsa_ratio"][:count][mask],
        color="#2f7d4a",
    )
    axes[2].set_ylabel("Normalized ECSA (-)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def _build_degradation_figure(trajectories, benchmark):
    cycles = trajectories["cycles"]
    membrane_um = trajectories["membrane_um"]
    ecsa_ratio = trajectories["ecsa_ratio"]

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    main_membrane_count = min(cycles.size, membrane_um.size)
    axes[0].plot(
        cycles[:main_membrane_count],
        membrane_um[:main_membrane_count],
        color="#264f9e",
        linewidth=1.5,
        label="1D",
    )
    if benchmark["status"]["success"]:
        benchmark_membrane_count = min(
            benchmark["cycles"].size, benchmark["hmem_um"].size
        )
        axes[0].plot(
            benchmark["cycles"][:benchmark_membrane_count],
            benchmark["hmem_um"][:benchmark_membrane_count],
            color="#009E73",
            linestyle="--",
            linewidth=1.5,
            label="0D benchmark",
        )
    axes[0].set_title("Membrane thinning")
    axes[0].set_xlabel("AST cycles (-)")
    axes[0].set_ylabel("Membrane thickness (um)")
    axes[0].grid(alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend(fontsize=8, loc="best")

    main_ecsa_count = min(cycles.size, ecsa_ratio.size)
    axes[1].plot(
        cycles[:main_ecsa_count],
        ecsa_ratio[:main_ecsa_count],
        color="#2f496e",
        linewidth=1.5,
        label="1D",
    )
    if benchmark["status"]["success"]:
        benchmark_ecsa_count = min(
            benchmark["cycles"].size, benchmark["ecsa_ratio"].size
        )
        axes[1].plot(
            benchmark["cycles"][:benchmark_ecsa_count],
            benchmark["ecsa_ratio"][:benchmark_ecsa_count],
            color="#009E73",
            linestyle="--",
            linewidth=1.5,
            label="0D benchmark",
        )
    axes[1].set_title("Pt active surface")
    axes[1].set_xlabel("AST cycles (-)")
    axes[1].set_ylabel("Normalized ECSA (-)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")

    expected_lines = 2 if benchmark["status"]["success"] else 1
    if len(axes[0].lines) != expected_lines:
        raise RuntimeError("Unexpected membrane-comparison plot layout.")
    if len(axes[1].lines) != expected_lines:
        raise RuntimeError("Unexpected ECSA-comparison plot layout.")
    figure.tight_layout()
    return figure


def _build_summary(run, status, trajectories, benchmark):
    summary = {
        "configured_duration_s": run["configured_end"] - run["configured_start"],
        "executed_duration_s": run["simulation_end"] - run["configured_start"],
        "configured_cycles": run["configured_cycles"],
        "executed_cycles": run["executed_cycles"],
        "model_variant": status["model_variant"],
        "auxiliary_system": bool(run["options"]["aux_system"]),
        "profile_kind": run["options"]["profile_kind"],
        "final_1D_cell_voltage_V": float(trajectories["voltage_V"][-1]),
        "final_1D_membrane_thickness_um": float(
            trajectories["membrane_um"][-1]
        ),
        "final_1D_normalized_ECSA": float(trajectories["ecsa_ratio"][-1]),
        "runtime_1D_s": float(status["runtime_s"]),
        "solver_method": status["method_actual"],
        "0D_comparison_success": bool(benchmark["status"]["success"]),
    }
    if benchmark["status"]["success"]:
        summary.update({
            "final_0D_membrane_thickness_um": float(benchmark["hmem_um"][-1]),
            "final_0D_normalized_ECSA": float(benchmark["ecsa_ratio"][-1]),
            "runtime_0D_s": float(benchmark["status"]["runtime_s"]),
        })
    return summary


def _write_outputs(summary, figures, output_options):
    if output_options["save_summary"]:
        with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        print(f"Result summary saved: {SUMMARY_PATH}")
    if output_options["save_figures"]:
        figures["response"].savefig(RESPONSE_FIGURE_PATH, dpi=160,
                                    bbox_inches="tight")
        figures["degradation"].savefig(DEGRADATION_FIGURE_PATH, dpi=160,
                                       bbox_inches="tight")
        print(f"Figures saved: {RESPONSE_FIGURE_PATH}, {DEGRADATION_FIGURE_PATH}")


def run_ast(*, duration_override=None, output_options=None):
    """Execute the configured AST and return its result summary."""
    output_options = deepcopy(
        OUTPUT_OPTIONS if output_options is None else output_options
    )
    run = _prepare_run(duration_override=duration_override)
    _print_run_header(run)

    warning_context = (
        warnings.catch_warnings()
        if output_options["suppress_model_warnings"]
        else nullcontext()
    )
    with warning_context:
        if output_options["suppress_model_warnings"]:
            warnings.simplefilter("ignore")
        main_progress = _TerminalProgressBar(
            "1D model", run["simulation_span"],
            enabled=output_options.get("show_progress", True),
        )
        main_progress.update(0.0, run["simulation_span"][0])
        try:
            model, _solution, status = _run_main_model(
                run, progress_callback=main_progress.update,
            )
        finally:
            main_progress.close()

        benchmark_progress = _TerminalProgressBar(
            "0D benchmark", run["simulation_span"],
            enabled=(output_options.get("show_progress", True)
                     and run["options"]["compare_with_0d"]),
        )
        if run["options"]["compare_with_0d"]:
            benchmark_progress.update(0.0, run["simulation_span"][0])
        try:
            benchmark = _run_benchmark(
                run, progress_callback=benchmark_progress.update,
            )
        finally:
            benchmark_progress.close()

    trajectories = _extract_main_trajectories(model, run)
    figures = {
        "response": _build_response_figure(trajectories, run),
        "degradation": _build_degradation_figure(trajectories, benchmark),
    }
    summary = _build_summary(run, status, trajectories, benchmark)
    _write_outputs(summary, figures, output_options)

    print("AST result summary:")
    print(json.dumps(summary, indent=2))
    if output_options["show_plots"]:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)
    return summary


def verify_ast_file(duration=VERIFY_DURATION_SECONDS):
    """Run a reduced 1D + 0D AST through the same production code path."""
    verification_output = deepcopy(OUTPUT_OPTIONS)
    verification_output.update({
        "show_plots": False,
        "save_summary": False,
        "save_figures": False,
    })
    summary = run_ast(
        duration_override=float(duration),
        output_options=verification_output,
    )
    if not 0.0 < summary["executed_duration_s"] <= float(duration):
        raise AssertionError("AST verification duration override was not honored.")
    if not summary["0D_comparison_success"]:
        raise AssertionError("The AST 0D companion did not complete successfully.")
    for key in (
        "final_1D_cell_voltage_V",
        "final_1D_membrane_thickness_um",
        "final_1D_normalized_ECSA",
        "final_0D_membrane_thickness_um",
        "final_0D_normalized_ECSA",
    ):
        if not np.isfinite(float(summary[key])):
            raise AssertionError(f"AST verification produced non-finite {key}.")
    print("All AST_example.py verification checks passed.")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run a short 1D + 0D AST without saving or showing plots",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="temporarily limit the executed duration in seconds",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="do not open matplotlib windows",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write AST_result.json or figure files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the live terminal progress bars",
    )
    arguments = parser.parse_args(argv)

    if arguments.verify:
        verify_duration = (
            VERIFY_DURATION_SECONDS
            if arguments.duration is None
            else arguments.duration
        )
        verify_ast_file(verify_duration)
        return 0

    duration_override = _requested_duration_override(arguments.duration)
    output_options = deepcopy(OUTPUT_OPTIONS)
    if arguments.no_plots:
        output_options["show_plots"] = False
    if arguments.no_save:
        output_options["save_summary"] = False
        output_options["save_figures"] = False
    if arguments.no_progress:
        output_options["show_progress"] = False
    run_ast(
        duration_override=duration_override,
        output_options=output_options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
