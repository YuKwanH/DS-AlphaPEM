"""PEMFC simulator - raw-code single-simulation entry point.

Mirrors the GUI Simulation page so any run reachable through the browser
is reachable from this script:

  * Same model variants exposed by the GUI (``Dual-scale`` and
    ``Dynamic``; ``Static`` polarization sweep is intentionally
    excluded because the GUI hides it too).
  * Same five test profiles built by ``gui.profiles``: Constant, Step,
    Polarization, EIS, AST cycling.
  * Same auto-routing (``Dual-scale`` + aux -> ``PEMFC_dyn``;
    ``Dynamic`` + no-aux -> ``PEMFC``).
  * Same 0D benchmark companion toggle.
  * Same ODE solver dispatch through ``gui.runner.run()`` — the
    integration loop is not re-implemented here.

Edit the USER SETTINGS block below, then run::

    python main.py

For parameter calibration over experimental data see ``calibration.py``.
For the interactive web GUI see ``run_gui.bat`` (or ``streamlit run
gui/app.py``).
"""
from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np

from config.initialize import parameters, operating_inputs
from gui import profiles
from gui.runner import run as run_simulation
from gui.runner import run_0d_companion


# =============================================================================
# USER SETTINGS  --  edit these, save, then re-run `python main.py`
# =============================================================================

# ---- Model variant (matches GUI § 2 Model variant selector) ---------------
# Options: "Dual-scale" (1-D, no BoP) | "Dynamic" (1-D + full BoP).
# The GUI does not expose "Static" here, so neither does this script.
MODEL_VARIANT = "Dual-scale"

# Balance-of-plant equations. Auto-routes exactly like the GUI:
#   Dual-scale + True  -> PEMFC_dyn (auto-promote; PEMFC has no BoP code)
#   Dual-scale + False -> PEMFC
#   Dynamic    + True  -> PEMFC_dyn
#   Dynamic    + False -> PEMFC     (auto-demote; PEMFC_dyn needs BoP)
AUX_SYSTEM = False

# ---- Operating point (matches GUI § 1 Operating section) ------------------
TFC       = 353.15   # K   -- cell temperature
PA_DES    = 1.8e5    # Pa  -- anode supply pressure
PC_DES    = 1.8e5    # Pa  -- cathode supply pressure
PHI_A_DES = 0.0      # -   -- anode inlet relative humidity  (0 = dry)
PHI_C_DES = 0.85     # -   -- cathode inlet relative humidity
SA        = 1.2      # -   -- anode stoichiometry
SC        = 2.0      # -   -- cathode stoichiometry

# ---- Test profile (matches GUI § 2 Test profile selector) -----------------
# Options: "Constant" | "Step" | "Polarization" | "EIS" | "AST cycling".
# Only the block matching PROFILE_KIND is read; the rest are ignored.
PROFILE_KIND = "AST cycling"

# Constant -------------------------------------------------------------------
I_CONST_A_M2 = 4000.0            # A/m^2 constant current density

# Step  (periodic tanh-smoothed square load; matches gui.profiles.step) -------
STEP_TSTART_S  = 0.0             # s      period start
STEP_TEND_S    = 6.0             # s      period end
STEP_I_LOW     = 20.0            # A/m^2  low level
STEP_I_HIGH    = 12000.0         # A/m^2  high level
STEP_TAU_S     = 1.0             # s      ramp-begin smoothing
STEP_T_SWITCH  = 3.0             # s      ramp duration

# Polarization ramp (transient staircase; NOT the algebraic Static sweep) ----
POLAR_I_MAX_A_M2 = 16500.0       # A/m^2  peak current density
POLAR_N_STEPS    = 30            # -      number of staircase steps
POLAR_T_PER_STEP = 60.0          # s      dwell per step

# EIS  (small-signal sinusoid on a DC bias) ---------------------------------
EIS_I_DC_A_M2 = 10000.0          # A/m^2  DC bias
EIS_RATIO     = 0.05             # -      AC amplitude / DC bias
EIS_FREQ_HZ   = 1.0              # Hz     excitation frequency

# AST cycling ---------------------------------------------------------------
# (currents here are TOTAL cell current in AMPERES, converted to
# current density inside gui.profiles.ast_cycling using Aact)
AST_PERIOD_S  = 6.0              # s      full cycle period
AST_I_LOW_A   = 1.0              # A      low level
AST_I_HIGH_A  = 25.8             # A      high level
AST_SMOOTHING = 4.0              # -      transition sharpness

# ---- Time span & solver (matches GUI § 2 Time span & solver) --------------
T_SPAN   = (0.0, 20.0)           # (t_start, t_end) in s
MAX_STEP = 0.1                   # s      solver max_step
METHOD   = "BDF"                 # "BDF" | "Radau" | "LSODA" | "RK45"

# ---- 0D benchmark companion (matches GUI "Compare with 0D benchmark") -----
BENCHMARK_0D = False             # True -> also run PEMFC_0D for overlay

# ---- Plotting -------------------------------------------------------------
SHOW_PLOTS = True                # True -> open the matplotlib window at end


# =============================================================================
# Run --  no editing needed below this line for a normal simulation
# =============================================================================

_PROFILE_BUILDERS = {"Constant", "Step", "Polarization", "EIS", "AST cycling"}


def _build_op():
    op = deepcopy({k: v for k, v in operating_inputs.items()
                   if k != "current_density"})
    op["Tfc"]       = TFC
    op["Pa_des"]    = PA_DES
    op["Pc_des"]    = PC_DES
    op["Phi_a_des"] = PHI_A_DES
    op["Phi_c_des"] = PHI_C_DES
    op["Sa"]        = SA
    op["Sc"]        = SC
    return op


def _build_profile(Aact):
    if PROFILE_KIND not in _PROFILE_BUILDERS:
        raise ValueError(f"Unknown PROFILE_KIND: {PROFILE_KIND!r}. "
                         f"Choose one of {sorted(_PROFILE_BUILDERS)}.")
    if PROFILE_KIND == "Constant":
        return profiles.constant(I_CONST_A_M2)
    if PROFILE_KIND == "Step":
        return profiles.step(
            STEP_TSTART_S, STEP_TEND_S, STEP_I_LOW, STEP_I_HIGH,
            STEP_TAU_S, STEP_T_SWITCH,
        )
    if PROFILE_KIND == "Polarization":
        return profiles.polarization_ramp(
            POLAR_I_MAX_A_M2, POLAR_N_STEPS, POLAR_T_PER_STEP,
        )
    if PROFILE_KIND == "EIS":
        return profiles.eis(EIS_I_DC_A_M2, EIS_RATIO, EIS_FREQ_HZ)
    # AST cycling — currents given in Amperes, needs Aact for density conversion
    return profiles.ast_cycling(
        AST_PERIOD_S, AST_I_LOW_A, AST_I_HIGH_A, AST_SMOOTHING, Aact,
    )


def _extract_ucell_and_ifc(model):
    """Pull (t, i_fc, Ucell) from whichever storage the model uses.

    ``PEMFC`` / ``PEMFC_0D`` store trajectories in ``echem_traj``;
    ``PEMFC_dyn`` originally used ``ec_kinetics`` — the GUI runner
    normalises to ``echem_traj`` so this loop handles all three variants.
    """
    t_var = model.variables.get("t", None)
    t = np.asarray(t_var) if t_var is not None else None
    src_chain = (getattr(model, "echem_traj", None),
                 getattr(model, "ec_kinetics", None),
                 model.variables)

    def _get(key):
        for src in src_chain:
            if src and key in src and hasattr(src[key], "__len__") and len(src[key]):
                return np.asarray(src[key], dtype=float)
        return None

    return t, _get("i_fc"), _get("Ucell")


def _print_header():
    print("=" * 74)
    print(" PEMFC simulator - raw-code single run")
    print("=" * 74)
    print(f"   Model variant : {MODEL_VARIANT}   (aux_system={AUX_SYSTEM})")
    print(f"   Operating     : Tfc={TFC} K, Pa/Pc={PA_DES/1e5:.2f}/{PC_DES/1e5:.2f} bar, "
          f"RHa/RHc={PHI_A_DES}/{PHI_C_DES}, Sa/Sc={SA}/{SC}")
    print(f"   Profile       : {PROFILE_KIND}")
    print(f"   Time span     : {T_SPAN[0]:.1f} -> {T_SPAN[1]:.1f} s   "
          f"(max_step={MAX_STEP}, method={METHOD})")
    print(f"   0D benchmark  : {BENCHMARK_0D}")
    print("=" * 74)


def _plot(model, bench_info):
    t, i_fc, Ucell = _extract_ucell_and_ifc(model)
    if Ucell is None or len(Ucell) == 0:
        print("   (nothing to plot — no Ucell trajectory)")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    # -- Load current panel --
    if i_fc is not None and len(i_fc):
        ax[0].plot(t[:len(i_fc)], i_fc, linewidth=1.4,
                   color="tab:blue", label="1-D")
    if bench_info and bench_info.get("echem_traj", {}).get("i_fc") is not None:
        b_t = np.asarray(bench_info["variables"].get("t", []))
        b_i = np.asarray(bench_info["echem_traj"]["i_fc"])
        n = min(len(b_t), len(b_i))
        if n:
            ax[0].plot(b_t[:n], b_i[:n], linewidth=1.2, linestyle="--",
                       color="tab:green", label="0D benchmark")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Current density (A/m$^2$)")
    ax[0].set_title("Load current")
    ax[0].grid(True, alpha=0.3)
    if bench_info is not None:
        ax[0].legend(fontsize=8)

    # -- Cell voltage panel --
    ax[1].plot(t[:len(Ucell)], Ucell, linewidth=1.5, color="tab:orange",
               label="1-D")
    if bench_info and bench_info.get("echem_traj", {}).get("Ucell") is not None:
        b_t = np.asarray(bench_info["variables"].get("t", []))
        b_U = np.asarray(bench_info["echem_traj"]["Ucell"])
        n = min(len(b_t), len(b_U))
        if n:
            ax[1].plot(b_t[:n], b_U[:n], linewidth=1.2, linestyle="--",
                       color="tab:green", label="0D benchmark")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Cell voltage (V)")
    ax[1].set_title("Cell voltage")
    ax[1].grid(True, alpha=0.3)
    if bench_info is not None:
        ax[1].legend(fontsize=8)

    fig.suptitle(f"PEMFC simulator -- {MODEL_VARIANT} ({PROFILE_KIND})  "
                 f"Tfc={TFC} K  Pc={PC_DES/1e5:.2f} bar")
    plt.tight_layout()
    plt.show()


def main():
    p = deepcopy(parameters)
    op = _build_op()
    profile_func = _build_profile(p["Aact"])

    _print_header()

    t0 = time.perf_counter()
    model, _sol_or_polar, status = run_simulation(
        params=p, op_inputs=op,
        model_variant=MODEL_VARIANT,
        profile_func=profile_func,
        t_span=T_SPAN,
        max_step=MAX_STEP,
        method=METHOD,
        polar_sweep=None,      # unused for transient runs; matches the GUI
        aux_system=AUX_SYSTEM,
    )
    wall = time.perf_counter() - t0

    tag = "OK" if status["success"] else "FAILED"
    print(f"\n   {tag}  -  variant={status['model_variant']}, "
          f"runtime={status['runtime_s']:.2f} s, wall={wall:.2f} s")
    if status.get("n_steps") is not None:
        print(f"   n_steps       : {status['n_steps']}")
    if status.get("n_states") is not None:
        print(f"   n_states      : {status['n_states']}")
    if status.get("message"):
        print(f"   message       : {status['message']}")
    # Same "auto-switch must be visible" rule the GUI enforces — surface it
    # loudly instead of leaving it as a suffix on the variant label.
    if status.get("fallback_used"):
        requested = status.get("method_requested", METHOD)
        print("")
        print("   " + "!" * 60)
        print(f"   WARNING: Solver auto-switched from {requested} -> LSODA.")
        print(f"            {requested} hit a transient NaN mid-integration;")
        print("            the runner re-ran with LSODA. The trajectory above")
        print(f"            is from LSODA, not {requested}. If you need a strict")
        print("            single-solver run, pick a different METHOD or use")
        print("            calibration.py (which never switches).")
        print("   " + "!" * 60)
    if not status["success"]:
        return

    # ---- Optional 0D benchmark companion (same code path the GUI uses) ---
    bench_info = None
    if BENCHMARK_0D:
        print()
        print("   Running 0D benchmark companion ...")
        bench_info = run_0d_companion(
            params=p, op_inputs=op,
            profile_func=profile_func, t_span=T_SPAN,
            max_step=MAX_STEP, method=METHOD,
        )
        bs = bench_info["status"]
        print(f"   0D benchmark  : runtime={bs.get('runtime_s', 0):.2f} s, "
              f"success={bs['success']}"
              + (f", {bs.get('message', '')}" if bs.get("message") else ""))

    _t, i_fc, Ucell = _extract_ucell_and_ifc(model)
    if Ucell is not None and len(Ucell):
        print(f"\n   Ucell (final) : {Ucell[-1]:.4f} V")
    if i_fc is not None and len(i_fc):
        print(f"   i_fc  (final) : {i_fc[-1]:.1f} A/m^2")

    if SHOW_PLOTS:
        _plot(model, bench_info)


if __name__ == "__main__":
    main()
