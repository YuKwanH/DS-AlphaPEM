"""
AST simulation example -- degradation assessment across the model hierarchy.
===========================================================================

Runs the *same* accelerated stress test (AST) square-wave load through the four
PEMFC model variants shipped in this project and compares the degradation they
predict:

    1. "Micro"          -- micro-scale model alone (model/state_eq.py::dxdt_PRD +
                           model/kinetic_eq.py). Pt particle-size distribution,
                           Pt-oxide coverage and membrane thinning ONLY. It has no
                           macro-scale transport, so the cell voltage that drives
                           the degradation kinetics must be PRESCRIBED as a square
                           wave (see `make_ast_voltage`).
    2. "0D dual-scale"  -- model/model.py::PEMFC_0D. Lumped (Pukrushpan-style)
                           electrochemistry two-way coupled to the micro-scale.
                           Ucell is COMPUTED from the load.
    3. "1D, no aux"     -- model/model.py::PEMFC with parameters["aux_system"]=False.
                           Through-plane resolved GDL/CL/membrane transport coupled
                           to the micro-scale; the BoP states are frozen.
    4. "1D + auxiliary" -- the SAME PEMFC class with aux_system=True, which switches
                           on the compressor / humidifier first-order dynamics
                           (state_eq.py::dxdt_CP).

The AST protocol
----------------
A DOE-style catalyst AST is defined by its VOLTAGE holds -- 0.60 V (loaded) and
0.95 V (near-OCV) -- because it is the POTENTIAL that drives Pt dissolution and
oxidation. These models, however, are CURRENT-driven. So at start-up the script
inverts the steady-state polarization curve of the reference model (the 1-D, no-aux
one) to find the two currents that hold it at those voltages:

    0.95 V  ->  0.00034 A/cm^2      (near-OCV; the curve is very steep here)
    0.60 V  ->  1.30081 A/cm^2

and that ONE current profile is then applied to every model. The inversion costs
~1 min, so the result is cached in `AST_calibration.json`; --recalibrate redoes it.

The point of the comparison
---------------------------
All four integrate the *identical* Pt-dissolution / Pt-oxidation / detachment /
fluoride-release chemistry -- the micro model literally calls the same `dxdt_PRD`
routine the dual-scale models do. The ONLY thing that differs is how the cell
voltage Ucell(t), which drives that chemistry, is obtained:

    micro           : prescribed (it is handed the ideal 0.60/0.95 V square wave)
    0D dual-scale   : computed from a lumped voltage model
    1D dual-scale   : computed from resolved through-plane transport

So the spread between the curves measures how much the macro-scale modelling
assumptions matter for a degradation assessment.

A caveat you must read: one shared current != one shared AST
------------------------------------------------------------
A single current profile CANNOT hold every model at the same voltage, because each
model has its own polarization curve. At the shared high current (1.30 A/cm^2,
calibrated to put the 1-D model at 0.60 V) the models land at:

    1-D dual-scale :  0.60 V   <- on protocol, by construction
    0-D dual-scale :  0.19 V   <- FAR off protocol: its polarization curve is much
                                  steeper, and its own 0.60 V point is at only
                                  0.71 A/cm^2, roughly half the current

So the 0-D model is not running the intended 0.60/0.95 V AST at all -- it is being
driven ~0.4 V lower during the loaded hold, which suppresses Pt dissolution (a
strongly potential-dependent process). Its degradation is therefore NOT comparable
like-for-like with the 1-D model's here. The summary table prints the voltage each
model actually reaches ("U hold lo/hi") and flags any model that misses the target
by more than 25 mV as OFF PROTOCOL. Read that column before drawing conclusions.

The alternative -- give each model its own current profile, calibrated on its own
polarization curve, so all of them genuinely see 0.60/0.95 V -- would make the
degradation comparison apples-to-apples. This script does the shared-current variant
as requested; switching would mean calling `calibrate_ast_currents` per model.

READ THIS BEFORE INTERPRETING THE "auxiliary" CURVES
----------------------------------------------------
Curves 3 and 4 will land exactly on top of each other, and that is the correct
result for this codebase, not a bug in this script. Two independent findings, both
re-verified at run time by `diagnose_auxiliary_coupling()` below:

  * In PEMFC, the auxiliary states (Wcp, Wa_inj, Wc_inj) are DANGLING. `aux_system`
    only decides whether `dxdt_CP` integrates them; the cell never reads them back.
    The cathode inlet fluxes in inst_values.py::calculate_flows (~line 130) are
    built straight from the stoichiometry and the instantaneous load:
        Jc_in = (...) * (1/yO2_ext) * Sc * iload / (4F) * Aact / (Hgc*Wgc)
    i.e. the air supply is ideal and instantaneous, as if the compressor had zero
    time constant. So toggling aux_system cannot change Ucell, and therefore cannot
    change the degradation. It is bit-identical here.

  * PEMFC_dyn -- the ONLY class with a genuinely coupled balance-of-plant (supply
    and exhaust manifolds, back-pressure valves) -- carries the 32 degradation
    states but NEVER EVOLVES THEM: it calls neither dxdt_PRD nor any of the Pt
    kinetics, and all 32 degradation derivatives are identically zero. It is a
    performance/BoP model, so it cannot be used for a degradation assessment and is
    deliberately not one of the four curves.

Net: as the code stands there is no model that does 1-D degradation with a real,
coupled auxiliary system. Wiring dxdt_PRD into PEMFC_dyn (or making PEMFC's inlet
fluxes consume Wcp instead of Sc*iload) is what would make curve 4 differ from
curve 3 -- that is the open modelling gap this example exposes.

Outputs
-------
* `AST_comparison.png`     -- ECSA loss, membrane thinning, driving voltage
* `AST_comparison.csv`     -- end-of-test summary table
* `AST_calibration.json`   -- cached i(0.60 V) / i(0.95 V) for this operating point
All written next to this script (override with --outdir).

Usage
-----
    python "AST simulation example.py"                     # quick demo (30 cycles)
    python "AST simulation example.py" --cycles 500        # longer test
    python "AST simulation example.py" --models micro 0d   # subset only
    python "AST simulation example.py" --u-low 0.65        # a gentler AST
    python "AST simulation example.py" --recalibrate       # redo the inversion
    python "AST simulation example.py" --cycles 15000      # full AST (very slow!)

AST duration
------------
Set `N_CYCLES` in the "1. AST protocol" block below (or pass --cycles, which wins).
Total AST time = N_CYCLES * CYCLE_PERIOD.

A real DOE catalyst AST is ~15 000 cycles, which costs ~15 h of wall clock for all
four models (measured: the two 1-D models need ~1.5 s per cycle each; the micro and
0-D models are 20x cheaper). Short runs are fine for comparing models: degradation
is close to linear early on, so the ranking is unchanged.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; drop this line to show() interactively
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --------------------------------------------------------------------------- #
# Project root on sys.path (same idiom as the notebooks in simulation/)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "config" / "initialize.py").exists() and (p / "model" / "model.py").exists()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate the MFC2024 project root from this file.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.initialize import parameters, operating_inputs, init_x_for      # noqa: E402
from config.settings import solver_variable_names, solver_flux_names        # noqa: E402
from model.model import PEMFC, PEMFC_0D, PEMFC_dyn                          # noqa: E402
from model.coefficients import R                                            # noqa: E402
from model.inst_values import Cproton_CCL, initPRD, getECSA                 # noqa: E402
from model.kinetic_eq import (PtDissolution, PtOxidation, PtOxideDissolution,  # noqa: E402
                              PtDetachment, flourideReleaseRate)
from model.state_eq import dxdt_PRD                                         # noqa: E402
from modules.signals import generate_step_load, generate_constant_load      # noqa: E402


# =========================================================================== #
# 0. Progress reporting (dependency-free -- tqdm is not a project requirement)
# =========================================================================== #
class Progress:
    """A one-line progress bar that redraws in place.

    Falls back to periodic plain lines when stdout is not a TTY (piped to a file,
    run from a notebook, captured by CI), because \\r spam is unreadable there.

    Usage:
        with Progress(total=120.0, label="1-D model") as bar:
            bar.update(t)          # t is monotone-ish; the bar never goes backwards
    """

    WIDTH = 28

    def __init__(self, total, label, stream=sys.stdout, min_interval=0.15):
        self.total = float(total) if total and total > 0 else 1.0
        self.label = label
        self.stream = stream
        self.min_interval = min_interval
        self.tty = hasattr(stream, "isatty") and stream.isatty()
        self.current = 0.0
        self.t_start = time.perf_counter()
        self._last_draw = 0.0
        self._last_pct = -1.0
        self._closed = False

    def __enter__(self):
        self._draw(force=True)
        return self

    def __exit__(self, *exc):
        self.close(success=exc[0] is None)
        return False

    def update(self, current):
        # solve_ivp probes the RHS at trial times (rejected steps, finite-difference
        # Jacobian columns) that can sit behind or ahead of the accepted step, so
        # clamp to a monotone maximum -- otherwise the bar jitters backwards.
        self.current = min(max(self.current, float(current)), self.total)
        self._draw()

    def _draw(self, force=False):
        now = time.perf_counter()
        if not force and (now - self._last_draw) < self.min_interval:
            return
        self._last_draw = now
        frac = max(0.0, min(1.0, self.current / self.total))
        pct = 100.0 * frac
        elapsed = now - self.t_start
        eta = (elapsed / frac - elapsed) if frac > 0.02 else float("nan")

        if self.tty:
            filled = int(round(self.WIDTH * frac))
            bar = "#" * filled + "-" * (self.WIDTH - filled)
            eta_s = f"{eta:5.0f}s" if np.isfinite(eta) else "  ..s"
            self.stream.write(f"\r  {self.label:<38}[{bar}] {pct:5.1f}%  "
                              f"{elapsed:5.1f}s  ETA {eta_s}")
            self.stream.flush()
        elif pct - self._last_pct >= 25.0:            # non-TTY: one line per quarter
            self._last_pct = pct
            self.stream.write(f"  {self.label}: {pct:.0f}%  ({elapsed:.1f}s)\n")
            self.stream.flush()

    def close(self, success=True):
        if self._closed:
            return
        self._closed = True
        elapsed = time.perf_counter() - self.t_start
        if self.tty:
            mark = "done" if success else "FAILED"
            bar = "#" * self.WIDTH if success else "-" * self.WIDTH
            self.stream.write(f"\r  {self.label:<38}[{bar}] {mark:>6}  "
                              f"{elapsed:5.1f}s{' ' * 12}\n")
            self.stream.flush()


def with_progress(dxdt, t_end, label):
    """Wrap an ODE right-hand side so integrating it drives a progress bar.

    The bar tracks the solver's *simulated* time, not its work, so it advances
    unevenly: stiff stretches (the current edges) crawl, easy holds fly. The ETA is
    therefore a rough guide, not a promise.

    Returns (wrapped_rhs, bar). The caller owns the bar and must close() it.
    """
    bar = Progress(total=t_end, label=label)
    bar.__enter__()

    def wrapped(t, x):
        bar.update(t)
        return dxdt(t, x)

    return wrapped, bar


# =========================================================================== #
# 1. AST protocol
# =========================================================================== #
# ---- How long the AST runs. THIS is the knob to change the test duration. -----
# Total AST time = N_CYCLES * CYCLE_PERIOD seconds.
# MEASURED wall-clock cost per AST cycle (600-cycle verification run, 0.6/0.95 V
# protocol, BDF, max_step=0.1):
#   micro 0.019 s/cyc | 0-D 0.38 s/cyc | 1-D no-aux 1.53 s/cyc | 1-D aux 1.59 s/cyc
# so, running all four models together:
#   N_CYCLES =     30  ->    180 s simulated    ~2 min wall clock
#   N_CYCLES =    600  ->  3 600 s (1 h)        ~35 min
#   N_CYCLES = 10 000  -> 60 000 s              ~10 h  (~9 h of it is the two 1-D
#   N_CYCLES = 15 000  -> 90 000 s (full AST)   ~15 h   models; if you don't need
#                                                       them: --models micro 0d
#                                                       is ~20x cheaper)
# Overridden on the command line by --cycles.
N_CYCLES     = 10000           # -      -- number of AST cycles to simulate

CYCLE_PERIOD = 6.0          # s      -- one AST cycle (matches simulation/Test_AST)
TAU_SWITCH   = 1.5          # s      -- rising edge, measured into the period
T_SWITCH     = 1.0          # s      -- edge duration (99.7 % of the transition)

# Total simulated AST time [s]. Derived -- change N_CYCLES or CYCLE_PERIOD, not this.
AST_DURATION = N_CYCLES * CYCLE_PERIOD

# The AST is defined by its VOLTAGE holds, not by a current. This is the DOE
# catalyst-AST protocol: the cell is cycled between a near-OCV hold and a loaded
# hold, and it is the POTENTIAL that drives Pt dissolution / oxidation.
U_TARGET_HIGH = 0.95        # V      -- near-OCV hold
U_TARGET_LOW  = 0.60        # V      -- loaded hold

# ...but these models are CURRENT-driven (operating_inputs["current_density"]), so
# the protocol has to be realised as a current profile. `calibrate_ast_currents`
# inverts the steady-state polarization curve of the REFERENCE model to find the two
# currents that hold it at U_TARGET_HIGH / U_TARGET_LOW, and that ONE profile is
# then applied to every model (see "A caveat you must read" in the header).
#
# The cycle deliberately STARTS in the low-current hold and only steps up at
# TAU_SWITCH. Do not "simplify" this to modules.signals.generate_ast_load: that wave
# sits at the high current already at t = 0, which slams the cell from its seed
# state straight to full load, starves the cathode (C_O2_ccl -> 0, then a NaN in
# eta_c's (C_O2ref/C_O2_ccl)**kappa_c) and kills the 1-D integration.
REFERENCE_MODEL = "1-D dual-scale (no aux)"

# Operating point shared by every model.
OPERATING_POINT = {
    "Tfc":       353.15,    # K
    "Pa_des":    1.8e5,     # Pa
    "Pc_des":    1.8e5,     # Pa
    "Phi_a_des": 0.85,
    "Phi_c_des": 0.85,
    "Sa":        1.2,
    "Sc":        2.0,
}

# Transport parameters. The calibrated nominals in config/initialize.py cannot
# sustain this AST in the 1-D model: the cathode starves above ~0.5 A/cm^2
# (C_O2_ccl 4.31 -> 0.46 mol/m^3 between 0.2 and 0.5 A/cm^2, then NaN at 0.6).
# Raising the cathode stoichiometry Sc does NOT help -- the bottleneck is
# through-plane diffusion, not inlet air -- so the GDL/CL transport is opened up
# instead. Same set the OAT sensitivity study used. Pass --nominal-transport to
# run with the calibrated values and see the starvation for yourself.
TRANSPORT_OVERRIDES = {
    "epsilon_gdl": 0.70,    # GDL porosity
    "epsilon_c":   0.15,    # compression ratio
    "Hgdl":        2e-4,    # m, GDL thickness
    "i0_c_ref":    0.5,     # A/m^2, cathode exchange current density
    "kappa_c":     1.0,     # cathode kinetic exponent
}

# Micro-model-only closures. With no macro transport, the micro model cannot know
# the CCL water content or the cathode O2 concentration, so both are frozen at a
# representative value. The dual-scale models compute them.
LAMBDA_CCL_FIXED = 8.5      # -        -- CCL water content (lambda)
C_O2_CCL_FIXED   = 6.0      # mol/m^3  -- cathode CL oxygen concentration
HMEM_INIT        = 1.2e-5   # m        -- reference initial membrane thickness


def _square_wave(t):
    """The AST duty wave: 0 during the low-current hold, 1 during the loaded hold.

    This is exactly the wave `modules.signals.generate_step_load` applies to the
    current, so anything built on it is phase-locked to the load by construction.
    """
    tau = np.asarray(t) % CYCLE_PERIOD
    inv_w = 6.0 / T_SWITCH
    return 0.5 * (np.tanh((tau - TAU_SWITCH) * inv_w)
                  - np.tanh((tau - (CYCLE_PERIOD - TAU_SWITCH)) * inv_w))


def make_ast_load(i_low, i_high):
    """Square-wave AST current density i(t) [A/m^2] between the calibrated holds."""
    return generate_step_load(tstart=0.0, tend=CYCLE_PERIOD,
                              i_low=i_low, i_high=i_high,
                              tau_switch=TAU_SWITCH, t_switch=T_SWITCH)


def make_ast_voltage():
    """Prescribed square-wave cell voltage U(t) [V] for the micro model.

    The micro model has no macro transport, so it cannot compute Ucell and must be
    TOLD it. It is therefore handed the AST protocol's IDEAL voltage -- an exact
    0.60 <-> 0.95 V square wave -- phase-locked to the current profile via the
    shared `_square_wave` (high current <-> low voltage, as physics requires).

    Note the asymmetry this creates, and it is the honest one: the micro model sees
    the protocol as specified, while the dual-scale models see whatever voltage the
    cell ACTUALLY reaches under the current profile -- which falls a little short of
    the holds, because a 3 s hold is not long enough to fully relax. The summary
    table reports both so you can see the gap.
    """
    def voltage(t):
        return U_TARGET_HIGH + (U_TARGET_LOW - U_TARGET_HIGH) * _square_wave(t)

    return voltage


# --------------------------------------------------------------------------- #
# Calibration: invert the reference model's polarization curve
# --------------------------------------------------------------------------- #
def steady_state_voltage(i_acm2, params, op, settle_s=40.0, label=None):
    """Steady-state Ucell [V] of the reference 1-D model at a constant i [A/cm^2]."""
    o = dict(op)
    o["current_density"] = generate_constant_load(i_acm2 * 1e4)
    p = dict(params)
    p["aux_system"] = False
    m = PEMFC(param=p, operating_inputs=o,
              variable_names=solver_variable_names, flux_names=solver_flux_names)
    y0 = init_x_for("dual-scale", o, p)
    rhs, bar = m.dxdt, None
    if label:
        rhs, bar = with_progress(m.dxdt, settle_s, label)
    try:
        sol = solve_ivp(rhs, (0.0, settle_s), y0, method="BDF", max_step=0.5, atol=1e-4)
    finally:
        if bar:
            bar.close()
    m._recovery(sol)
    return float(np.asarray(m.echem_traj["Ucell"])[-1])


def calibrate_ast_currents(params, op, cache_path=None):
    """Find the currents that hold the REFERENCE model at U_TARGET_LOW / _HIGH.

    Returns (i_low, i_high) in A/m^2. Root-finds on log(i) because the polarization
    curve is near-logarithmic: 0.95 V sits at ~3e-4 A/cm^2, where the curve is very
    steep (0.991 V at 1e-4, 0.913 V at 1e-3), while 0.60 V sits ~4000x higher up.
    A linear bracket would waste every iteration at the top end.

    Each evaluation integrates the 1-D model to steady state, so this costs ~1 min.
    The result is cached (keyed on the operating point + parameters + targets) and
    reused on later runs; --recalibrate forces a fresh solve.
    """
    key = json.dumps({
        "targets": [U_TARGET_LOW, U_TARGET_HIGH],
        "op": {k: op[k] for k in sorted(OPERATING_POINT)},
        "par": {k: params[k] for k in sorted(TRANSPORT_OVERRIDES)},
    }, sort_keys=True)

    if cache_path and Path(cache_path).exists():
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if cached.get("key") == key:
                print(f"  using cached calibration from {Path(cache_path).name} "
                      f"(--recalibrate to redo)")
                return cached["i_low"], cached["i_high"]
        except (ValueError, KeyError):
            pass                                    # corrupt cache -> just redo it

    out = {}
    for target, lo, hi in ((U_TARGET_LOW, 0.05, 3.0), (U_TARGET_HIGH, 1e-5, 0.05)):
        # Each evaluation is a full steady-state integration of the 1-D model (~3-5 s),
        # and brentq needs ~10 of them per target, so show progress for every one.
        # Memoised: brentq re-evaluates the bracket endpoints we probe below, and
        # without this that is four wasted integrations per target.
        n_eval = [0]
        seen = {}

        def U_at(i_acm2):
            key_i = round(float(i_acm2), 9)
            if key_i in seen:
                return seen[key_i]
            n_eval[0] += 1
            u = steady_state_voltage(
                i_acm2, params, op,
                label=f"U={target:.2f} V | eval {n_eval[0]:2d} | i={i_acm2:8.5f} A/cm2")
            seen[key_i] = u
            return u

        u_lo, u_hi = U_at(lo), U_at(hi)
        if not (min(u_lo, u_hi) <= target <= max(u_lo, u_hi)):
            raise RuntimeError(
                f"{target:.2f} V is not reachable by the reference model: over "
                f"i = {lo}..{hi} A/cm2 it only spans {min(u_lo, u_hi):.3f}"
                f"..{max(u_lo, u_hi):.3f} V. Adjust --u-low / --u-high.")
        root = np.exp(brentq(lambda x: U_at(np.exp(x)) - target,
                             np.log(lo), np.log(hi), xtol=1e-3, rtol=1e-4))
        out[target] = root
        print(f"  => U = {target:.2f} V  ->  i = {root:.5f} A/cm2 "
              f"({n_eval[0]} evaluations)", flush=True)

    i_low, i_high = out[U_TARGET_HIGH] * 1e4, out[U_TARGET_LOW] * 1e4   # A/m^2
    if cache_path:
        Path(cache_path).write_text(
            json.dumps({"key": key, "i_low": i_low, "i_high": i_high}, indent=2),
            encoding="utf-8")
    return i_low, i_high


# =========================================================================== #
# 2. The micro-scale model, standalone
# =========================================================================== #
class MicroCCL:
    """Micro-scale degradation model driven by a PRESCRIBED cell voltage.

    Deliberately delegates to `state_eq.dxdt_PRD` -- the very same routine the
    dual-scale models call -- so the Pt chemistry is bit-for-bit identical and the
    comparison isolates the effect of the macro-scale coupling, nothing else.

    State vector (2 * n_pt + 2):
        [C_Pt2_ccl, theta_ccl_1..n, S_N_ccl_1..n, Hmem]
    """

    def __init__(self, params, ucell, Tfc):
        self.parameters = dict(params)
        self.Ucell = ucell
        self.Tfc = float(Tfc)

        # Identical PSD grid to PEMFC / PEMFC_0D (both build it in their __init__).
        n_pt = self.parameters["n_group_pt"]
        dr = 1e-6 / n_pt
        self.r_m = (np.linspace(1e-8, 1e-6, n_pt + 1) + dr / 2)[1:]
        self.prd0 = initPRD(resolution=n_pt)
        self.n_pt = n_pt

        # Pre-computed constants (as PEMFC does in self.inst_constant).
        from model.coefficients import rho_Pt
        from scipy.integrate import trapezoid
        self.M_Pt0 = 4 / 3 * np.pi * rho_Pt * trapezoid(y=self.prd0 * self.r_m ** 3, x=self.r_m)
        self.ECSA0 = getECSA(self.prd0, self.r_m)

        # Frozen macro state (the micro model has no transport to compute these).
        self.C_H_CCL = Cproton_CCL(lambda_w=float(LAMBDA_CCL_FIXED))
        self.P_O2_c = C_O2_CCL_FIXED * R * self.Tfc      # mol/m^3 -> Pa

    def initial_state(self):
        return ([0.0]                      # C_Pt2_ccl
                + [0.0] * self.n_pt        # theta_ccl (no oxide initially)
                + self.prd0.tolist()       # S_N_ccl (particle-size distribution)
                + [self.parameters["Hmem"]])

    def dxdt(self, t, x):
        n = self.n_pt
        C_Pt2_ccl = x[0]
        theta_ccl = np.asarray(x[1:1 + n])
        prd = np.asarray(x[1 + n:1 + 2 * n])
        Hmem = x[-1]
        u = float(self.Ucell(t))

        # --- Pt kinetics (identical calls to the dual-scale models') ---
        kdis = PtDissolution(u, self.Tfc, C_Pt2_ccl, theta_ccl)
        kox = PtOxidation(u, self.Tfc, self.C_H_CCL, theta_ccl)
        kcdis = PtOxideDissolution(theta_ccl, self.C_H_CCL)
        kdet = PtDetachment(u, self.Tfc, self.r_m)

        dif = {}
        dxdt_PRD(dif=dif, Hmem=Hmem, n_mem=self.parameters["n_mem"],
                 epsilon_mc=self.parameters["epsilon_mc"], M_Pt0=self.M_Pt0,
                 prd=prd, theta_ccl=theta_ccl,
                 kdis=kdis, kox=kox, kcdis=kcdis, kdet=kdet,
                 r_m=self.r_m, prd0=self.prd0,
                 C_Pt2_ccl=C_Pt2_ccl, J_Pt2_mem=np.zeros(self.parameters["n_mem"]))

        # --- Membrane chemical thinning (same law as model.py) ---
        dHmem = -20.8 / (0.82 * 1980e3) * flourideReleaseRate(
            MT=Hmem, U=u, Tmem=self.Tfc, PO2_ca=self.P_O2_c, Hmem_init=HMEM_INIT)

        out = np.empty(2 * n + 2)
        out[0] = dif["dC_Pt2_ccl / dt"]
        out[1:1 + n] = [dif[f"dtheta_ccl_{i + 1} / dt"] for i in range(n)]
        out[1 + n:1 + 2 * n] = [dif[f"dS_N_ccl_{i + 1} / dt"] for i in range(n)]
        out[-1] = dHmem
        return out

    def ecsa_ratio(self, y):
        """ECSA retention (-) from the solver's state trajectory `y`."""
        n = self.n_pt
        return np.array([getECSA(y[1 + n:1 + 2 * n, j], self.r_m) / self.ECSA0
                         for j in range(y.shape[1])])


# =========================================================================== #
# 3. Result container + solver helper
# =========================================================================== #
@dataclass
class Result:
    """Degradation trajectory of one model, on a common set of fields."""
    label: str
    t: np.ndarray                      # s
    ecsa: np.ndarray                   # - (retention, 1.0 = pristine)
    hmem: np.ndarray                   # m (membrane thickness)
    ucell: np.ndarray = field(default=None)   # V (prescribed for the micro model)
    n_states: int = 0
    runtime_s: float = 0.0
    solver: str = ""
    issues: list = field(default_factory=list)   # physicality-guard complaints

    @property
    def cycles(self):
        return self.t / CYCLE_PERIOD

    def holds_reached(self, n_cycles):
        """(U_low, U_high) actually reached over the SETTLED cycles.

        Cycle 1 is skipped: every model starts from a seeded state and overshoots
        on the first load step (the 1-D model dips to ~0.55 V before recovering),
        which is start-up transient, not the AST.
        """
        if self.ucell is None or self.ucell.size == 0:
            return (float("nan"), float("nan"))
        settled = self.cycles >= min(1.0, max(0.0, n_cycles - 1))
        if not settled.any():
            settled = slice(None)
        u = np.asarray(self.ucell)[settled]
        return (float(np.nanmin(u)), float(np.nanmax(u)))

    @property
    def ecsa_loss_rate(self):
        """Mean ECSA loss per 1000 cycles over the simulated window (%/kcycle).

        Only meaningful near the start of life: the loss rate decelerates as the
        small particles (which dissolve first) are consumed, so extrapolating this
        linearly to a full 15 000-cycle AST OVERestimates the damage by ~10-20 %.
        """
        n_kcycles = self.cycles[-1] / 1000.0
        if n_kcycles <= 0:
            return float("nan")
        return 100.0 * (1.0 - float(self.ecsa[-1])) / n_kcycles


def _solve(dxdt, t_span, y0, max_step, atol=1e-4, label=None):
    """solve_ivp with the project's BDF -> LSODA fallback chain.

    The dual-scale models can throw a NaN inside the BDF Jacobian on recent scipy
    (>=1.15); LSODA gets through. Mirrors gui/runner.py::_solve_with_fallback.

    Pass `label` to draw a progress bar. Each method in the chain gets its own bar:
    if BDF dies half-way the bar closes as FAILED and LSODA starts a fresh one, so
    you can see the fallback happen rather than watching the bar mysteriously reset.
    """
    for method in ("BDF", "LSODA"):
        rhs, bar = dxdt, None
        if label:
            rhs, bar = with_progress(dxdt, t_span[1], f"{label} [{method}]")
        try:
            sol = solve_ivp(rhs, t_span=t_span, y0=y0, method=method,
                            max_step=max_step, atol=atol)
        except Exception:
            if bar:
                bar.close(success=False)
            continue
        ok = sol.success and sol.y.shape[1] > 0 and np.all(np.isfinite(sol.y[:, -1]))
        if bar:
            bar.close(success=ok)
        if ok:
            return sol, method
    raise RuntimeError("Both BDF and LSODA failed to integrate this model.")


def _check_physical(model, ucell, ecsa):
    """Catch results that are numerically 'successful' but physically nonsense.

    This is NOT paranoia. Pushed past its envelope, PEMFC_dyn drives the cathode
    oxygen concentration negative and emits NaN cell voltages while scipy still
    returns success=True -- so without this check the script would happily tabulate
    a meaningless degradation number. Returns a list of human-readable complaints.
    """
    issues = []

    # Any oxygen concentration below zero is unphysical, whatever the model calls
    # it (PEMFC: C_O2_ccl / C_O2_cgdl_* / ...; PEMFC_0D: its own lumped names).
    variables = getattr(model, "variables", {}) or {}
    worst = None
    for name, arr in variables.items():
        if not str(name).startswith("C_O2"):
            continue
        a = np.asarray(arr, dtype=float)
        if a.size and np.nanmin(a) < 0.0 and (worst is None or np.nanmin(a) < worst[1]):
            worst = (name, float(np.nanmin(a)))
    if worst:
        issues.append(f"{worst[0]} went NEGATIVE (min {worst[1]:.3g} mol/m3): the "
                      f"model cannot supply O2 fast enough for this load step")

    if ucell is not None:
        u = np.asarray(ucell, dtype=float)
        if u.size and np.nanmin(u) < 0.0:
            issues.append(f"cell voltage went NEGATIVE (min {np.nanmin(u):.3f} V): "
                          f"the load drives this model far beyond its polarization "
                          f"envelope, so its degradation numbers are meaningless")

    for name, arr in (("Ucell", ucell), ("ECSA", ecsa)):
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            issues.append(f"{n_nan} NaN value(s) in {name}")

    return issues


# =========================================================================== #
# 4. One runner per model variant
# =========================================================================== #
def run_micro(params, op, t_end, max_step):
    t0 = time.perf_counter()
    model = MicroCCL(params, ucell=make_ast_voltage(), Tfc=op["Tfc"])
    y0 = model.initial_state()
    sol, method = _solve(model.dxdt, (0.0, t_end), y0, max_step,
                         label="[1/4] micro (prescribed U)")
    u = make_ast_voltage()
    return Result(label="Micro (prescribed U)",
                  t=sol.t,
                  ecsa=model.ecsa_ratio(sol.y),
                  hmem=sol.y[-1],
                  ucell=np.array([u(t) for t in sol.t]),
                  n_states=len(y0),
                  runtime_s=time.perf_counter() - t0,
                  solver=method)


def run_0d(params, op, t_end, max_step):
    t0 = time.perf_counter()
    model = PEMFC_0D(parameters=dict(params), operating_inputs=dict(op))
    y0 = init_x_for("0D", op, params)
    sol, method = _solve(model.dxdt, (0.0, t_end), y0, max_step,
                         label="[2/4] 0-D dual-scale")
    model._recovery(sol)
    ecsa = np.asarray(model.echem_traj["S_N"])        # already a ratio
    ucell = np.asarray(model.echem_traj["Ucell"])
    return Result(label="0-D dual-scale",
                  t=np.asarray(model.echem_traj["t"]),
                  ecsa=ecsa,
                  hmem=np.asarray(model.variables["Hmem"]),      # NB: not "delta_mem"
                  ucell=ucell,
                  n_states=len(y0),
                  runtime_s=time.perf_counter() - t0,
                  solver=method,
                  issues=_check_physical(model, ucell, ecsa))


def _run_pemfc_1d(params, op, t_end, max_step, aux_system, label, step):
    """Shared body for the two 1-D runs -- the ONLY difference is `aux_system`."""
    t0 = time.perf_counter()
    p = dict(params)
    p["aux_system"] = aux_system
    model = PEMFC(param=p, operating_inputs=dict(op),
                  variable_names=solver_variable_names, flux_names=solver_flux_names)
    y0 = init_x_for("dual-scale", op, p)
    sol, method = _solve(model.dxdt, (0.0, t_end), y0, max_step,
                         label=f"[{step}/4] {label}")
    model._recovery(sol)
    ecsa = np.asarray(model.echem_traj["S_N"])        # already a ratio
    ucell = np.asarray(model.echem_traj["Ucell"])
    return Result(label=label,
                  t=np.asarray(model.echem_traj["t"]),
                  ecsa=ecsa,
                  hmem=np.asarray(model.variables["delta_mem"]),
                  ucell=ucell,
                  n_states=len(y0),
                  runtime_s=time.perf_counter() - t0,
                  solver=method,
                  issues=_check_physical(model, ucell, ecsa))


def run_1d(params, op, t_end, max_step):
    return _run_pemfc_1d(params, op, t_end, max_step, aux_system=False,
                         label="1-D dual-scale (no aux)", step=3)


def run_1d_aux(params, op, t_end, max_step):
    return _run_pemfc_1d(params, op, t_end, max_step, aux_system=True,
                         label="1-D + auxiliary", step=4)


def diagnose_auxiliary_coupling(params, op):
    """Re-verify, at run time, the two structural claims made in the header.

    Documentation goes stale; this does not. Both checks are a single RHS
    evaluation each, so they cost nothing. Returns a list of note strings.
    """
    notes = []

    # (a) Are PEMFC_dyn's degradation states actually integrated?
    p = dict(params)
    p["aux_system"] = True
    y0 = init_x_for("dynamic", op, p)
    dyn = PEMFC_dyn(parameters=p, operating_inputs=dict(op),
                    initial_variable_values=y0, time_interval=(0.0, 1.0))
    names = dyn.solver_variable_names
    deriv = np.asarray(dyn.dxdt(0.5, np.asarray(y0, dtype=float)), dtype=float)
    deg_idx = [i for i, n in enumerate(names)
               if n.startswith(("S_N_ccl", "theta_ccl", "C_Pt2")) or n == "delta_mem"]
    deg_max = float(np.abs(deriv[deg_idx]).max())
    if deg_max == 0.0:
        notes.append(f"PEMFC_dyn (the only class with a real coupled BoP) leaves all "
                     f"{len(deg_idx)} of its degradation\n      derivatives at exactly "
                     f"zero -- its degradation model is not wired up, so it cannot be\n"
                     f"      used for a degradation assessment. Excluded from this "
                     f"comparison.")
    else:
        notes.append(f"PEMFC_dyn now evolves its degradation states "
                     f"(max|d/dt| = {deg_max:.3e}) -- the header note is out of date "
                     f"and PEMFC_dyn could be added as a fifth curve.")

    # (b) Do PEMFC's inlet fluxes actually depend on the auxiliary states?
    #     Compare the full RHS with aux on vs off, ignoring the aux states themselves.
    rhs = {}
    for aux in (False, True):
        pp = dict(params)
        pp["aux_system"] = aux
        m = PEMFC(param=pp, operating_inputs=dict(op),
                  variable_names=solver_variable_names, flux_names=solver_flux_names)
        x0 = np.asarray(init_x_for("dual-scale", op, pp), dtype=float)
        rhs[aux] = (np.asarray(m.dxdt(0.5, x0), dtype=float), m.variable_names)

    d_off, vnames = rhs[False]
    d_on, _ = rhs[True]
    cell_idx = [i for i, n in enumerate(vnames) if n not in ("Wcp", "Wa_inj", "Wc_inj")]
    delta = float(np.abs(d_on[cell_idx] - d_off[cell_idx]).max())
    if delta == 0.0:
        notes.append("Toggling PEMFC's aux_system changes NOTHING in the cell equations "
                     "(max|delta d/dt|\n      over all non-aux states = 0). The Wcp / "
                     "Wa_inj / Wc_inj states are dangling: the\n      cathode inlet flux "
                     "is built from Sc*iload directly (inst_values.calculate_flows),\n"
                     "      i.e. an ideal, infinitely fast compressor. The two 1-D curves "
                     "must coincide.")
    else:
        notes.append(f"PEMFC's aux_system now feeds back into the cell equations "
                     f"(max|delta d/dt| = {delta:.3e}) -- the two 1-D curves may "
                     f"legitimately differ.")
    return notes


RUNNERS = {
    "micro": run_micro,
    "0d":    run_0d,
    "1d":    run_1d,
    "1d+aux": run_1d_aux,
}
STYLE = {                       # colour, linestyle, linewidth
    "micro":  ("#d70000", "--", 1.8),
    "0d":     ("#00A651", "-.", 1.8),
    "1d":     ("#000AC6", "-",  3.6),   # thick, so the aux curve drawn on top of it
    "1d+aux": ("#F28E00", (0, (1, 2.5)), 2.2),   # (which coincides exactly) stays visible
}


# =========================================================================== #
# 5. Reporting
# =========================================================================== #
def summarize(results, n_cycles):
    """Print the end-of-test degradation table and return it as CSV rows."""
    header = (f"{'Model':<26}{'states':>7}{'U hold lo/hi':>15}{'ECSA loss':>11}"
              f"{'ECSA rate':>12}{'Hmem loss':>11}{'runtime':>10}")
    print("\n" + "=" * len(header))
    print(f"Degradation after {n_cycles:g} AST cycles ({n_cycles * CYCLE_PERIOD:g} s)")
    print("=" * len(header))
    print(header)
    print(f"{'':<26}{'':>7}{'(V)':>15}{'(%)':>11}{'(%/kcycle)':>12}{'(%)':>11}{'':>10}")
    print("-" * len(header))

    rows = [("model", "n_states", "u_hold_low_V", "u_hold_high_V", "ecsa_retention",
             "ecsa_loss_pct", "ecsa_loss_pct_per_kcycle", "hmem_final_m",
             "hmem_loss_pct", "runtime_s", "solver", "issues")]
    off_protocol = []
    for r in results:
        ecsa_end = float(r.ecsa[-1])
        h_end, h_0 = float(r.hmem[-1]), float(r.hmem[0])
        h_loss = 100.0 * (h_0 - h_end) / h_0
        u_lo, u_hi = r.holds_reached(n_cycles)
        # Does this model actually see the AST it was supposed to see?
        miss = max(abs(u_lo - U_TARGET_LOW), abs(u_hi - U_TARGET_HIGH))
        flag = ""
        if r.issues:
            flag = "  <-- UNPHYSICAL"
        elif miss > 0.025:
            flag = "  <-- OFF PROTOCOL"
            off_protocol.append((r, u_lo, u_hi))
        print(f"{r.label:<26}{r.n_states:>7}{u_lo:>7.3f}/{u_hi:<7.3f}"
              f"{100 * (1 - ecsa_end):>11.3f}{r.ecsa_loss_rate:>12.2f}"
              f"{h_loss:>11.3f}{r.runtime_s:>9.1f}s{flag}")
        rows.append((r.label, r.n_states, f"{u_lo:.4f}", f"{u_hi:.4f}",
                     f"{ecsa_end:.6f}", f"{100 * (1 - ecsa_end):.4f}",
                     f"{r.ecsa_loss_rate:.4f}", f"{h_end:.6e}", f"{h_loss:.6f}",
                     f"{r.runtime_s:.2f}", r.solver, "; ".join(r.issues)))
    print("=" * len(header))
    print(f"U hold lo/hi = voltage actually reached over the settled cycles; the AST "
          f"targets\n               {U_TARGET_LOW:.2f} / {U_TARGET_HIGH:.2f} V.")
    print("ECSA rate    = mean loss per 1000 cycles; degradation decelerates, so "
          "extrapolating\n               it to a full 15 000-cycle AST overestimates "
          "the damage by ~10-20 %.")

    if off_protocol:
        print("\n" + "-" * len(header))
        print("OFF PROTOCOL -- these models do NOT see the intended 0.60/0.95 V AST.")
        print("The current profile was calibrated on the reference model "
              f"({REFERENCE_MODEL}),\nand a shared current profile cannot hold every "
              "model at the same voltage: each\nhas its own polarization curve. Their "
              "degradation is driven by the voltage in\nthe 'U hold' column, not by "
              "the protocol -- compare them with that in mind.")
        for r, u_lo, u_hi in off_protocol:
            print(f"  * {r.label}: reaches {u_lo:.3f}/{u_hi:.3f} V "
                  f"(target {U_TARGET_LOW:.2f}/{U_TARGET_HIGH:.2f} V)")
        print("-" * len(header))

    flagged = [r for r in results if r.issues]
    if flagged:
        print("\n" + "!" * 79)
        print("UNPHYSICAL RESULTS -- do not trust the rows above for these models:")
        for r in flagged:
            for issue in r.issues:
                print(f"  * {r.label}: {issue}")
        print("  The solver still reported success. Lower --i-high (the auxiliary "
              "model is\n  usually the first to break down) or lengthen the current "
              "edge (T_SWITCH).")
        print("!" * 79)
    return rows


def plot(results, keys, n_cycles, outdir, i_low, i_high):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for key, r in zip(keys, results):
        color, ls, lw = STYLE[key]
        # Plot the LOSS from pristine, not the absolute value: over a short run the
        # absolute curves differ in the 4th decimal and matplotlib hides that behind
        # an axis offset like "+9.999e1". Losses start at 0 and read directly.
        axes[0].plot(r.cycles, 100 * (1.0 - r.ecsa), color=color, ls=ls, lw=lw,
                     label=r.label)
        axes[1].plot(r.cycles, 1e9 * (r.hmem[0] - r.hmem), color=color, ls=ls, lw=lw,
                     label=r.label)
        # Last cycle only -- shows HOW each model gets the voltage that drives
        # the degradation kinetics above.
        mask = r.cycles >= (n_cycles - 1)
        if r.ucell is not None and mask.any():
            axes[2].plot(r.cycles[mask] - (n_cycles - 1), r.ucell[mask],
                         color=color, ls=ls, lw=lw, label=r.label)

    axes[0].set_xlabel("AST cycles")
    axes[0].set_ylabel("ECSA loss (%)")
    axes[0].set_title("Catalyst degradation")

    axes[1].set_xlabel("AST cycles")
    axes[1].set_ylabel("Membrane thinning (nm)")
    axes[1].set_title("Membrane degradation")

    # The AST is defined by these two voltages -- draw them, so it is immediately
    # obvious which models actually sit on the protocol and which drift off it.
    for u_target in (U_TARGET_LOW, U_TARGET_HIGH):
        axes[2].axhline(u_target, color="#888", ls="--", lw=1.0, zorder=0)
        axes[2].text(0.01, u_target + 0.008, f"{u_target:.2f} V target",
                     color="#666", fontsize=7)
    axes[2].set_xlabel("Time within the last cycle (-)")
    axes[2].set_ylabel("Cell voltage (V)")
    axes[2].set_title("Driving voltage (last cycle)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"AST degradation assessment across the model hierarchy -- "
                 f"{U_TARGET_LOW:g}/{U_TARGET_HIGH:g} V protocol, {n_cycles:g} cycles "
                 f"@ {CYCLE_PERIOD:g} s\n(shared load "
                 f"{i_low / 1e4:.4f}$\\rightarrow${i_high / 1e4:.3f} A/cm$^2$, "
                 f"calibrated on {REFERENCE_MODEL})", fontsize=10)
    fig.tight_layout()

    png = outdir / "AST_comparison.png"
    fig.savefig(png, dpi=150)
    print(f"\nFigure written to  {png}")
    return png


def main():
    global U_TARGET_LOW, U_TARGET_HIGH

    ap = argparse.ArgumentParser(
        description="Compare AST degradation predicted by the micro, 0-D dual-scale "
                    "and 1-D dual-scale (with/without auxiliary) models.")
    ap.add_argument("--cycles", type=float, default=N_CYCLES,
                    help="number of AST cycles to simulate; overrides the N_CYCLES "
                         "constant at the top of this file (default: %(default)s)")
    ap.add_argument("--models", nargs="+", default=list(RUNNERS),
                    choices=list(RUNNERS), metavar="M",
                    help=f"which models to run (default: all of {list(RUNNERS)})")
    ap.add_argument("--max-step", type=float, default=0.1,
                    help="solver max_step in s (default: 0.1; must stay well below "
                         "the cycle period to resolve the square wave)")
    ap.add_argument("--u-low", type=float, default=U_TARGET_LOW, metavar="V",
                    help="loaded-hold voltage (default: %(default)s V)")
    ap.add_argument("--u-high", type=float, default=U_TARGET_HIGH, metavar="V",
                    help="near-OCV hold voltage (default: %(default)s V)")
    ap.add_argument("--recalibrate", action="store_true",
                    help="force a fresh polarization inversion instead of reusing "
                         "the cached currents")
    ap.add_argument("--nominal-transport", action="store_true",
                    help="use config/initialize.py's calibrated transport parameters "
                         "instead of the opened-up set. The 1-D models will then "
                         "starve above ~0.5 A/cm2, so the 0.60 V hold is unreachable.")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent,
                    help="where to write the figure and the CSV summary")
    args = ap.parse_args()

    U_TARGET_LOW, U_TARGET_HIGH = args.u_low, args.u_high
    t_end = args.cycles * CYCLE_PERIOD
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Every model shares one operating point, one parameter set and one load.
    params = dict(parameters)
    if not args.nominal_transport:
        params.update(TRANSPORT_OVERRIDES)
    op = dict(operating_inputs)
    op.update(OPERATING_POINT)

    print(f"AST: {args.cycles:g} cycles x {CYCLE_PERIOD:g} s = {t_end:g} s | "
          f"holds {U_TARGET_LOW:g} / {U_TARGET_HIGH:g} V | T = {op['Tfc']:g} K | "
          f"transport: {'NOMINAL' if args.nominal_transport else 'opened-up'}")

    # The AST is voltage-defined, but the models are current-driven: invert the
    # reference model's polarization curve to realise the protocol as a current.
    print(f"Calibrating the load profile on the reference model ({REFERENCE_MODEL}):")
    cache_path = args.outdir / "AST_calibration.json"
    if args.recalibrate and cache_path.exists():
        cache_path.unlink()
    i_low, i_high = calibrate_ast_currents(params, op, cache_path=cache_path)
    op["current_density"] = make_ast_load(i_low, i_high)
    print(f"  -> shared AST load: {i_low / 1e4:.5f} -> {i_high / 1e4:.5f} A/cm2, "
          f"applied to ALL models.")

    results, keys = [], []
    for key in args.models:
        try:
            results.append(RUNNERS[key](params, op, t_end, args.max_step))
            keys.append(key)
        except Exception as exc:                       # keep the other models going
            print(f"      !! {key} failed: {type(exc).__name__}: {exc}")

    if not results:
        print("No model completed -- nothing to compare.")
        return 1

    rows = summarize(results, args.cycles)

    print("\nStructural notes (re-checked against the model code on this run):")
    for note in diagnose_auxiliary_coupling(params, op):
        print(f"  * {note}")

    plot(results, keys, args.cycles, args.outdir, i_low, i_high)
    csv = args.outdir / "AST_comparison.csv"
    csv.write_text("\n".join(",".join(str(c) for c in row) for row in rows), encoding="utf-8")
    print(f"Summary written to {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
