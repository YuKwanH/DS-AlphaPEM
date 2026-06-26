"""Comprehensive simulation-functionality verification matrix.

Exercises every meaningful combination of (model variant × aux mode × test
profile) plus the auto-routing and 0D-benchmark companion paths. Captures
runtime, solver step count, success flag, and a sanity check on the
primary output. Prints a final per-combo PASS/FAIL table.

Run from the project root:
    python .claude/verify_matrix.py
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Always line-buffer stdout so the background monitor sees progress as it happens.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # silence solve_ivp DeprecationWarnings

from copy import deepcopy
from config.initialize import parameters as _PARAMS_DEFAULT
from config.initialize import operating_inputs as _OP_DEFAULT
from gui import profiles
from gui.runner import run as run_simulation, run_0d_companion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fresh_inputs(mesh_override=None):
    p = deepcopy(_PARAMS_DEFAULT)
    if mesh_override:
        p.update(mesh_override)
    o = deepcopy({k: v for k, v in _OP_DEFAULT.items() if k != "current_density"})
    return p, o


def _ucell_stats(model):
    u = np.asarray(getattr(model, "echem_traj", {}).get("Ucell", []))
    if u.size == 0:
        return (None, None, None)
    return (float(u.min()), float(u.max()), int(np.isfinite(u).sum()))


def _record(label, success, runtime, n_steps, n_states, message, **extra):
    rec = dict(label=label, success=success, runtime=runtime,
               n_steps=n_steps, n_states=n_states, message=message,
               **extra)
    RESULTS.append(rec)
    tag = "PASS" if success else "FAIL"
    print(f"  [{tag}] {label:<52s} t={runtime:>7.2f}s  n_steps={n_steps:<5} "
          f"n_states={n_states or '–':<5}  {message[:60] if message else ''}")
    return rec


def _run_transient(label, model_variant, aux_system, profile_func,
                   t_span, mesh_override=None, max_step=0.1):
    p, o = _fresh_inputs(mesh_override)
    t0 = time.perf_counter()
    try:
        model, sol, status = run_simulation(
            params=p, op_inputs=o,
            model_variant=model_variant, profile_func=profile_func,
            t_span=t_span, max_step=max_step, method="BDF",
            polar_sweep=None, aux_system=aux_system,
        )
        wall = time.perf_counter() - t0
        umin, umax, ufin = _ucell_stats(model)
        extra = dict(resolved_class=type(model).__name__,
                     variant_label=status["model_variant"],
                     ucell_min=umin, ucell_max=umax,
                     ucell_finite=ufin)
        return _record(label, status["success"], wall, status["n_steps"],
                       status["n_states"], status["message"], **extra)
    except Exception as exc:
        wall = time.perf_counter() - t0
        return _record(label, False, wall, 0, 0,
                       f"EXCEPTION {type(exc).__name__}: {exc}")


def _run_polar(label, polar_sweep):
    p, o = _fresh_inputs()
    t0 = time.perf_counter()
    try:
        model, polar, status = run_simulation(
            params=p, op_inputs=o,
            model_variant="Static", profile_func=None,
            t_span=(0.0, 0.0), max_step=0.1, method="BDF",
            polar_sweep=polar_sweep, aux_system=False,
        )
        wall = time.perf_counter() - t0
        i_arr = np.asarray(polar.get("i_A_m2", []))
        u_arr = np.asarray(polar.get("Ucell_V", []))
        extra = dict(n_points=status.get("n_points"),
                     i_range=(float(i_arr.min()), float(i_arr.max())) if i_arr.size else None,
                     ucell_range=(float(u_arr.min()), float(u_arr.max())) if u_arr.size else None)
        return _record(label, status["success"], wall, status.get("n_points", 0),
                       None, status["message"], **extra)
    except Exception as exc:
        wall = time.perf_counter() - t0
        return _record(label, False, wall, 0, None,
                       f"EXCEPTION {type(exc).__name__}: {exc}")


def _run_0d_companion(label, profile_func, t_span, polar_sweep=None):
    p, o = _fresh_inputs()
    t0 = time.perf_counter()
    try:
        info = run_0d_companion(p, o, profile_func, t_span,
                                polar_sweep=polar_sweep)
        wall = time.perf_counter() - t0
        s = info["status"]
        is_polar = polar_sweep is not None
        if is_polar:
            polar = info.get("polar") or {}
            u_arr = np.asarray(polar.get("Ucell_V", []))
            extra = dict(n_points=s.get("n_points"),
                         ucell_range=(float(u_arr.min()), float(u_arr.max())) if u_arr.size else None)
        else:
            u = np.asarray(info["echem_traj"].get("Ucell", []))
            extra = dict(ucell_min=float(u.min()) if u.size else None,
                         ucell_max=float(u.max()) if u.size else None)
        return _record(label, s["success"], wall,
                       s.get("n_steps", s.get("n_points", 0)),
                       None, s["message"], **extra)
    except Exception as exc:
        wall = time.perf_counter() - t0
        return _record(label, False, wall, 0, None,
                       f"EXCEPTION {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Build profile-func builders
# ---------------------------------------------------------------------------
def pf_constant(): return profiles.constant(4000.0)
def pf_step():     return profiles.step(0.0, 6.0, 20.0, 12000.0, 1.0, 3.0)
def pf_eis():      return profiles.eis(10000.0, 0.05, 1.0)
def pf_ast(Aact):  return profiles.ast_cycling(6.0, 1.0, 25.8, 4.0, Aact)

Aact = _PARAMS_DEFAULT["Aact"]

# Coarser mesh for the long AST runs so wall-time stays manageable.
AST_MESH = dict(n_gdl=5, n_mem=5, n_group_pt=5)

RESULTS = []


# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------
print("=" * 90)
print("BLOCK A — PEMFC (Dual-scale, aux=off): short transient profiles")
print("=" * 90)

_run_transient("Dual-scale + Constant (t=30 s)",
               "Dual-scale", False, pf_constant(), (0.0, 30.0))

_run_transient("Dual-scale + Step (t=30 s)",
               "Dual-scale", False, pf_step(), (0.0, 30.0))

_run_transient("Dual-scale + EIS @1 Hz (t=10 s)",
               "Dual-scale", False, pf_eis(), (0.0, 10.0))

print()
print("=" * 90)
print("BLOCK B — PEMFC_dyn (Dynamic, aux=on): short transient profiles")
print("=" * 90)

_run_transient("Dynamic + Constant (t=30 s)",
               "Dynamic", True, pf_constant(), (0.0, 30.0))

_run_transient("Dynamic + Step (t=30 s)",
               "Dynamic", True, pf_step(), (0.0, 30.0))

print()
print("=" * 90)
print("BLOCK C — Auto-routing of Variant × Aux combinations")
print("=" * 90)

_run_transient("Auto-promote: Dual-scale + aux=on -> PEMFC_dyn",
               "Dual-scale", True, pf_step(), (0.0, 15.0))

_run_transient("Auto-demote:  Dynamic + aux=off -> PEMFC",
               "Dynamic", False, pf_step(), (0.0, 15.0))

print()
print("=" * 90)
print("BLOCK D — Static polarization sweep")
print("=" * 90)

_run_polar("Static polar (n=20, i_max=1.65 A/cm^2)",
           polar_sweep=dict(i_max_A_cm2=1.65, n_points=20))

print()
print("=" * 90)
print("BLOCK E — 0D benchmark companion")
print("=" * 90)

_run_0d_companion("0D companion + Step (t=15 s)",
                  pf_step(), (0.0, 15.0))
_run_0d_companion("0D companion polar sweep (n=15)",
                  None, (0.0, 0.0),
                  polar_sweep=dict(i_max_A_cm2=1.65, n_points=15))

print()
print("=" * 90)
print("BLOCK F — Long-term AST cycling (1-hour sim time, coarse mesh)")
print("=" * 90)

# AST 1 hour: 600 cycles of the 6-s waveform. Coarser mesh + larger max_step
# keep the solver tractable. We tolerate a long wall time.
_run_transient("Dual-scale + AST 1 h (mesh 5/5/5)",
               "Dual-scale", False, pf_ast(Aact), (0.0, 3600.0),
               mesh_override=AST_MESH, max_step=0.5)

_run_transient("Dynamic + AST 1 h (mesh 5/5/5)",
               "Dynamic", True, pf_ast(Aact), (0.0, 3600.0),
               mesh_override=AST_MESH, max_step=0.5)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 90)
n_pass = sum(1 for r in RESULTS if r["success"])
n_fail = len(RESULTS) - n_pass
total_t = sum(r["runtime"] for r in RESULTS)
print(f"SUMMARY:  {n_pass}/{len(RESULTS)} passed  ·  total wall time {total_t:.1f} s")
print("=" * 90)

# Persist machine-readable record for the report write-up.
with open(os.path.join(os.path.dirname(__file__), "verify_matrix_results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

# Highlight any failures
fails = [r for r in RESULTS if not r["success"]]
if fails:
    print()
    print("FAILURES:")
    for r in fails:
        print(f"  • {r['label']}: {r['message']}")
else:
    print()
    print("All combinations completed without error.")
