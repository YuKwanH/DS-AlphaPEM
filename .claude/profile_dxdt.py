"""Profile PEMFC.dxdt and a short solve_ivp run, print the project-level
functions sorted by cumulative time -- this tells us where wall time goes."""
import cProfile
import io
import os
import pstats
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.initialize import operating_inputs, parameters, init_x
from config.settings import solver_variable_names, solver_flux_names
from model.dualscale import PEMFC
from modules.signals import generate_step_load
from scipy.integrate import solve_ivp
import numpy as np

operating_inputs["current_density"] = generate_step_load(0.0, 6.0, 0.002e4, 1.0e4, 1.0, 3.0)
operating_inputs["Phi_c_des"] = 0.2
model = PEMFC(param=parameters, operating_inputs=operating_inputs,
              variable_names=solver_variable_names, flux_names=solver_flux_names)
y0 = np.asarray(init_x(operating_inputs, parameters), dtype=float)


def project_only(stats_text):
    """Keep only lines that mention a project source file."""
    SEP = os.sep
    tokens = (f"{SEP}model{SEP}", f"{SEP}config{SEP}", f"{SEP}modules{SEP}",
              "_ivp", "function calls", "Ordered by", "tottime", "ncalls")
    return "\n".join(line for line in stats_text.splitlines()
                     if any(tok in line for tok in tokens))


# ---------------------------------------------------------------------------
# Part 1: 1000 standalone dxdt(0, y0) calls -- pure model cost
# ---------------------------------------------------------------------------
N = 1000
print(f"=== Part 1: {N} standalone dxdt(0, y0) calls ===\n")
t0 = time.perf_counter()
prof = cProfile.Profile()
prof.enable()
for _ in range(N):
    model.dxdt(0.0, y0)
prof.disable()
wall = time.perf_counter() - t0
print(f"  wall = {wall:.2f} s  ->  per-call = {wall / N * 1000:.2f} ms\n")
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(30)
print(project_only(buf.getvalue()))


# ---------------------------------------------------------------------------
# Part 2: short solve_ivp, BDF + max_step=0.1 (the GUI's settings)
# ---------------------------------------------------------------------------
print("\n=== Part 2: solve_ivp((0, 0.2), method='BDF', max_step=0.1) ===\n")
t0 = time.perf_counter()
prof = cProfile.Profile()
prof.enable()
sol = solve_ivp(model.dxdt, (0, 0.2), y0, method="BDF", max_step=0.1)
prof.disable()
wall = time.perf_counter() - t0
print(f"  wall = {wall:.2f} s  ->  {len(sol.t)} BDF steps,  "
      f"~{wall / max(len(sol.t), 1) * 1000:.1f} ms / step\n")
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
print(project_only(buf.getvalue()))


# ---------------------------------------------------------------------------
# Part 3: per-module breakdown of dxdt cost
# ---------------------------------------------------------------------------
print("\n=== Part 3: per-sub-step breakdown of one dxdt call (mean over 200) ===\n")
from model.inst_values import dif_eq_int_values, calculate_flows
from model.state_eq import (
    dxdt_AGC, dxdt_CGC, dxdt_AGDL, dxdt_CGDL, dxdt_ACL, dxdt_CCL,
    dxdt_MEM, dxdt_CP, dxdt_Manifold, dxdt_TH, dxdt_U, dxdt_N2, dxdt_PRD,
)

states = {n: y0[i] for n, i in model._idx.items()}
inst   = dif_eq_int_values(0.0, states, operating_inputs, model.parameters)
flows  = calculate_flows(0.0, states, operating_inputs, model.parameters, **inst)
allv   = {**model.parameters, **inst, **flows, **operating_inputs}

def bench(label, fn, *args, **kwargs):
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<28s} {dt_ms:>7.3f} ms")

# Recompute inst_states / flows each iteration so it counts that cost.
def make_inst():
    return dif_eq_int_values(0.0, states, operating_inputs, model.parameters)
def make_flows():
    return calculate_flows(0.0, states, operating_inputs, model.parameters, **inst)

bench("dif_eq_int_values",      make_inst)
bench("calculate_flows",        make_flows)
dif = dict.fromkeys(model._dif_keys, 0.0)
bench("dxdt_AGC",               dxdt_AGC, dif, **allv)
bench("dxdt_CGC",               dxdt_CGC, dif, **allv)
bench("dxdt_AGDL",              dxdt_AGDL, dif, states, **allv)
bench("dxdt_CGDL",              dxdt_CGDL, dif, states, **allv)
bench("dxdt_ACL",               dxdt_ACL, dif, states, **allv)
bench("dxdt_CCL",               dxdt_CCL, dif, states, **allv)
bench("dxdt_MEM",               dxdt_MEM, dif, states, **allv)
bench("dxdt_CP",                dxdt_CP, dif, **states, **allv)
bench("dxdt_Manifold",          dxdt_Manifold, dif, **allv)
bench("dxdt_TH",                dxdt_TH, dif, **states, **allv)
bench("dxdt_U",                 dxdt_U, dif, **states, **allv)
bench("dxdt_N2",                dxdt_N2, dif, **allv)
bench("dxdt_PRD",               dxdt_PRD, dif=dif, **states, **allv)
