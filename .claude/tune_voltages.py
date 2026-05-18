"""Tune the undetermined physical parameters to produce
   V_low  = 0.95 V at i = i_low (low plateau)
   V_high = 0.60 V at i = i_high (high plateau)

Each "trial" runs two short steady-state simulations (constant i = i_low
and constant i = i_high) and reads back Ucell at the end of each.
"""
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.initialize import operating_inputs as DEFAULT_OP, parameters as DEFAULT_PARAMS, init_x
from config.settings import solver_variable_names, solver_flux_names
from model.dualscale import PEMFC
from model.kinetic_eq import Ucell


# ------------------------------------------------------------------ operating
PHI_C_DES = 0.85
SA, SC    = 1.2, 2.0
PA_DES    = 1.8e5
PC_DES    = 1.8e5
TFC       = 353.15
I_LOW, I_HIGH = 20.0, 12000.0
T_HOLD     = 3.0          # seconds at each constant current
MAX_STEP_S = 1e-2


def steady_voltage(i_const, overrides):
    """Run the model at a constant current `i_const` for T_HOLD seconds and
    return the cell voltage at the final step (or None on failure)."""
    params = deepcopy(DEFAULT_PARAMS)
    params.update(overrides)
    op = deepcopy(DEFAULT_OP)
    op["current_density"] = (lambda v: lambda t: v)(i_const)
    op["Phi_c_des"] = PHI_C_DES
    op["Pa_des"]    = PA_DES
    op["Pc_des"]    = PC_DES
    op["Tfc"]       = TFC
    op["Sa"]        = SA
    op["Sc"]        = SC

    model = PEMFC(param=params, operating_inputs=op,
                  variable_names=solver_variable_names,
                  flux_names=solver_flux_names)
    y0 = np.asarray(init_x(op, params), dtype=float)
    try:
        sol = solve_ivp(model.dxdt, (0.0, T_HOLD), y0,
                        method="BDF", max_step=MAX_STEP_S,
                        rtol=1e-3, atol=1e-6)
    except Exception as exc:
        return None, str(exc)[:60]
    if not sol.success:
        return None, sol.message[:60]
    states_end = {k: sol.y[i, -1] for i, k in enumerate(model.variable_names)}
    return float(Ucell(sol.t[-1], states_end, op, model.parameters)), "ok"


def report(name, overrides):
    print(f"\n--- {name} ---  {overrides}")
    t0 = time.perf_counter()
    Vl, msg_l = steady_voltage(I_LOW,  overrides)
    Vh, msg_h = steady_voltage(I_HIGH, overrides)
    print(f"  V(i={I_LOW:6.0f}) = {Vl}  [{msg_l}]")
    print(f"  V(i={I_HIGH:6.0f}) = {Vh}  [{msg_h}]")
    print(f"  wall = {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    BASE = {
        "epsilon_gdl": 0.7,
        "epsilon_cl":  0.5,
        "i0_c_ref":   10.0,
        "a_slim":      0.2,
        "b_slim":      0.2,
        "Hgdl":      1e-4,
        "Re":         1e-7,
    }
    # Fine tune around OCV=0.97-0.98 with kappa_c=0.05.
    for ocv in [0.973, 0.975, 0.977]:
        report(f"OCV={ocv}  kappa_c=0.05",
               {**BASE, "OCV": ocv, "kappa_c": 0.05})
