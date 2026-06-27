"""PEMFC simulator - raw-code single-simulation entry point.

Edit the USER SETTINGS block below, then run::

    python main.py

For the interactive web GUI use ``run_gui.bat`` (or ``streamlit run gui/app.py``).
For Bayesian parameter calibration over experimental data use ``calibration.py``.
"""
from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from config.initialize import parameters, operating_inputs, init_x_for
from config.settings   import solver_variable_names, solver_flux_names
from model.model       import PEMFC, PEMFC_dyn, PEMFC_0D


# =============================================================================
# USER SETTINGS  --  edit these, save, then re-run `python main.py`
# =============================================================================

# Model file to integrate: 'dual-scale' (1-D, BoP toggle), 'dynamic' (1-D + full BoP), or '0D' (lumped).
MODEL_VARIANT = "dual-scale"

# Balance-of-plant equations: True simulates compressor/manifolds, False freezes them at the design point.
AUX_SYSTEM = False

# Cell temperature in Kelvin.
TFC = 333.15

# Anode supply pressure in Pa.
PA_DES = 1.4e5

# Cathode supply pressure in Pa.
PC_DES = 1.4e5

# Anode relative humidity at the inlet (0 = dry, 1 = saturated).
PHI_A_DES = 0.0

# Cathode relative humidity at the inlet (0 = dry, 1 = saturated).
PHI_C_DES = 0.5

# Anode stoichiometry (excess H2 supply ratio relative to the load demand).
SA = 1.2

# Cathode stoichiometry (excess air supply ratio relative to the load demand).
SC = 2.0

# Constant load current in Amperes applied throughout the simulation.
I_LOAD = 30.0

# Integration time window in seconds, given as (t_start, t_end).
T_SPAN = (0.0, 20.0)

# Maximum time step the solver may take, in seconds.
MAX_STEP = 0.1

# ODE solver name passed to scipy.integrate.solve_ivp ('BDF', 'Radau', 'LSODA', or 'RK45').
METHOD = "BDF"

# When True, open a matplotlib window with the load-current and cell-voltage trajectories.
SHOW_PLOTS = True


# =============================================================================
# Run --  no editing needed below this line for a normal simulation
# =============================================================================

def _build_op_and_params():
    op = deepcopy(operating_inputs)
    op["Tfc"]       = TFC
    op["Pa_des"]    = PA_DES
    op["Pc_des"]    = PC_DES
    op["Phi_a_des"] = PHI_A_DES
    op["Phi_c_des"] = PHI_C_DES
    op["Sa"]        = SA
    op["Sc"]        = SC
    op["current_density"] = lambda t: I_LOAD / parameters["Aact"]

    p = deepcopy(parameters)
    p["aux_system"] = AUX_SYSTEM
    return op, p


def _construct_model(op, p, x_init):
    variant = MODEL_VARIANT.lower().replace("_", "").replace("-", "")
    if variant in ("dualscale", "pemfc"):
        return PEMFC(param=p, operating_inputs=op,
                     variable_names=solver_variable_names,
                     flux_names=solver_flux_names)
    if variant in ("dynamic", "dyn", "pemfcdyn"):
        return PEMFC_dyn(parameters=p, operating_inputs=op,
                         initial_variable_values=x_init,
                         time_interval=T_SPAN)
    if variant in ("0d", "pemfc0d", "lumped"):
        return PEMFC_0D(parameters=p, operating_inputs=op)
    raise ValueError(f"Unknown MODEL_VARIANT: {MODEL_VARIANT!r}")


def _extract_ucell_and_ifc(model, sol):
    """Pull (t, i_fc, Ucell) trajectories from whichever storage the model uses."""
    t = np.asarray(model.variables.get("t", sol.t))
    src_chain = (getattr(model, "echem_traj", None),
                 getattr(model, "ec_kinetics", None),
                 model.variables)
    def _get(key):
        for src in src_chain:
            if src and key in src and hasattr(src[key], "__len__") and len(src[key]):
                return np.asarray(src[key], dtype=float)
        return None
    return t, _get("i_fc"), _get("Ucell")


def main():
    op, p = _build_op_and_params()
    x_init = init_x_for(MODEL_VARIANT, op, p)
    model  = _construct_model(op, p, x_init)

    print("=" * 64)
    print(" PEMFC simulator - raw-code single run")
    print("=" * 64)
    print(f"   Model        : {MODEL_VARIANT}   (aux_system={AUX_SYSTEM})")
    print(f"   Operating    : Tfc={TFC} K, Pa/Pc={PA_DES/1e5:.2f}/{PC_DES/1e5:.2f} bar, "
          f"RHa/RHc={PHI_A_DES}/{PHI_C_DES}, Sa/Sc={SA}/{SC}")
    print(f"   Load         : I = {I_LOAD} A  (i = {I_LOAD / p['Aact']:.0f} A/m2)")
    print(f"   Time span    : {T_SPAN[0]:.1f} -> {T_SPAN[1]:.1f} s   (max_step={MAX_STEP}, method={METHOD})")
    print(f"   State size   : {len(x_init)} ODEs")
    print("=" * 64)

    t0  = time.perf_counter()
    sol = solve_ivp(model.dxdt, T_SPAN, x_init, method=METHOD, max_step=MAX_STEP)
    runtime = time.perf_counter() - t0

    status = "OK" if sol.success else "FAILED"
    print(f"\n   {status}  -  runtime={runtime:.2f} s, n_steps={len(sol.t)}")
    if not sol.success:
        print(f"   Solver message: {sol.message}")
        return

    try:
        model._recovery(sol)
    except Exception as exc:
        print(f"   (model._recovery raised {type(exc).__name__}: {exc} -- continuing)")

    t, i_fc, Ucell = _extract_ucell_and_ifc(model, sol)
    if Ucell is not None and len(Ucell):
        print(f"   Ucell at t_end : {Ucell[-1]:.4f} V")
    if i_fc is not None and len(i_fc):
        print(f"   i_fc at t_end  : {i_fc[-1]:.1f} A/m2")

    if SHOW_PLOTS and Ucell is not None and len(Ucell):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        if i_fc is not None and len(i_fc):
            ax[0].plot(t[:len(i_fc)], i_fc, linewidth=1.4, color="tab:blue")
        ax[0].set_xlabel("Time (s)");  ax[0].set_ylabel("Current density (A/m$^2$)")
        ax[0].set_title("Load current");  ax[0].grid(True, alpha=0.3)
        ax[1].plot(t[:len(Ucell)], Ucell, linewidth=1.5, color="tab:orange")
        ax[1].set_xlabel("Time (s)");  ax[1].set_ylabel("Cell voltage (V)")
        ax[1].set_title("Cell voltage");  ax[1].grid(True, alpha=0.3)
        fig.suptitle(f"PEMFC simulator -- {MODEL_VARIANT}  I={I_LOAD} A  "
                     f"Tfc={TFC} K  Pc={PC_DES/1e5:.2f} bar")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
