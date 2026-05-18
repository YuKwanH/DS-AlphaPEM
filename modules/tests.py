"""Standardised test-protocol runners for the PEMFC dual-scale model.

Each public function packages one fuel-cell test protocol behind a single
call. The pipeline is identical for every protocol so results are directly
comparable:

    1. build the ``current_density`` callable from ``modules.signals``
    2. construct a fresh ``PEMFC`` with project default config (deep-copied
       so the global dicts stay clean)
    3. integrate with ``scipy.integrate.solve_ivp``
    4. recover trajectories into ``model.variables`` / ``model.fluxes`` /
       ``model.echem_traj``

All functions return ``(model, sol)``.
"""

from copy import deepcopy

from scipy.integrate import solve_ivp

from config.initialize import init_x, parameters as _DEFAULT_PARAMS
from config.initialize import operating_inputs as _DEFAULT_OP_INPUTS
from config.settings import solver_variable_names, solver_flux_names
from model.dualscale import PEMFC
from modules.signals import (
    generate_ast_load,
    generate_constant_load,
    generate_eis_load,
    generate_polarization_load,
    generate_step_load,
)


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def _run(profile, t_span, parameters=None, operating_inputs=None,
         method="BDF", max_step=0.1):
    """Build a PEMFC for the given current-density ``profile`` and integrate
    over ``t_span``. Returns ``(model, sol)``."""
    params    = deepcopy(parameters       if parameters       is not None else _DEFAULT_PARAMS)
    op_inputs = deepcopy({k: v for k, v in (operating_inputs or _DEFAULT_OP_INPUTS).items()
                          if k != "current_density"})
    op_inputs["current_density"] = profile

    model = PEMFC(param=params, operating_inputs=op_inputs,
                  variable_names=solver_variable_names,
                  flux_names=solver_flux_names)
    y0 = init_x(op_inputs, params)
    sol = solve_ivp(model.dxdt, t_span, y0, method=method, max_step=max_step)
    model._recovery(sol)
    return model, sol


# ---------------------------------------------------------------------------
# Test protocols
# ---------------------------------------------------------------------------

def constant_load_test(i_density=4000.0, t_span=(0, 20), **kwargs):
    """Hold a steady current density ``i_density`` [A/m^2] for ``t_span`` seconds."""
    return _run(generate_constant_load(i_density), t_span, **kwargs)


def step_load_test(tstart=0.0, tend=6.0, i_low=20.0, i_high=12000.0,
                   tau_switch=1.5, t_switch=0.5, t_span=(0, 20), **kwargs):
    """Periodic symmetric square load between ``i_low`` and ``i_high`` [A/m^2].

    Defaults give a 6 s period with ~50/50 LOW/HIGH duty cycle and 0.5 s
    smooth transitions (rising edge at ``tau_switch``, falling edge at
    ``tend - tau_switch``).
    """
    profile = generate_step_load(tstart, tend, i_low, i_high, tau_switch, t_switch)
    return _run(profile, t_span, **kwargs)


def polarization_test(i_max=16500.0, n_steps=30, t_per_step=60.0, **kwargs):
    """Staircase polarization sweep from ``i_max/n_steps`` up to ``i_max`` [A/m^2].

    Each plateau is held for ``t_per_step`` seconds; total run length is
    ``n_steps * t_per_step``.
    """
    profile = generate_polarization_load(i_max, n_steps, t_per_step)
    return _run(profile, (0, n_steps * t_per_step), **kwargs)


def eis_test(i_dc=10000.0, ratio=0.05, frequency=1.0, n_periods=10, **kwargs):
    """Sinusoidal EIS perturbation at ``frequency`` Hz around DC bias ``i_dc`` [A/m^2].

    The default run length covers ``n_periods`` periods of the AC signal.
    """
    profile = generate_eis_load(i_dc, ratio, frequency)
    return _run(profile, (0, n_periods / float(frequency)), **kwargs)


def ast_cycling_test(period=60.0, I_low=1.0, I_high=25.8, smoothing=4.0,
                     n_cycles=10, parameters=None, operating_inputs=None,
                     **kwargs):
    """Smoothed square-wave Accelerated Stress Test (AST) cycling load.

    Total currents ``I_low`` and ``I_high`` are in Amperes and divided by
    the active area ``Aact`` internally. ``parameters`` supplies that area;
    defaults to ``config/initialize.py`` if omitted. The simulation runs
    for ``n_cycles`` periods.
    """
    params = parameters if parameters is not None else _DEFAULT_PARAMS
    profile = generate_ast_load(period, I_low, I_high, smoothing, params["Aact"])
    return _run(profile, (0, n_cycles * period),
                parameters=parameters, operating_inputs=operating_inputs,
                **kwargs)
