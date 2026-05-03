"""Current-density profile adapters for the GUI.

Thin re-exports of the canonical builders in :mod:`modules.signals` so the
GUI options panel and the rest of the codebase share one source of truth
for every test-profile waveform. All builders take and return values in
A/m².
"""

import numpy as np

from modules.signals import (
    generate_ast_load,
    generate_constant_load,
    generate_eis_load,
    generate_polarization_load,
    generate_step_load,
)


def constant(i_density_A_m2):
    return generate_constant_load(i_density_A_m2)


def step(tstart, tend, i_low_A_m2, i_high_A_m2, tau_switch, t_switch):
    """Periodic tanh-smoothed square load.

    Mirrors the canonical setup in ``simulation/control/square load.ipynb``.
    """
    return generate_step_load(
        tstart, tend, i_low_A_m2, i_high_A_m2, tau_switch, t_switch,
    )


def polarization_ramp(i_max_A_m2, n_steps, t_per_step):
    return generate_polarization_load(i_max_A_m2, n_steps, t_per_step)


def eis(i_dc_A_m2, ratio, frequency_Hz):
    return generate_eis_load(i_dc_A_m2, ratio, frequency_Hz)


def ast_cycling(period_s, i_low_A, i_high_A, smoothing, Aact):
    return generate_ast_load(period_s, i_low_A, i_high_A, smoothing, Aact)


PROFILE_KINDS = ("Constant", "Step", "Polarization", "EIS", "AST cycling")


def sample(profile_func, t_span, n=400):
    ts = np.linspace(t_span[0], t_span[1], n)
    ys = np.array([profile_func(t) for t in ts])
    return ts, ys
