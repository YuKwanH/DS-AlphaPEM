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


# ---------------------------------------------------------------------------
# Soft start
# ---------------------------------------------------------------------------
# The model's default initial state (init_x / init_x_for) supports an
# instant start only up to moderate current densities: a constant load of
# 4000 A/m^2 from t=0 integrates fine, but >= ~8000 A/m^2 blows up the very
# first solver steps (both BDF and LSODA stall within ~0.1 s). RAMPED
# transitions to high current are no problem -- the Step profile reaches
# 12000 A/m^2 through a ~3 s tanh ramp and works.
#
# Profiles that demand a high current at t=0 (AST cycling starts at I_high;
# EIS starts near i_DC) therefore get a short cosine ramp-in from the
# profile's own minimum level. Signals already starting low are returned
# unchanged, so Constant / Step / Polarization behave exactly as before.

SOFT_START_THRESHOLD_A_M2 = 5000.0   # instant starts above this get ramped
SOFT_START_DURATION_S     = 1.5      # ramp-in length


def soft_start(profile_func, t_ramp=SOFT_START_DURATION_S,
               threshold=SOFT_START_THRESHOLD_A_M2):
    """Wrap ``profile_func`` with a smooth ramp-in when it starts too high.

    If ``profile_func(0) <= threshold`` the function is returned unchanged.
    Otherwise the returned wrapper blends from the profile's minimum level
    up to the true signal over ``t_ramp`` seconds using a C1-continuous
    half-cosine, leaving everything at t >= t_ramp exactly identical.
    """
    i0 = float(profile_func(0.0))
    if i0 <= threshold:
        return profile_func

    # Start the ramp from the profile's own low level (sampled over one
    # ramp window plus a safety factor), floored at a small positive value.
    ts_probe = np.linspace(0.0, max(t_ramp * 10.0, 10.0), 200)
    i_floor = max(float(np.min([profile_func(t) for t in ts_probe])), 1.0)

    def wrapped(t):
        y = profile_func(t)
        if isinstance(t, np.ndarray):
            w = np.clip(t / t_ramp, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            return i_floor + (y - i_floor) * w
        if t >= t_ramp:
            return y
        w = min(max(t / t_ramp, 0.0), 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        return i_floor + (y - i_floor) * w

    return wrapped


def sample(profile_func, t_span, n=400):
    ts = np.linspace(t_span[0], t_span[1], n)
    ys = np.array([profile_func(t) for t in ts])
    return ts, ys
