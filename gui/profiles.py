"""Current-density profile generators.

Each builder returns a callable ``t -> i (A/m^2)`` consumable by the PEMFC
models, which all expect ``operating_inputs['current_density']`` to be such
a function.
"""

import numpy as np


def constant(i_density_A_cm2):
    i = float(i_density_A_cm2) * 1e4
    return lambda t: i


def step(i_low_A_cm2, i_high_A_cm2, t_switch, smoothing=1.5):
    i_low = float(i_low_A_cm2) * 1e4
    i_high = float(i_high_A_cm2) * 1e4
    return lambda t: i_low + (i_high - i_low) * 0.5 * (1.0 + np.tanh(smoothing * (t - t_switch)))


def polarization_ramp(i_max_A_cm2, n_steps, t_per_step):
    i_max = float(i_max_A_cm2) * 1e4
    levels = np.linspace(i_max / n_steps, i_max, n_steps)

    def profile(t):
        idx = min(int(t // t_per_step), n_steps - 1)
        return float(levels[idx])

    return profile


def eis(i_dc_A_cm2, ratio, frequency_Hz):
    i_dc = float(i_dc_A_cm2) * 1e4
    i_ac = ratio * i_dc
    omega = 2.0 * np.pi * float(frequency_Hz)
    return lambda t: i_dc + i_ac * np.sin(omega * t)


def ast_cycling(period_s, i_low_A, i_high_A, smoothing, Aact):
    period = float(period_s)
    i_low = float(i_low_A)
    i_high = float(i_high_A)
    smooth = float(smoothing)
    area = float(Aact)

    def profile(t):
        phase = 2.0 * np.pi * t / period
        wave = 0.5 * (1.0 + np.tanh(smooth * np.cos(phase)))
        current_A = i_low + (i_high - i_low) * wave
        return current_A / area

    return profile


PROFILE_KINDS = ("Constant", "Step", "Polarization", "EIS", "AST cycling")


def sample(profile_func, t_span, n=400):
    ts = np.linspace(t_span[0], t_span[1], n)
    ys = np.array([profile_func(t) for t in ts])
    return ts, ys
