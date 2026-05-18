import math
import numpy as np

def generate_step_load(tstart, tend, i_low, i_high, tau_switch, t_switch):

    # --- Pre-computed constants (rebuild if any global above changes) ---
    _period  = tend - tstart
    _di      = i_high - i_low
    _t_rise  = float(tau_switch)               # rising-edge centre, in seconds from tstart
    _t_fall  = _period - float(tau_switch)     # falling-edge centre (symmetric within the period)
    _inv_w   = 6.0 / float(t_switch)           # 99.7 % of each transition completes within t_switch


    def current(t):
        """
        Periodic symmetric square-wave current density i(t) [A/m^2].

        Two C^infinity-smooth tanh transitions per period
        (tend - tstart):

            i(tau) = i_low + (i_high - i_low) * 0.5
                            * ( tanh((tau - tau_switch ) * 6 / t_switch)
                              - tanh((tau - (T - tau_switch)) * 6 / t_switch) )

        with T = tend - tstart. The rising edge is centred at
        ``tau_switch`` seconds into the period and the falling edge at
        ``T - tau_switch``, so the wave is symmetric around T/2:

          *   tau in [0, tau_switch - t_switch/2]            -> i = i_low
          *   tau in [tau_switch -+ t_switch/2]              -> smooth rise
          *   tau in [tau_switch + t_switch/2,
                      T - tau_switch - t_switch/2]           -> i = i_high
          *   tau in [T - tau_switch -+ t_switch/2]          -> smooth fall
          *   tau in [T - tau_switch + t_switch/2, T]        -> i = i_low

        and the boundary i(0) = i(T) = i_low is continuous (no jump at
        the period seam, unlike the legacy single-edge form).

        Typical 50/50 duty cycle for a 6 s period:
            tau_switch = 1.5   (so the transitions are at 1.5 s and 4.5 s)
            t_switch   = 0.5   (sharp but smooth transitions)

        Accepts scalar or array t.
        """
        tau = (t - tstart) % _period
        a   = (tau - _t_rise) * _inv_w
        b   = (tau - _t_fall) * _inv_w
        if isinstance(tau, np.ndarray):
            return i_low + _di * 0.5 * (np.tanh(a) - np.tanh(b))
        return i_low + _di * 0.5 * (math.tanh(a) - math.tanh(b))

    return current

def generate_constant_load(i_density):

    # --- Pre-computed constants ---
    _i = float(i_density)


    def current(t):
        """
        Constant current density i(t) = i_density [A/m^2].

        Returned for any t; broadcasts to the shape of t when t is an
        np.ndarray (so the result composes with vectorised consumers).
        Accepts scalar or array t.
        """
        if isinstance(t, np.ndarray):
            return np.full_like(t, _i, dtype=float)
        return _i

    return current


def generate_polarization_load(i_max, n_steps, t_per_step):

    # --- Pre-computed constants ---
    _n_steps = int(n_steps)
    _last    = _n_steps - 1
    _t_per   = float(t_per_step)
    _levels  = np.linspace(float(i_max) / _n_steps, float(i_max), _n_steps)


    def current(t):
        """
        Staircase polarization sweep current density i(t) [A/m^2].

        Holds discrete current levels for t_per_step seconds each,
        ramping from i_max / n_steps up to i_max over n_steps
        plateaus. The k-th plateau occupies t in [k*t_per_step,
        (k+1)*t_per_step); after the final plateau the current stays at
        i_max.
        Accepts scalar or array t.
        """
        if isinstance(t, np.ndarray):
            idx = np.floor(t / _t_per).astype(int)
            np.clip(idx, 0, _last, out=idx)
            return _levels[idx]
        idx = max(0, min(int(t // _t_per), _last))
        return float(_levels[idx])

    return current


def generate_eis_load(i_dc, ratio, frequency):

    # --- Pre-computed constants ---
    _i_dc  = float(i_dc)
    _i_ac  = float(ratio) * _i_dc
    _omega = 2.0 * math.pi * float(frequency)


    def current(t):
        """
        Sinusoidal EIS perturbation current density i(t) [A/m^2]:

            i(t) = i_dc + (ratio * i_dc) * sin(2 * pi * frequency * t)

        Used to extract the impedance spectrum of the cell at the DC
        operating point i_dc with relative AC amplitude ratio.
        Accepts scalar or array t.
        """
        if isinstance(t, np.ndarray):
            return _i_dc + _i_ac * np.sin(_omega * t)
        return _i_dc + _i_ac * math.sin(_omega * t)

    return current


def generate_ast_load(period, I_low, I_high, smoothing, Aact):

    # --- Pre-computed constants ---
    _period = float(period)
    _I_low  = float(I_low)
    _delta  = float(I_high) - _I_low
    _smooth = float(smoothing)
    _aact   = float(Aact)
    _two_pi_over_T = 2.0 * math.pi / _period


    def current(t):
        """
        Smoothed square-wave Accelerated Stress Test (AST) cycling load
        current density i(t) [A/m^2].

        Within each period of length period the total cell current
        oscillates between I_low and I_high (in Amperes); the
        returned value is divided by the active area Aact to express
        it as a current density:

            wave  = 0.5 * (1 + tanh(smoothing * cos(2*pi*t/period)))
            I(t)  = I_low + (I_high - I_low) * wave            # Amperes
            i(t)  = I(t) / Aact                                # A/m^2

        Accepts scalar or array t.
        """
        if isinstance(t, np.ndarray):
            phase = _two_pi_over_T * t
            wave = 0.5 * (1.0 + np.tanh(_smooth * np.cos(phase)))
            return (_I_low + _delta * wave) / _aact
        phase = _two_pi_over_T * t
        wave = 0.5 * (1.0 + math.tanh(_smooth * math.cos(phase)))
        return (_I_low + _delta * wave) / _aact

    return current
