import math
import numpy as np

def generate_step_load(tstart, tend, i_low, i_high, tau_switch, t_switch):

    # --- Pre-computed constants (rebuild if any global above changes) ---
    _period  = tend - tstart
    _i_mid   = 0.5 * (i_low  + i_high)
    _di_half = 0.5 * (i_high - i_low)
    _t_c     = tau_switch + 0.5 * t_switch   # tanh center, relative to tstart
    _inv_w   = 6.0 / t_switch                


    def current(t):
        """
        Periodic, C^∞-smooth ramp current density i(t) [A/m^2].

        Within each period of length (tend - tstart), starting at tstart:

            i(tau) = (i_low + i_high)/2
                + (i_high - i_low)/2
                    * tanh( (tau - (tau_switch + t_switch/2)) / (t_switch/6) )

        The transition is centered at tau = tau_switch + t_switch/2 with a
        characteristic width of t_switch/6, so ~99.7 % of the step is
        completed within [tau_switch, tau_switch + t_switch].
        Accepts scalar or array t.
        """
        tau = (t - tstart) % _period
        arg = (tau - _t_c) * _inv_w
        # Scalar fast path matters inside ODE/IDA inner loops
        if isinstance(arg, np.ndarray):
            return _i_mid + _di_half * np.tanh(arg)
        return _i_mid + _di_half * math.tanh(arg)
    
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
