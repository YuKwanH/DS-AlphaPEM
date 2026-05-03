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