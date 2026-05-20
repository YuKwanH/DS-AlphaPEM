"""NaN / Inf watchdog for `scipy.integrate.solve_ivp`.

Used to localise the first non-finite value encountered during a PEMFC
integration so the user can identify which state variable diverges and
whether the divergence first appears in the state itself or in `dxdt`.
"""
import numpy as np
from scipy.integrate import solve_ivp


def trace_nan(model, t_span, y0, method="BDF", max_step=None,
              verbose=True, max_offenders=10, **solve_ivp_kwargs):
    """Run `solve_ivp(model.dxdt, ...)` with a watchdog that catches the
    first non-finite (NaN or Inf) value in either the input state ``x`` or
    the returned ``dxdt``.

    Parameters
    ----------
    model : object
        Must expose ``dxdt(t, x)`` and ``variable_names`` (list of state names).
        Any object satisfying that interface works — typically a
        ``model.dualscale.PEMFC`` instance.
    t_span, y0, method, max_step, **solve_ivp_kwargs :
        Forwarded to ``scipy.integrate.solve_ivp``.
    verbose : bool
        If True, print a one-line warning at the first NaN/Inf detection.
    max_offenders : int
        How many simultaneously non-finite variables to record.

    Returns
    -------
    info : dict with keys
        sol       : the OdeSolution returned by ``solve_ivp``
                    (None if ``solve_ivp`` raised before returning).
        variable  : name of the first non-finite variable
                    (None if the integration was clean).
        source    : "state" (NaN in input ``x``) or
                    "derivative" (NaN in ``dxdt`` output).
        t         : simulation time of first detection.
        value     : the offending value.
        offenders : list of ``(name, value)`` for all variables that were
                    non-finite simultaneously at that moment, up to
                    ``max_offenders`` entries.
        error     : (optional) text of any exception ``solve_ivp`` raised.
    """
    variable_names = list(model.variable_names)
    first = {"variable": None, "source": None, "t": None, "value": None,
             "offenders": []}

    def _scan(arr, label):
        if first["variable"] is not None:
            return
        bad = ~np.isfinite(arr)
        if not bad.any():
            return
        bad_idxs = np.where(bad)[0]
        idx0 = int(bad_idxs[0])
        first["variable"] = variable_names[idx0]
        first["source"] = label
        first["value"] = float(arr[idx0])
        first["offenders"] = [(variable_names[i], float(arr[i]))
                              for i in bad_idxs[:max_offenders]]

    def tracked_dxdt(t, x):
        if first["variable"] is None:
            _scan(x, "state")
            if first["variable"] is not None:
                first["t"] = float(t)
                if verbose:
                    n = len(first["offenders"])
                    extra = "" if n == 1 else f" (+{n - 1} more)"
                    print(f"[NaN @ t={t:.6g}] non-finite STATE in "
                          f"'{first['variable']}' = {first['value']}{extra}")
        dx = model.dxdt(t, x)
        if first["variable"] is None:
            _scan(dx, "derivative")
            if first["variable"] is not None:
                first["t"] = float(t)
                if verbose:
                    n = len(first["offenders"])
                    extra = "" if n == 1 else f" (+{n - 1} more)"
                    print(f"[NaN @ t={t:.6g}] non-finite dxdt of "
                          f"'{first['variable']}' = {first['value']}{extra}")
        return dx

    info = {"sol": None, **first}
    if max_step is None:
        max_step = np.inf
    try:
        sol = solve_ivp(fun=tracked_dxdt, t_span=t_span, y0=y0,
                        method=method, max_step=max_step,
                        **solve_ivp_kwargs)
        info["sol"] = sol
        if verbose and not sol.success:
            print(f"solve_ivp returned success=False at t={sol.t[-1] if len(sol.t) else 'n/a'}: "
                  f"{sol.message}")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"solve_ivp raised: {info['error']}")

    info.update(first)
    return info
