# `model/dualscale.py`

**Purpose.** Primary fuel-cell model. Couples the macro-scale electrochemistry (gas/liquid transport, manifolds, voltage) with a Pt-particle micro-kinetics layer (`S_N`, `theta`, PRD evolution).

## Class `PEMFC`

```python
PEMFC(param, operating_inputs, variable_names, flux_names)
```

Construction does the heavy bookkeeping:
- Defensive copies of `param`, `variable_names`, `flux_names` so the caller's dicts/lists are never mutated.
- Discretises GDL, membrane, and Pt-particle bins into individual state-variable names.
- Computes the Pt-particle radius grid (`r_m`) and the initial particle radius distribution (`prd0`) via `initPRD`.
- Initialises empty `variables`, `fluxes`, and `echem_traj` recorders.
- Pre-builds the index map (`_idx`) and `dif`-key tuple used by the hot-path `dxdt`.

## Key methods

- **`dxdt(t, x) -> np.ndarray`** — ODE right-hand side. Calls each region module from [`state_eq.md`](state_eq.md) plus the BoP / temperature / overpotential / N₂ / PRD modules. Returns the stacked time derivatives.
- **`_recovery(sol)`** — populates `variables`, `fluxes`, and `echem_traj` from a finished `solve_ivp` solution, ready for plotting.
- **`compute_jac_sparsity(y0)`** — builds the Jacobian sparsity pattern (used by `solve_ivp` for cheaper BDF steps).

## Use

```python
from scipy.integrate import solve_ivp
from config.initialize import init_x, parameters, operating_inputs
from config.settings import solver_variable_names, solver_flux_names
from model.dualscale import PEMFC

m = PEMFC(param=parameters, operating_inputs=operating_inputs,
          variable_names=solver_variable_names, flux_names=solver_flux_names)
sol = solve_ivp(m.dxdt, (0, 30), init_x(operating_inputs, parameters), method="BDF", max_step=0.1)
m._recovery(sol)
```

## Related

- [`modules/tests.md`](../modules/tests.md) wraps the entire pipeline in one call per protocol.
- [`gui/runner.md`](../gui/runner.md) is the GUI's adapter that calls the same model.
