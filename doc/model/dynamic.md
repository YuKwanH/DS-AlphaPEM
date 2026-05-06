# `model/dynamic.py`

**Purpose.** Legacy time-marched model (`PEMFC_dyn`) without the Pt micro-kinetics that `PEMFC` (dual-scale) adds. Kept for backward compatibility and as an option in [`gui/options.md`](../gui/options.md).

## Class `PEMFC_dyn`

```python
PEMFC_dyn(parameters, operating_inputs, initial_variable_values, time_interval=None)
```

Interface mirrors `PEMFC`:
- `dxdt(t, x_sol)` — same RHS contract.
- `variables`, `dif_eq`, `ec_kinetics` — recorded trajectory dicts.

Differs from `PEMFC` in that the micro-kinetics state (`S_N_ccl_*`, `theta_ccl_*`) is initialised but the macro coupling is simpler.

## Use

```python
from model.dynamic import PEMFC_dyn
m = PEMFC_dyn(parameters=parameters, operating_inputs=operating_inputs,
              initial_variable_values=init_x(operating_inputs, parameters))
```

For new work, prefer `PEMFC` from [`model/dualscale.md`](dualscale.md).
