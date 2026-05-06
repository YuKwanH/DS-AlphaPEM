# `modules/tests.py`

**Purpose.** One call per fuel-cell test protocol. Every runner follows the same four-step pipeline so results are directly comparable:

1. build the `current_density` profile via `modules.signals`
2. construct a fresh `PEMFC` with default config (deep-copied)
3. integrate with `scipy.integrate.solve_ivp`
4. recover trajectories into `model.variables` / `model.fluxes` / `model.echem_traj`

The shared pipeline lives in the private `_run` helper, so the public functions are nearly trivial.

## Public API

All functions return `(model, sol)`. Defaults are sensible smoke-test values.

| Function | Protocol | Defaults |
|---|---|---|
| `constant_load_test(i_density, t_span)` | Steady current | `4 000 A/m²`, `0–20 s` |
| `step_load_test(tstart, tend, i_low, i_high, tau_switch, t_switch, t_span)` | Periodic square load (matches `simulation/control/square load.ipynb`) | notebook values, `0–20 s` |
| `polarization_test(i_max, n_steps, t_per_step)` | Staircase sweep | `16 500 A/m²`, 30 plateaus × 60 s |
| `eis_test(i_dc, ratio, frequency, n_periods)` | Sinusoidal EIS | `10 000 A/m²`, 5 % AC, 1 Hz, 10 periods |
| `ast_cycling_test(period, I_low, I_high, smoothing, n_cycles)` | AST cycling | 60 s, 1–25.8 A, 10 cycles |

All accept the keyword overrides `parameters`, `operating_inputs`, `method`, `max_step` (forwarded to the shared pipeline).

## Use

```python
from modules.tests import step_load_test, polarization_test
model, sol = step_load_test()                          # notebook-default square load
model, sol = polarization_test(i_max=16500, n_steps=30, t_per_step=60)
```

## Related

- [`modules/signals.md`](signals.md) — the underlying profile builders.
- [`gui/runner.md`](../gui/runner.md) — the GUI's parallel of `_run` (with LSODA fallback for newer scipy).
