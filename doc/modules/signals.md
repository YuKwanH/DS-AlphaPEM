# `modules/signals.py`

**Purpose.** Single source of truth for the current-density profiles. Each builder takes plain numbers and returns a callable `t -> i (A/m²)` consumable by `solve_ivp` via `operating_inputs["current_density"]`. Scalar and array `t` are both supported (scalar fast path inside ODE inner loops).

## Builders

| Function | Profile | Key arguments |
|---|---|---|
| `generate_constant_load(i_density)` | `i(t) = i_density` | A/m² |
| `generate_step_load(tstart, tend, i_low, i_high, tau_switch, t_switch)` | Periodic tanh-smoothed square load | period, plateaus, ramp centre + duration |
| `generate_polarization_load(i_max, n_steps, t_per_step)` | Staircase sweep `i_max/n_steps` → `i_max` | plateau hold time |
| `generate_eis_load(i_dc, ratio, frequency)` | `i_dc + ratio·i_dc · sin(2πft)` | AC perturbation around a DC bias |
| `generate_ast_load(period, I_low, I_high, smoothing, Aact)` | Smoothed square-wave AST load | total currents in A, divided by `Aact` to A/m² |

## Use

```python
from modules.signals import generate_step_load
profile = generate_step_load(0.0, 6.0, 20.0, 12000.0, 1.0, 3.0)
operating_inputs["current_density"] = profile
```

## Related

- [`modules/tests.md`](tests.md) wraps each builder in a one-call test runner.
- [`gui/profiles.md`](../gui/profiles.md) is the thin GUI adapter that re-exports these.
