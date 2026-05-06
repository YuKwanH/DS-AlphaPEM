# `gui/profiles.py`

**Purpose.** Thin adapter onto [`modules/signals.md`](../modules/signals.md). Keeps the GUI sub-forms decoupled from the canonical profile builders.

## Functions

| Function | Delegates to |
|---|---|
| `constant(i_density_A_m2)` | `generate_constant_load` |
| `step(tstart, tend, i_low, i_high, tau_switch, t_switch)` | `generate_step_load` |
| `polarization_ramp(i_max, n_steps, t_per_step)` | `generate_polarization_load` |
| `eis(i_dc, ratio, frequency)` | `generate_eis_load` |
| `ast_cycling(period, I_low, I_high, smoothing, Aact)` | `generate_ast_load` |

## Constants and helpers

- **`PROFILE_KINDS`** — tuple of human-readable names matching the radio in [`gui/options.md`](options.md): `("Constant", "Step", "Polarization", "EIS", "AST cycling")`.
- **`sample(profile_func, t_span, n=400) -> (ts, ys)`** — densely sampled `t, i(t)` pair used by the live preview plot in § 2.
