# `gui/options.py`

**Purpose.** Renders § 2 — the simulator-options panel: model variant, test profile, time span, solver, mesh — and previews the resulting current-density profile so the user sees what they're about to feed the model.

## What's rendered

1. Run-error banner (top of the panel) when `last_result.success` is `False`.
2. **Model variant** radio: `Dual-scale` / `Dynamic` / `Static`.
3. **Test profile** radio: `Constant` / `Step` / `Polarization` / `EIS` / `AST cycling` — each kind reveals its own sub-form with the protocol-specific arguments.
4. **Time span & solver**: `t_start`, `t_end`, `max_step`, `method` (`BDF`, `Radau`, `LSODA`, `RK45`).
5. **Mesh**: `n_gdl`, `n_mem`, `n_group_pt`.
6. Live current-density preview rendered with the project plot style.

## Public API

- **`render(state) -> state`** — main entry, builds every widget.
- **`build_profile_func(state) -> callable`** — assembles the `current_density(t)` function from the current sub-form values via `gui.profiles`.

## Related

- [`gui/profiles.md`](profiles.md) — adapter that wires the profile builder to `modules.signals`.
- [`gui/runner.md`](runner.md) — receives the profile callable and runs the simulation.
