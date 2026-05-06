# Documentation index

One short companion doc per Python source file. Layout mirrors the project tree.

## Top-level

- [`main.md`](main.md) — minimal example entry point.

## `config/`
- [`config/initialize.md`](config/initialize.md) — operating inputs, geometry, materials, numerics, initial state vector.
- [`config/settings.md`](config/settings.md) — solver variable / flux name lists, plot styling, spatial node grids.

## `data/`
- [`data/export.md`](data/export.md) — load experimental Excel datasets (polarization, HFR, EIS, auxiliary).

## `model/`
- [`model/dualscale.md`](model/dualscale.md) — dual-scale `PEMFC` class (macro electrochem coupled with Pt micro-kinetics).
- [`model/dynamic.md`](model/dynamic.md) — legacy time-marched `PEMFC_dyn` class.
- [`model/static.md`](model/static.md) — algebraic steady-state `PEMFC_stat` class for polarization curves.
- [`model/state_eq.md`](model/state_eq.md) — region-by-region ODE right-hand-side (`dxdt_*` functions).
- [`model/kinetic_eq.md`](model/kinetic_eq.md) — voltage breakdown helpers (`fdrop`, `Rproton`, …).
- [`model/inst_values.md`](model/inst_values.md) — instantaneous physical quantities and mass flows used by `dxdt`.
- [`model/coefficients.md`](model/coefficients.md) — physical constants and material correlations.

## `modules/`
- [`modules/signals.md`](modules/signals.md) — current-density profile generators.
- [`modules/tests.md`](modules/tests.md) — packaged test-protocol runners (`constant_load_test`, …).
- [`modules/display.md`](modules/display.md) — spatial-profile plot helper.
- [`modules/control.md`](modules/control.md) — placeholder operating-condition controllers.

## `gui/` (Streamlit dashboard)
- [`gui/app.md`](gui/app.md) — Streamlit entry point and three-section layout.
- [`gui/parameters.md`](gui/parameters.md) — § 1 parameters panel.
- [`gui/options.md`](gui/options.md) — § 2 simulator-options panel.
- [`gui/results.md`](gui/results.md) — § 3 results panel (tabbed plots).
- [`gui/save.md`](gui/save.md) — save & download box.
- [`gui/runner.md`](gui/runner.md) — model dispatcher with LSODA fallback.
- [`gui/profiles.md`](gui/profiles.md) — thin adapter onto `modules.signals`.
- [`gui/style.md`](gui/style.md) — research-paper matplotlib `rcParams`.
