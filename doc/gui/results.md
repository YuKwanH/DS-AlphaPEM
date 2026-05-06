# `gui/results.py`

**Purpose.** Renders § 3 — the result tabs — for the most recent simulation. Handles two top-level shapes:

- **transient**: `model.variables`, `model.echem_traj`, `model.fluxes` populated; `solution` is a scipy `OdeResult`.
- **polar**: a dict `{"i_A_m2", "Ucell_V"}` from the static solver.

## Tabs (transient)

| Tab | Plot |
|---|---|
| Cell performance | Load current and cell voltage vs time |
| Spatial profile | `build_profile_figure(solution, model, t_index)` from [`modules/display.md`](../modules/display.md) at a slider-selected time index |
| Manifolds | 2 × 2 panel of supply / exhaust pressures and humidities |
| Water content | `lambda_acl`, `lambda_ccl`, plus a heat-map of `lambda_mem_*` over time |
| Degradation | Membrane thickness `delta_mem` and ECSA ratio `S_N` |
| Custom | Multi-select of any state variable; plotted with units pulled from `VAR_UNITS` / `PREFIX_UNITS` |

## Polar tab (Static variant)

Single Ucell-vs-i plot (axes in A/cm²).

## Helpers

- **`VAR_UNITS`, `PREFIX_UNITS`** — variable-name → `(label, unit)` lookup tables, with prefix fallback for discretised names (e.g. `C_H2_agdl_3` → `H₂ concentration (mol/m³)`).
- **`axis_label(name) -> str`** — composed `"<label> (<unit>)"`.

## Public API

- **`render(state) -> state`** — main entry, picks the right rendering branch from `state["last_result"]`.
