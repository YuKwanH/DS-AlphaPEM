# `model/inst_values.py`

**Purpose.** Computes every *instantaneous* physical quantity that the region-by-region `dxdt_*` functions need: concentrations, partial pressures, humidities, mass flows, source terms.

## Key functions

- **`dif_eq_int_values(t, x, operating_inputs, parameters) -> dict`** — pulls state, parameters, operating inputs and produces all derived values (`Pagc`, `Pcgc`, `Phi_agc`, gas molar masses, concentrations, …) needed by the ODE RHS.
- **`calculate_flows(t, x, operating_inputs, parameters, **inst_states) -> dict`** — builds all transport fluxes (`Jv_*`, `Jl_*`, `J_H2_*`, `J_O2_*`, `JT_*`, `J_lambda_*`) using the instantaneous values.

## Use

These are typically not called directly — `PEMFC.dxdt` invokes them once per RHS evaluation and merges the results into a single kwargs dict that fans out to every region module.

## Related

- [`model/state_eq.md`](state_eq.md) — consumer of the kwargs assembled here.
- [`model/coefficients.md`](coefficients.md) — material correlations (`Psat`, `Dw`, `lambda_eq`, …) imported throughout this module.
