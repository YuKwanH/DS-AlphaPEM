# `model/coefficients.py`

**Purpose.** Physical constants and material correlations imported via `from model.coefficients import *` everywhere a kinetic / transport equation needs them.

## Categories

| Category | Examples |
|---|---|
| Universal constants | `F`, `R`, `N_A`, `M_O2`, `M_H2`, `M_N2`, `M_H2O`, `gamma`, `gamma_H2` |
| Ambient | `Text`, `Pext`, `Phi_ext`, `yO2_ext` |
| Material densities | `rho_mem`, `rho_cl`, `rho_gdl`, `rho_H2O(T)` |
| Membrane | `M_eq`, `theta_c_gdl`, `theta_c_cl`, `Kshape` |
| Phase change | `gamma_cond`, `gamma_evap` |
| Voltage | `C_O2ref`, `alpha_c`, `E0`, `Pref`, `Eact` |
| Correlations (functions) | `Psat(T)`, `C_v_sat(T)`, `Dw(lambda, T)`, `lambda_eq(C_v, s, T, K)`, `Da_eff`, `Dc_eff`, `K0`, `nu_l`, `nu_g`, `sigma`, `h_a`, `h_c`, `Cp_gdl`, … |

## Use

Treat as a constants/utilities namespace — every kinetic, transport, or static-solver file does `from model.coefficients import *`.

## Related

- Anything inside `model/` will reach for these helpers.
