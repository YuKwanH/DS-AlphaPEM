# `model/kinetic_eq.py`

**Purpose.** Voltage-related helpers used both by the dynamic ODE and by the static algebraic solver.

## Key functions

- **`fdrop(x, operating_inputs, parameters) -> float`** — liquid-water-induced voltage-drop factor `f_drop` derived from the cathode-CL saturation `s_ccl` and the saturation regime parameters `a_slim`, `b_slim`, `a_switch`.
- **`Rproton(variables, parameters) -> tuple`** — proton resistance of the membrane (`Rmem` per node) plus catalyst-layer resistances (`Rccl`, `Racl`) consistent with `lambda_*` and Tafel-Arrhenius.
- Helpers for the activation overpotential, exchange current density temperature dependence, and Nernst voltage assembly.

## Related

- [`model/static.md`](static.md) and [`model/dualscale.md`](dualscale.md) call these helpers when computing `Ucell`, `eta_act`, `eta_conc`, `Rohm`.
