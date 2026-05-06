# `model/static.py`

**Purpose.** Steady-state algebraic model used to draw polarization curves quickly without time integration. Solves the GDL water/gas distribution by fixed-point iteration on the net membrane water flux $J_{net}$.

## Class `PEMFC_stat`

```python
PEMFC_stat(parameters, operating_inputs)
```

Expects extra entries in `operating_inputs` that are not part of the dynamic-model dict: `Win_a`, `Wout_a`, `Win_c`, `Wout_c` (anode/cathode inlet/outlet mass flows). The GUI's static adapter ([`gui/runner.md`](../gui/runner.md)) derives them from stoichiometry.

## Method

- **`solve(i)`** — given a current density $i$ (A/m²), iterates until $|J_{net} - J_{mem}| < 10^{-4}$ and returns a dict with:
  - `Ueq`, `eta_c` — Nernst voltage and cathode overpotential
  - `Rohm`, `Rccl`, `Racl` — proton-resistance contributions
  - `lambda_*`, `C_v_*`, `s_*`, `J*` — full water/gas distribution

## Use

```python
m = PEMFC_stat(parameters=parameters, operating_inputs=op_inputs_with_flows)
res = m.solve(0.4e4)
Ucell = res["Ueq"] - res["eta_c"] - 0.4e4 * (res["Rohm"] + res["Rccl"] + res["Racl"])
```

## Related

- [`gui/runner.md`](../gui/runner.md) — wraps the static solver into a polarization sweep.
- [`config/initialize.md`](../config/initialize.md) — supplies the parameter dict (must contain `Hcl`, `Hmem`, `Hgdl`, etc.).
