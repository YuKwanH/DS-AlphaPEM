# `model/state_eq.py`

**Purpose.** Region-by-region time derivatives that feed `PEMFC.dxdt`. Each function writes into the shared `dif` dict.

## Functions

| Function | Region |
|---|---|
| `dxdt_AGC` | Anode gas channel (vapour, H₂) |
| `dxdt_CGC` | Cathode gas channel (vapour, O₂) |
| `dxdt_AGDL` | Anode gas-diffusion layer (per-node vapour, H₂, saturation, T) |
| `dxdt_CGDL` | Cathode gas-diffusion layer |
| `dxdt_ACL` | Anode catalyst layer |
| `dxdt_CCL` | Cathode catalyst layer |
| `dxdt_MEM` | Membrane (per-node λ, dissolved O₂/H₂, Pt²⁺, T) |
| `dxdt_CP` | Compressor flow `Wcp` |
| `dxdt_Manifold` | Anode/cathode supply/exhaust manifold pressures and humidities |
| `dxdt_TH` | Thermal coupling between regions |
| `dxdt_U` | Cathode overpotential `eta_c` |
| `dxdt_N2` | N₂ accumulation in the cathode |
| `dxdt_PRD` | Pt particle radius distribution + ECSA + oxide coverage |

All functions follow the same shape: take pre-computed instantaneous values via keyword arguments, write `dif["d<name> / dt"] = …`. Unused kwargs are absorbed by `**kwargs` so the dispatch in `PEMFC.dxdt` can pass everything to everyone.

## Related

- [`model/inst_values.md`](inst_values.md) — produces the kwargs (concentrations, fluxes, source terms) consumed here.
- [`model/dualscale.md`](dualscale.md) — orchestrator that calls every `dxdt_*` per RHS evaluation.
