# `config/initialize.py`

**Purpose.** Single source of truth for default operating inputs, physical / material parameters, computing settings, and the initial-state-vector builder.

## Module-level dicts

| Dict | Meaning | Examples |
|---|---|---|
| `operating_inputs` | What the user *drives* (control inputs + setpoints) | `current_density`, `Tfc`, `Pa_des`, `Pc_des`, `Phi_a_des`, `Phi_c_des`, `Sa`, `Sc`, `Imin_aux` |
| `accessible_physical_parameters` | Geometry the user can set | `Aact`, `Hmem`, `Hgc`, `Wgc`, `Lgc` |
| `undetermined_physical_parameters` | Material constants needing calibration | `epsilon_*`, `i0_c_ref`, `kappa_c`, `Re`, `tau`, `Hcl`, `Hgdl`, … |
| `current_parameters` | Test-protocol parameters | `t_step`, `i_step`, `delta_pola`, `i_max_pola`, `i_EIS`, `f_EIS`, … |
| `computing_parameters` | Numerical settings | `n_gdl`, `n_mem`, `n_group_pt`, `max_step`, `type_*` switches |
| `parameters` | Merged dict of the four parameter groups above | (consumed by every model) |

## Functions

- **`init_x(operating_inputs, parameters) -> list[float]`** — assembles the 181-entry initial state vector consistent with the supplied operating point. Builds initial gas concentrations from RH/pressure, sets initial liquid saturation, computes the initial cathode overpotential from Tafel kinetics, and concatenates everything in the order expected by `solver_variable_names`.

## Notes

- Mutating these globals leaks into every model run — always `deepcopy` first if you need to override locally. The protocol runners in [`modules/tests.md`](../modules/tests.md) and the GUI runner do this for you.
