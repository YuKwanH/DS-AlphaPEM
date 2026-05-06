# `gui/parameters.py`

**Purpose.** Renders § 1 — the parameter panel — and writes user edits back into `state["params"]` and `state["op_inputs"]` (deep copies, so the global config in `config.initialize` is never mutated).

## Layout

- A multiselect at the top picks which region groups are visible.
- Each visible group is an `st.expander` with a 2-column grid of `st.number_input` fields.
- Field labels are short (`Tfc (K)`); the descriptive name is the hover tooltip.

## Region groups (`PARAM_GROUPS`)

| Group | Parameters |
|---|---|
| Operating | `Tfc`, `Pa_des`, `Pc_des`, `Phi_a_des`, `Phi_c_des`, `Sa`, `Sc`, `Imin_aux` |
| GC (gas channel) | `Hgc`, `Wgc`, `Lgc`, `Aact` |
| GDL | `Hgdl`, `epsilon_gdl`, `tau` |
| CL (catalyst layer) | `Hcl`, `epsilon_cl`, `epsilon_c`, `epsilon_mc`, `i0_c_ref`, `kappa_c`, `C_scl` |
| MEM (membrane) | `Hmem`, `kappa_co`, `Re`, `e` |
| Saturation transitions | `a_slim`, `b_slim`, `a_switch` |
| Numerics | `max_step`, `n_gdl`, `n_mem`, `n_group_pt` |

## Public API

- **`render(state) -> state`** — builds the widgets in the current Streamlit container.

## Related

- [`config/initialize.md`](../config/initialize.md) — supplies the defaults shown in the form.
