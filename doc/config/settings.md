# `config/settings.py`

**Purpose.** Bookkeeping for the ODE solver and the spatial post-processing grid. Imported wherever a list of state names, flux names, or node coordinates is needed.

## Lists / arrays

- **`solver_variable_names`** — ordered names of every state variable consumed by `PEMFC.dxdt`. Discretised regions (e.g. `C_v_agdl_1` … `C_v_agdl_n`) are expanded inside the model `__init__`.
- **`solver_flux_names`** — ordered transport-flux names recorded into `model.fluxes`.
- **`nodes`** — physical positions of the spatial grid (`agc`, `agdl_*`, `acl`, `mem_*`, `ccl`, `cgdl_*`, `cgc`).
- **`borders`** — cumulative thicknesses marking region interfaces (used for vertical guides in profile plots).
- **`nodes_postfix`, `nodes_names_vp`, `nodes_names_H2`, `nodes_name_O2`, `nodes_names_s`, `nodes_lambda`, `nodes_T`** — per-species lookup helpers.

## Plot styling

- **`colormap_temp`, `linemap_pressure`, `markermap_rh`** — categorical maps for experimental conditions (50/60/70 °C, 1.3/1.4/1.5 bar, 0/0.5 RH).
- **`get_plot_properties(cond_key)` / `plot_condition(...)`** — extract style attributes from a condition string and plot a curve consistently.

## Functions

- **`expand_profile_on_nodes(profile_key, compact_values)`** — pad a per-species profile so it lines up with the global node grid (used by the spatial-profile plot).
- **`has_species_value(profile_key, postfix)`** — quick boolean check for "does species X exist in region Y".
