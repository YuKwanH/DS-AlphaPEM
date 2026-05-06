# `gui/save.py`

**Purpose.** Save & download box rendered below the result section. Lets the user pick a format, specify a directory + filename, and either write the file to disk on the local machine or trigger a browser download.

## Format options (`FORMATS`)

| Format | Extension | MIME |
|---|---|---|
| CSV | `.csv` | `text/csv` |
| NumPy (`.npz`) | `.npz` | `application/octet-stream` |
| Excel (`.xlsx`) | `.xlsx` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |

## Behaviour

- Filename is auto-suggested as `pemfc_<variant>_<profile>_<YYYYMMDD_HHMMSS><ext>`.
- For transient runs the data dump joins `model.variables`, `model.fluxes`, and `model.echem_traj` (column names prefixed `var_`, `flux_`, `echem_`). 2-D entries (e.g. per-node profiles) are auto-expanded into `name[0]`, `name[1]`, ….
- For polar runs the dump is a small `(i_A_per_m2, i_A_per_cm2, Ucell_V)` table.
- The Save button writes the file under the chosen directory; the Download button bypasses the path and lets the browser save dialog handle it.

## Public API

- **`render(state) -> None`** — main entry; renders nothing useful when there's no successful run yet (shows a "Run a simulation first" caption instead).
- **`_serialize(res, fmt) -> bytes`** — internal helper, format-specific.
- **`_to_dataframe(d)`** — internal helper that pads / truncates / expands ragged trajectories so pandas accepts them.
