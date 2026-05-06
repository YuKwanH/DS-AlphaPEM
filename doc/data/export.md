# `data/export.py`

**Purpose.** Loader for the Baltic 300 W experimental dataset shipped under `data/`. Returns one DataFrame (or list of DataFrames) per operating condition.

## API

- **`export_experiment_data(data_type)`** — `data_type` is one of:
  - `"pola"` — polarization curves (V_FC vs I_LOAD)
  - `"hfr"` — high-frequency-resistance measurements
  - `"auxiliary"` — auxiliary sensor channels (manifold pressures, temperatures, flows…)
  - `"eis"` — electrochemical-impedance spectra (one DataFrame per frequency)

  Returns a `dict` keyed by condition string in the format `RHA{rh}/RHC{rh}_P{p}_T{t}` (or `T{t}_P{p}_HRC{rh}` for auxiliary), values are pandas DataFrames or lists of DataFrames.

## Use

```python
from data.export import export_experiment_data
pola = export_experiment_data("pola")
print(pola["RHA0/RHC0.5_P1.5_T70"].head())
```
