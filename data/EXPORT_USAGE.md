# Export Experiment Data - Usage Guide

## Function Overview

The `export_experiment_data()` function extracts and organizes all experimental data of a specified type across all operating conditions.

Returns a dictionary where:
- **Keys** = Operating condition identifiers
- **Values** = Data DataFrames (or list of DataFrames for EIS)

## Input Parameter

The function accepts a single string parameter specifying the data type:

```python
export_experiment_data(data_type)
```

**Supported data_type values:**
- `"pola"` — Polarization curves (Voltage vs Current)
- `"hfr"` — High Frequency Resistance measurements
- `"auxiliary"` — Auxiliary sensor measurements
- `"eis"` — Electrochemical Impedance Spectroscopy spectra

## Usage Examples

```python
from data.export_experiment import export_experiment_data

# Load all polarization curve data across all conditions
all_pola = export_experiment_data("pola")
for condition, df in all_pola.items():
    print(f"{condition}: {len(df)} voltage points")
    print(df.head())

# Load all HFR measurements
all_hfr = export_experiment_data("hfr")

# Load all auxiliary sensor data
all_aux = export_experiment_data("auxiliary")

# Load all EIS spectra
all_eis = export_experiment_data("eis")
for condition, spectra_list in all_eis.items():
    print(f"{condition}: {len(spectra_list)} frequency sweeps")
```

## Return Value Structure

### For "pola" and "hfr"
```python
{
    "T50_P300_HRC50": <DataFrame>,  # Columns: VFC, I_LOAD (for pola)
    "T50_P300_HRC0": <DataFrame>,   # Columns: I_LOAD, R (for hfr)
    "T60_P400_HRC50": <DataFrame>,
    ...
}
```

### For "auxiliary"
```python
{
    "RHA0/RHC0.5_P1.3_T50": <DataFrame>,  # Sensor measurements
    "RHA0.5/RHC0_P1.5_T70": <DataFrame>,
    ...
}
```

### For "eis"
```python
{
    "RHC0.5_P1.3_T50": [<DataFrame>, <DataFrame>, ...],  # List of spectra
    "RHC0_P1.5_T70": [<DataFrame>, ...],
    ...
}
```
Each list is sorted by current value (ascending).

## Condition Key Formats

The function internally handles different condition key formats used in different Excel files:

### Sheet name format (Pola/HFR)
```
T{t}_P{p}_HRC{rh}
```
Example: `"T50_P300_HRC50"` where:
- `T`: Temperature (50, 60, 70 °C)
- `P`: Pressure (300, 400, 500 mbar)
- `HRC`: Cathode humidity (0 or 50 representing 0% or 50%)

### Condition key format (Auxiliary)
```
RHA{rha}/RHC{rhc}_P{p}_T{t}
```
Example: `"RHA0/RHC0.5_P1.3_T50"` where:
- `RHA`: Anode relative humidity (0 to 0.5)
- `RHC`: Cathode relative humidity (0 to 0.5)
- `P`: Pressure in bar (1.3, 1.4, 1.5)
- `T`: Temperature (50, 60, 70 °C)

### EIS data format
Internally parsed from sheet names like `"I{current}_P{pressure}_{temp}_HR{humidity}"` and reorganized by condition.

## Error Handling

- Missing Excel files will print warnings but return empty dict
- Invalid `data_type` raises `ValueError` with list of valid options
- Gracefully handles missing/corrupted sheets with warnings
