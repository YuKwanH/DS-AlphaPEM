"""
Export experiment data organized by data type and operating condition.

This module provides a function to extract and organize all experimental data
from a specified group (polarization curves, HFR measurements, auxiliary states,
or EIS spectra) across all operating conditions.
"""

import re
import pandas as pd
from pathlib import Path


def export_experiment_data(data_type):
    """
    Export all experiment data for a specified data type across all conditions.

    Returns a dictionary where keys are operating condition identifiers and
    values are the corresponding data DataFrames or lists.

    Parameters
    ----------
    data_type : str
        Type of data to export. Must be one of:
        - "pola" : Polarization curve (Voltage vs Current)
        - "hfr" : High Frequency Resistance measurements
        - "auxiliary" : Auxiliary sensor measurements
        - "eis" : Electrochemical Impedance Spectroscopy spectra

    Returns
    -------
    dict
        Dictionary where:
        - Keys: Condition identifiers (format "RHA#/RHC#_P#_T#" or "T#_P#_HRC#")
        - Values: Data as pd.DataFrame for pola/hfr/auxiliary, or list of DataFrames for eis

        For "pola" and "hfr":
        - Columns: ["VFC", "I_LOAD"] for pola; ["I_LOAD", "R"] for hfr

        For "auxiliary":
        - Columns: Sensor measurements (temperature, pressure, humidity, etc.)

        For "eis":
        - Each key includes the current point, e.g. "I10_RHC0.5_P1.3_T50"
        - Value is a single pd.DataFrame for that specific spectrum

    Examples
    --------
    >>> # Load all polarization curves
    >>> all_pola = export_experiment_data("pola")
    >>> for condition, df in all_pola.items():
    ...     print(f"{condition}: {len(df)} points")

    >>> # Load all HFR data
    >>> all_hfr = export_experiment_data("hfr")

    Notes
    -----
    The function searches for Excel files (Polar_curves.xlsx, HFR.xlsx,
    auxiliary.xlsx, eis.xlsx) in the data directory.

    Condition key format: "RHA([0-9.]+)/RHC([0-9.]+)_P([0-9.]+)_T([0-9]+)"
    - RHA/RHC: Anode/Cathode relative humidity (0 to 0.5, representing 0-50%)
    - P: Pressure in bar (1.3, 1.4, or 1.5)
    - T: Temperature in °C (50, 60, or 70)

    Internal sheet name mappings:
    - RHC 0.5 ↔ HRC50, RHC 0 ↔ HRC0
    - P 1.3 ↔ P300, P 1.4 ↔ P400, P 1.5 ↔ P500
    """

    # Validate data_type input
    valid_types = {"pola", "hfr", "auxiliary", "eis"}
    data_type = data_type.lower()
    if data_type not in valid_types:
        raise ValueError(f"Invalid data_type: '{data_type}'. Must be one of {valid_types}")

    # Get data directory path
    data_dir = Path(__file__).parent

    result = {}

    if data_type in ["pola", "hfr"]:
        # ========== Load Polarization Curve or HFR Data ==========
        file_map = {"pola": "Polar_curves.xlsx", "hfr": "HFR.xlsx"}
        excel_file = data_dir / file_map[data_type]

        try:
            excel = pd.ExcelFile(excel_file)
            # Load all sheets from the Excel file
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                result[sheet_name] = df
        except FileNotFoundError:
            print(f"Warning: File not found: {excel_file}")

    elif data_type == "auxiliary":
        # ========== Load Auxiliary Data ==========
        aux_file = data_dir / "auxiliary.xlsx"
        try:
            excel = pd.ExcelFile(aux_file)
            # Load all sheets; condition key is used as sheet name
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(aux_file, sheet_name=sheet_name)
                result[sheet_name] = df
        except FileNotFoundError:
            print(f"Warning: File not found: {aux_file}")

    elif data_type == "eis":
        # ========== Load EIS Data ==========
        eis_file = data_dir / "eis.xlsx"
        try:
            excel = pd.ExcelFile(eis_file)

            # Parse EIS sheet names into condition keys that include the current point
            # Sheet name format: "I{current}_P{pressure}_T{temp}_HR{humidity}"
            condition_pattern = r"^I(\d+)_P(\d+)_T(\d+)_HR(\d+)"

            for sheet in excel.sheet_names:
                match = re.match(condition_pattern, sheet)
                if not match:
                    continue  # Skip non-data sheets (e.g. SYNTH, TRACES)

                current, p, t, hr = match.groups()
                # Convert raw values: HR50 -> 0.5, P300 -> 1.3, etc.
                rh_val = int(hr) / 100
                p_val = int(p) / 100
                # Build condition key: "I{current}_RHC{rh}_P{p}_T{t}"
                cond_key = f"I{int(current)}_RHC{rh_val}_P{p_val}_T{int(t)}"

                result[cond_key] = pd.read_excel(eis_file, sheet_name=sheet)

        except FileNotFoundError:
            print(f"Warning: File not found: {eis_file}")

    return result


if __name__ == "__main__":
    # Example usage
    try:
        # Load all polarization curves
        print("Loading polarization curve data...")
        pola_data = export_experiment_data("pola")
        print(f"Loaded {len(pola_data)} conditions")
        for cond, df in list(pola_data.items())[:2]:
            print(f"  {cond}: {df.shape}")

        # Load all HFR data
        print("\nLoading HFR data...")
        hfr_data = export_experiment_data("hfr")
        print(f"Loaded {len(hfr_data)} conditions")

        # Load all auxiliary data
        print("\nLoading auxiliary data...")
        aux_data = export_experiment_data("auxiliary")
        print(f"Loaded {len(aux_data)} conditions")

        # Load all EIS data
        print("\nLoading EIS data...")
        eis_data = export_experiment_data("eis")
        print(f"Loaded {len(eis_data)} conditions")
        for cond, spectra_list in list(eis_data.items())[:3]:
            print(f"  {cond}: {spectra_list.shape}")

    except Exception as e:
        print(f"Error: {e}")
