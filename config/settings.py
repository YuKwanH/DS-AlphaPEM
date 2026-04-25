import numpy as np
from matplotlib.lines import Line2D
from ast import literal_eval
import re


# Display settings 
regions = ["agdl", "acl", "mem", "ccl", "cgdl"]
species = ["v", "O2", "H2", "s", "lambda"]

temp_colors = {50.0: "#5b2a86", 60.0: "#2a9d8f", 70.0: "#f1c40f"}
pressure_styles = {1.3: "-", 1.4: "--", 1.5: "-."}
humidity_markers = {"RHC0.0": "o", "RHC0.5": "x", "RHA0.0_RHC0.0": "o", "RHA0.0_RHC0.5": "x", "RHA0.5_RHC0.0": "^", "RHA0.5_RHC0.5": "D"}

def plot_condition(axis, x_values, y_values, label, linewidth=1.8, markersize=5):
    temperature_match = re.search(r"T(?P<value>\d+(?:\.\d+)?)", label)
    pressure_match = re.search(r"P(?P<value>\d+(?:\.\d+)?)", label)
    humidity_matches = re.findall(r"(?:RH|HR)([AC])(?P<value>\d+(?:\.\d+)?)", label)
    temperature = float(temperature_match.group("value")) if temperature_match else None
    pressure = float(pressure_match.group("value")) if pressure_match else None
    if temperature is not None and temperature > 200:
        temperature = round(temperature - 273.15, 2)
    if pressure is not None and pressure > 20:
        pressure = round(pressure / 1000, 1) + 1
    humidity_parts = []
    for side, value_text in humidity_matches:
        value = float(value_text)
        if value > 1:
            value = value / 100
        humidity_parts.append(f"RH{side}{value:.1f}")
    humidity = "_".join(humidity_parts) if humidity_parts else None
    axis.plot(x_values, y_values, color=temp_colors.get(temperature, "0.35"), linestyle=pressure_styles.get(pressure, "-"), marker=humidity_markers.get(humidity, "o"), linewidth=linewidth, markersize=markersize)