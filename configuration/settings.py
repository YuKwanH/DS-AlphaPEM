import numpy as np
from matplotlib.lines import Line2D
from ast import literal_eval
import re
# Physical constants
N_A = 6.022e23 # /mol
F = 96485  # C.mol-1. It is the Faraday constant.
R = 8.314  # J.mol-1.K-1. It is the universal gas constant.
M_O2 = 3.2e-2  # kg.mol-1. It is the molar mass of O2.
M_H2 = 2e-3  # kg.mol-1. It is the molar mass of H2.
M_N2 = 2.8e-2  # kg.mol-1. It is the molar mass of N2.
M_H2O = M_H2 + M_O2 / 2  # kg.mol-1. It is the molar mass of H2O.
gamma = 1.401  # . It is the heat capacity ratio of dry air at 100°C.
gamma_H2 = 1.404  # . It is the heat capacity ratio of H2 at 100°C.

# External environmental parameters
Text = 298  # K. It is the outside temperature.
Pext = 101325  # Pa. It is the outside pressure.
Phi_ext = 0.4  # It is the outside relative humidity.
yO2_ext = 0.2095  # . It is the molar fraction of O2 in dry air.

# Model parameters for the cell
rho_mem = 1980  # kg.m-3. It is the density of the dry membrane.
rho_cl = 3.87e2 # kg.m-3. It is the density of the catalyst layer.
rho_gdl = 2e3  # kg.m-3. It is the density of the gas diffusion layer.
M_eq = 2.1  # kg.mol-1. It is the equivalent molar mass of ionomer.
theta_c_gdl = 120 * np.pi / 180  # radian. It is the contact angle of GDL for liquid water.
theta_c_cl = 95 * np.pi / 180  # radian. It is the contact angle of CL for liquid water.
gamma_cond = 5e3  # s-1. It is the overall condensation rate constant for water.
gamma_evap = 1e-4  # Pa-1.s-1. It is the overall evaporation rate constant for water.
Kshape = 20  # . Mathematical factor governing lambda_eq smoothing.
R_H20 = 1.8e-5
R_iono = 5.56e-4

# Model parameters for the voltage calculation
C_O2ref = 3.39  # mol.m-3. It is the reference concentration of oxygen.
alpha_c = 0.5  # It is the transfer coefficient of the cathode.
E0 = 1.229  # V. It is the standard-state reversible voltage.
Pref = 1e5  # Pa. It is the reference pressure.
Eact = 73.2e3  # J.mol-1. It is the activation energy.

# Model parameters for the balance of plant
#   Physical parameters
n_cell = 22 # . It is the number of cell in the stack.
# Auxiliary system parameters
Vsm = 5e-6  # m3. It is the supply manifold volume.
Vem = 5e-6  # m3. It is the exhaust manifold volume.
A_T = 1.18e-3  # m². It is the exhaust manifold throttle area
tau_cp = 10  # s. It is the air compressor time constant.
tau_hum = 15  # s. It is the humidifier time constant.
Kp = 1e-8  # m².s-1.Pa-1. It is the proportional constant of the PD controller at the back pressure valve.
Kd = 2e-8  # m².s-1.Pa-1. It is the derivative constant of the PD controller at the back pressure valve.
C_D = 5e-2  # . It is the throttle discharge coefficient.
Ksm_in = 1.0e-5  # kg.s-1.Pa-1. It is the supply manifold inlet orifice constant.
Ksm_out = 1.0e-6  # kg.s-1.Pa-1. It is the supply manifold outlet orifice constant.
Kem_in = Ksm_out  # kg.s-1.Pa-1. It is the exhaust manifold inlet orifice constant.
Kem_out = Ksm_in  # kg.s-1.Pa-1. It is the exhaust manifold outlet orifice constant.


# Temperature dynamic
k_GDL = 6.5 # [W/(m*K)] thermal conductivity of GDL
k_CL = 0.27 # [W/(m*K)] thermal conductivity of CL
k_PEM = 21 # [W/(m*K)] thermal conductivity of PEM
Cp_cl = 7.7e2 # J/(kg*K) specific heat capacity of CL
Cp_gdl = 8.4e2 # J/(kg*K) specific heat capacity of GDL
Cp_mem = 1.1e3 # J/(kg*K) specific heat capacity of PEM
deltaS_OOR = -163.3 #  J/(mol*K)
deltaS_HOR = 0.104 # J/(mol*K)

# Display settings 
regions = ["agdl", "acl", "mem", "ccl", "cgdl"]
species = ["v", "O2", "H2", "s", "lambda"]



temp_colors = {50.0: "#5b2a86", 60.0: "#2a9d8f", 70.0: "#f1c40f"}
pressure_styles = {1.3: "-", 1.4: "--", 1.5: "-."}
humidity_markers = {"RHC0.0": "o", "RHC0.5": "s", "RHA0.0_RHC0.0": "o", "RHA0.0_RHC0.5": "s", "RHA0.5_RHC0.0": "^", "RHA0.5_RHC0.5": "D"}

def plot_condition(axis, x_values, y_values, label, linewidth=1.8, markersize=5):
    temperature_match = re.search(r"T(?P<value>\d+(?:\.\d+)?)", label)
    pressure_match = re.search(r"P(?P<value>\d+(?:\.\d+)?)", label)
    humidity_tokens = re.findall(r"RH[AC]\d+(?:\.\d+)?", label)
    temperature = float(temperature_match.group("value")) if temperature_match else None
    pressure = float(pressure_match.group("value")) if pressure_match else None
    humidity = float(humidity_tokens[0][2:]) if humidity_tokens else None
    if temperature is not None and temperature > 200:
        temperature = round(temperature - 273.15, 2)
    if pressure is not None and pressure > 20:
        pressure = round(pressure / 1000, 1) + 1
    if humidity is not None and humidity > 1:
        humidity = humidity/100
    pressure = float(pressure_match.group("value")) if pressure_match else None
    humidity = "_".join(humidity_tokens) if humidity_tokens else None
    axis.plot(x_values, y_values, color=temp_colors.get(temperature, "0.35"), linestyle=pressure_styles.get(pressure, "-"), marker=humidity_markers.get(humidity, "o"), linewidth=linewidth, markersize=markersize)