import numpy as np

# Catalyst
Aact = 3.0e-3  # m². It is the active area of the catalyst layer.
Hcl = 1e-5  # m. It is the thickness of the anode or cathode catalyst layer.
epsilon_mc = 0.399  # It is the volume fraction of ionomer in the CL.
tau = 1.016  # It is the pore structure coefficient, without units.
alpha_c = 0.5
# Membrane
Hmem = 2e-5  # m. It is the thickness of the membrane.
# Gas diffusion layer
Hgdl = 2e-4  # m. It is the thickness of the gas diffusion layer.
epsilon_gdl = 0.701  # It is the anode/cathode GDL porosity.
epsilon_c = 0.271  # It is the compression ratio of the GDL.
# Gas channel
Hgc = 5e-4  # m. It is the thickness of the gas channel.
Wgc = 4.5e-4  # m. It is the width of the gas channel.
Lgc = 2.67  # m. It is the length of the gas channel.
# Interaction parameters between water and PEMFC structure
e = 5.0  # It is the capillary exponent
# Voltage polarization
Re = 5.70e-07  # ohm.m². It is the electron conduction resistance of the circuit.
i0_c_ref = 0.79  # A.m-2.It is the reference exchange current density at the cathode.
kappa_co = 27.2  # mol.m-1.s-1.Pa-1. It is the crossover correction coefficient.
kappa_c = 1.61  # It is the overpotential correction exponent.
a_slim, b_slim, a_switch = 0.05553, 0.10514, 0.63654  # It is the limit liquid saturation coefficients.
C_scl = 2e7
E0 = 1.229  # V. It is the standard-state reversible voltage.
Pref = 1e5  # Pa. It is the reference pressure.
Eact = 73.2e3  # J.mol-1. It is the activation energy.
C_O2ref = 3.39  # mol.m-3. It is the reference concentration of oxygen.

# Model parameters for the balance of plant
#   Physical parameters
n_cell = 22 # . It is the number of cell in the stack.
Vsm = 7.0e-3  # m3. It is the supply manifold volume.
Vem = 2.4e-3  # m-3. It is the exhaust manifold volume.
A_T = 1.18e-3  # m². It is the exhaust manifold throttle area
F = 96485  # C.mol-1. It is the Faraday constant.
R = 8.314  # J.mol-1.K-1. It is the universal gas constant.
M_O2 = 3.2e-2  # kg.mol-1. It is the molar mass of O2.
M_H2 = 2e-3  # kg.mol-1. It is the molar mass of H2.
M_N2 = 2.8e-2  # kg.mol-1. It is the molar mass of N2.
M_H2O = M_H2 + M_O2 / 2  # kg.mol-1. It is the molar mass of H2O.
gamma = 1.401  # . It is the heat capacity ratio of dry air at 100°C.
gamma_H2 = 1.404  # . It is the heat capacity ratio of H2 at 100°C.
#   Model parameters
tau_cp = 1  # s. It is the air compressor time constant.
tau_hum = 5  # s. It is the humidifier time constant.
Kp = 5e-8  # m².s-1.Pa-1. It is the proportional constant of the PD controller at the back pressure valve.
Kd = 1e-8  # m².Pa-1. It is the derivative constant of the PD controller at the back pressure valve.
C_D = 5e-2  # . It is the throttle discharge coefficient.
Ksm_in = 1.0e-5  # kg.s-1.Pa-1. It is the supply manifold inlet orifice constant.
Ksm_out = 8.0e-6  # kg.s-1.Pa-1. It is the supply manifold outlet orifice constant.
Kem_in = Ksm_out  # kg.s-1.Pa-1. It is the exhaust manifold inlet orifice constant.
Kem_out = Ksm_in  # kg.s-1.Pa-1. It is the exhaust manifold outlet orifice constant.
# Model parameters for the cell
rho_mem = 1980  # kg.m-3. It is the density of the dry membrane.
M_eq = 1.1  # kg.mol-1. It is the equivalent molar mass of ionomer.
epsilon_cl = 0.25  # It is the porosity of the catalyst layer, without units.
theta_c_gdl = 120 * np.pi / 180  # radian. It is the contact angle of GDL for liquid water.
theta_c_cl = 95 * np.pi / 180  # radian. It is the contact angle of CL for liquid water.
gamma_cond = 5e3  # s-1. It is the overall condensation rate constant for water.
gamma_evap = 1e-4  # Pa-1.s-1. It is the overall evaporation rate constant for water.
Kshape = 2  # . Mathematical factor governing lambda_eq smoothing.

# External environmental parameters
Text = 298  # K. It is the outside temperature.
Pext = 101325  # Pa. It is the outside pressure.
Phi_ext = 0.4  # It is the outside relative humidity.
yO2_ext = 0.2095  # . It is the molar fraction of O2 in dry air.




