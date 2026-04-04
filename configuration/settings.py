# Importing the necessary libraries
import numpy as np
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

# Model parameters for the voltage calculation
C_O2ref = 3.39  # mol.m-3. It is the reference concentration of oxygen.
alpha_c = 0.5  # It is the transfer coefficient of the cathode.
E0 = 1.229  # V. It is the standard-state reversible voltage.
Pref = 1e5  # Pa. It is the reference pressure.
Eact = 73.2e3  # J.mol-1. It is the activation energy.

# Model parameters for the balance of plant
#   Physical parameters
n_cell = 22 # . It is the number of cell in the stack.
Vsm = 5e-6  # m3. It is the supply manifold volume.
Vem = 5e-6  # m3. It is the exhaust manifold volume.
A_T = 1.18e-3  # m². It is the exhaust manifold throttle area
#   Model parameters
tau_cp = 3  # s. It is the air compressor time constant.
tau_hum = 5  # s. It is the humidifier time constant.
Kp = 1e-8  # m².s-1.Pa-1. It is the proportional constant of the PD controller at the back pressure valve.
Kd = 2e-8  # m².s-1.Pa-1. It is the derivative constant of the PD controller at the back pressure valve.
C_D = 5e-2  # . It is the throttle discharge coefficient.
Ksm_in = 1.0e-5  # kg.s-1.Pa-1. It is the supply manifold inlet orifice constant.
Ksm_out = 1.0e-6  # kg.s-1.Pa-1. It is the supply manifold outlet orifice constant.
Kem_in = Ksm_out  # kg.s-1.Pa-1. It is the exhaust manifold inlet orifice constant.
Kem_out = Ksm_in  # kg.s-1.Pa-1. It is the exhaust manifold outlet orifice constant.

# Anode variables
C_v_agc = 0
C_v_agdl = 0
C_v_acl = 0
s_acl = 0
lambda_acl = 0
C_H2_agc = 0
C_H2_acl = 0
Paem = 0  # exhaust manifold
Pasm = 0  # supply manifold
Phi_asm = 0
Phi_aem = 0
Wa_inj = 0  # flow rate of the air compressor at the anode side
Adp_a = 0  # the throttle area of the back pressure valve at the anode

# Cathode variables
C_v_cgc = 0
C_v_cgdl = 0
C_v_ccl = 0
s_ccl = 0
lambda_ccl = 0
C_O2_ccl = 0
C_O2_cgc = 0
C_N2 = 0
Pcem = 0  # exhaust manifold
Pcsm = 0  # supply manifold
Phi_csm = 0
Phi_cem = 0
Wc_inj = 0  # flow rate of the air compressor at the anode side
Adp_c = 0  # the throttle area of the back pressure valve at the anode


# Chemical constant
k1 = 3e-9
k1_ref = 1e-18
k2 = 1e-13
k2_ref = 1e-13
k3 = 1e-15
krdp = 1e-10
k4 = 0
k5 = 0
kdet_ref = 0#1.3e-22
rho_cc = 2.26
Mcc = 12.01
Ueq_4 = 0.2
Vm_Pt = 9.09  # Molar volume of Pt cm3/mol
M_Pt = 195.0849  # Molar mass of platinum (g/mol)
R0 = 0.2e-7

# Dynamic parameters
Cpt2_ref = 1e-3
rho_Pt = 21.45  # Density of platinum g/cm^3
GAMMA_max = 2.18e-9 # GAMMA(strong assumption): The active site quantity in moles per platinum area (mole/cm^2)