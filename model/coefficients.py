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
R_H2O = 1.8e-5
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

# Temperature dynamic
k_GDL = 6.5 # [W/(m*K)] thermal conductivity of GDL
k_CL = 0.27 # [W/(m*K)] thermal conductivity of CL
k_PEM = 21 # [W/(m*K)] thermal conductivity of PEM
Cp_cl = 7.7e2 # J/(kg*K) specific heat capacity of CL
Cp_gdl = 8.4e2 # J/(kg*K) specific heat capacity of GDL
Cp_mem = 1.1e3 # J/(kg*K) specific heat capacity of PEM
deltaS_OOR = -163.3 #  J/(mol*K)
deltaS_HOR = 0.104 # J/(mol*K)

# Coefficients functions
C_v_sat = lambda T: Psat(T) / (R * T)  # saturated vapor concentration for a perfect gas
Da = lambda P, T: 1.644e-4 * (T / 298.15) ** 2.334 * (101325 / P)  # diffusion coefficient at the anode
Dc = lambda P, T: 3.242e-5 * (T / 298.15) ** 2.334 * (101325 / P)  # diffusion coefficient at the cathode
h_a = lambda P, T, Wgc, Hgc: (0.9247 * np.log(Wgc / Hgc) + 2.3787) * Da(P, T) / Hgc  # effective convective-conductive mass transfer coefficient at the anode
h_c = lambda P, T, Wgc, Hgc: (0.9247 * np.log(Wgc / Hgc) + 2.3787) * Dc(P, T) / Hgc  # effective convective-conductive mass transfer coefficient at the cathode
sigma_old = lambda T: 235.8e-3 * ((647.15 - T) / 647.15) ** 1.256 * (1 - 0.625 * (647.15 - T) / 647.15)  # water surface tension (old version)
K0_old = lambda epsilon, epsilon_c: epsilon / (8 * np.log(epsilon) ** 2) * (epsilon - 0.11) ** (0.785 + 2) * 4.6e-6 ** 2 / ((1 - 0.11) ** 0.785 * ((0.785 + 1) * epsilon - 0.11) ** 2) * np.exp(-3.60 * epsilon_c)  # intrinsic permeability (old version)
nu_l = lambda T: (2.414 * 10 ** (-5 + 247.8 / (T - 140.0))) / rho_H2O(T)  # liquid water kinematic viscosity
sigma = lambda T: 235.8e-3 * ((647.15 - T) / 647.15) ** 1.256 * (1 - 0.625 * (647.15 - T) / 647.15)  # water surface tension


def Psat(T):
    # saturated partial pressure of vapor
    return 101325 * 10 ** (-2.1794 + 0.02953 * (T - 273.15) - 9.1837e-5 * (T - 273.15) ** 2 + 1.4454e-7 * (T - 273.15) ** 3)


def Da_eff(s, epsilon, P, T, epsilon_c, epsilon_gdl):
    # effective diffusion coefficient at the anode considering GDL compression
    beta2 = -1.59 if epsilon_gdl < 0.67 else -0.90 if 0.67 <= epsilon_gdl < 0.8 else None
    return epsilon * ((epsilon - 0.11) / (1 - 0.11)) ** 0.785 * np.exp(beta2 * epsilon_c) * (1 - s) ** 2 * Da(P, T)


def Dc_eff(s, epsilon, P, T, epsilon_c, epsilon_gdl):
    # effective diffusion coefficient at the cathode considering GDL compression
    beta2 = -1.59 if epsilon_gdl < 0.67 else -0.90 if 0.67 <= epsilon_gdl < 0.8 else None
    return epsilon * ((epsilon - 0.11) / (1 - 0.11)) ** 0.785 * np.exp(beta2 * epsilon_c) * (1 - s) ** 2 * Dc(P, T)


def rho_H2O(T):
    # water density
    return ((999.83952 + 16.945176 * (T - 273.15) - 7.9870401e-3 * (T - 273.15) ** 2 - 46.170461e-6 * (T - 273.15) ** 3 + 105.56302e-9 * (T - 273.15) ** 4 - 280.54253e-12 * (T - 273.15) ** 5) / (1 + 16.879850e-3 * (T - 273.15)))


def lambda_eq(C_v, s, T, Kshape):
    # water content in the membrane
    a_w = C_v / C_v_sat(T) + 2 * s
    return 0.5 * (0.3 + 10.8 * a_w - 16.0 * a_w ** 2 + 14.1 * a_w ** 3) * (1 - np.tanh(100 * (a_w - 1))) + 0.5 * (9.2 + 8.6 * (1 - np.exp(-Kshape * (a_w - 1)))) * (1 + np.tanh(100 * (a_w - 1)))


def Dw(lambdaa, T):
    # diffusion coefficient of water in the membrane (alternative)
    lambdaa = np.asarray(lambdaa)
    Dw_low = 3.1 * 1e-7 * lambdaa * (np.exp(0.28 * lambdaa) - 1) * np.exp(-2436 / T)
    Dw_high = 4.17 * 1e-8 * lambdaa * (161 * np.exp(-lambdaa) + 1) * np.exp(-2436 / T)
    return np.where(lambdaa < 3, Dw_low, Dw_high)


def gamma_sorp(C_v, s, lambdaa, T, Hcl, Kshape):
    # phase transfer rate for water sorption/desorption
    fv = (lambdaa * M_H2O / rho_H2O(T)) / (M_eq / rho_mem + lambdaa * M_H2O / rho_H2O(T))
    if lambda_eq(C_v, s, T, Kshape) >= lambdaa:
        return (1.14e-5 * fv) / Hcl * np.exp(2416 * (1 / 303 - 1 / T))
    else:
        return (4.59e-5 * fv) / Hcl * np.exp(2416 * (1 / 303 - 1 / T))
    
    
def Svl(s, C_v, Ctot, epsilon, T, gamma_cond, gamma_evap):
    # phase transfer rate of water condensation or evaporation
    if C_v > C_v_sat(T):
        return gamma_cond * epsilon * (1 - s) * (C_v / Ctot) * (C_v - C_v_sat(T))
    else:
        return -gamma_evap * epsilon * s * rho_H2O(T) / M_H2O * R * T * (C_v_sat(T) - C_v)


def K0(epsilon, epsilon_c, epsilon_gdl):
    # intrinsic permeability considering GDL compression
    beta1 = -3.60 if epsilon_gdl < 0.67 else -2.60 if 0.67 <= epsilon_gdl < 0.8 else None
    return epsilon / (8 * np.log(epsilon) ** 2) * (epsilon - 0.11) ** (0.785 + 2) * 4.6e-6 ** 2 / ((1 - 0.11) ** 0.785 * ((0.785 + 1) * epsilon - 0.11) ** 2) * np.exp(beta1 * epsilon_c)


def k_H2(lambdaa, T, kappa_co):
    # permeability coefficient of the membrane for hydrogen
    E_H2_v, E_H2_l, Tref = 2.1e4, 1.8e4, 303.15; fv = (lambdaa * M_H2O / rho_H2O(T)) / (M_eq / rho_mem + lambdaa * M_H2O / rho_H2O(T))
    return kappa_co * (0.29 + 2.2 * fv) * 1e-14 * np.exp(E_H2_v / R * (1 / Tref - 1 / T)) if lambdaa < 17.6 else kappa_co * 1.8 * 1e-14 * np.exp(E_H2_l / R * (1 / Tref - 1 / T))


def k_O2(lambdaa, T, kappa_co):
    # permeability coefficient of the membrane for oxygen
    E_O2_v, E_O2_l, Tref = 2.2e4, 2.0e4, 303.15; fv = (lambdaa * M_H2O / rho_H2O(T)) / (M_eq / rho_mem + lambdaa * M_H2O / rho_H2O(T))
    return kappa_co * (0.11 + 1.9 * fv) * 1e-14 * np.exp(E_O2_v / R * (1 / Tref - 1 / T)) if lambdaa < 17.6 else kappa_co * 1.2 * 1e-14 * np.exp(E_O2_l / R * (1 / Tref - 1 / T))