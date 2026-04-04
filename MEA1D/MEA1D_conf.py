import matplotlib.pyplot as plt
import numpy as  np
from scipy.integrate import solve_bvp


# CONSTANTS
M_w = 18e-3 # [kg/mol] molar mass of water
T_0 = 273.15 # [K] zero degrees Celsius
R = 8.31446 # [J*mol/K] universal gas constant
F = 96485.333 # [C/mol] Faraday constant
P_ref = 101325 # [Pa] reference pressure
T_ref = T_0+80 # [K] reference temperature
Nd = 5

# OPERATING CONDITIONS
U = np.arange(1.15,0,-0.05) # [V] list of cell voltages
P_A = 1.5e5 # [Pa] total pressure in anode gas channel
P_C = 1.5e5 # [Pa] total pressure in cathode gas channel
RH_A = 0.90 # [-] relative humidity in anode gas channel
RH_C = 0.90 # [-] relative humidity in cathode gas channel
s_C = 0.12 # [-] liquid water saturation at cathode GDL/GC interface
T_A = T_0+70 # [K] temperature of anode bipolar plate and gas channel
T_C = T_0+70 # [K] temperature of cathode bipolar plate and gas channel
alpha_H2 = 1 # [-] mole fraction of hydrogen in dry fuel gas
alpha_O2 = 0.21 # [-] mole fraction of oxygen in dry oxidant gas

# ELECTROCHEMICAL PARAMETERS
beta_HOR = 0.5 # [-] HOR symmetry factor
beta_ORR = 0.5 # [-] ORR symmetry factor
DeltaH = -285.83e3 # [J/mol] enthalpy of formation of liquid water
DeltaS_HOR = 0.104 # [J/(mol*K)] reaction entropy of HOR
DeltaS_ORR = -163.3 # [J/(mol*K)] reaction entropy of ORR

# MATERIAL PARAMETERS
L = np.array([160, 10, 25, 10, 160])*1e-6 # [m] MEA layer thicknesses
a_ACL = 1e7 # [1/m] ECSA density of ACL
a_CCL = 3e7 # [1/m] ECSA density of CCL
H_ec = 42e3 # [J/mol] molar enthalphy of evaporation/condensation
H_ad = H_ec # [J/mol] molar enthalphy of absorption/desorption
k_GDL = 1.6 # [W/(m*K)] thermal conductivity of GDL
k_CL = 0.27 # [W/(m*K)] thermal conductivity of CL
k_PEM = 0.3 # [W/(m*K)] thermal conductivity of PEM
s_im = s_C # [-] immobile liquid water saturation
V_m = 1.02/1.97e3 # [m^3/mol] molar volume of dry membrane (equivalent weight divided by mass density)
V_w = M_w/0.978e3 # [m^3/mol] molar volume of liquid water (molar mass divided by mass density)
eps_i_CL = 0.3 # [-] volume fraction of ionomer in dry CL
eps_p_GDL = 0.76 # [-] porosity of GDL
eps_p_CL = 0.4 # [-] porosity of CL
kappa_GDL = 6.15e-12 # [m^2] absolute permeability of GDL
kappa_CL = 1e-13 # [m^2] absolute permeability of CL
sigma_e_GDL = 1250 # [S/m] electrical conductivity of GDL
sigma_e_CL = 350 # [S/m] electrical conductivity of CL
tau_GDL = 1.6 # [-] pore tortuosity of GDL
tau_CL = 1.6 # [-] pore tortuosity of CL

def P_sat(T):
    return np.exp(23.1963-3816.44/(T-46.13))

x_H2O_A = RH_A*P_sat(T_A)/P_A #[-] mole fraction of water vapor in anode gas channel
x_H2O_C = RH_C*P_sat(T_C)/P_C #[-] mole fraction of water vapor in cathode gas channel
x_H2_A = alpha_H2*(1-x_H2O_A) #[-] mole fraction of hydrogen in anode gas channel
x_O2_C = alpha_O2*(1-x_H2O_C) #[-] mole fraction of oxygen in cathode gas channel

def D(eps_p,tau,s,P,T):
    return eps_p/tau**2*(1-s)**3*(T/T_ref)**1.5*(P_ref/P)
def D_H2(eps_p,tau,s,T):
    return 1.24e-4*D(eps_p,tau,s,P_A,T)
def D_O2(eps_p,tau,s,T):
    return 0.28e-4*D(eps_p,tau,s,P_C,T)
def D_H2O_A(eps_p,tau,s,T):
    return 1.24e-4*D(eps_p,tau,s,P_A,T)
def D_H2O_C(eps_p,tau,s,T):
    return 0.36e-4*D(eps_p,tau,s,P_C,T)
def A(E,T):
    return np.exp(E/R*(1/T_ref-1/T)) # [-] Arrhenius correction
def i_0_HOR(T):
    return 0.27e4*A(16e3,T) # [A/m^2] exchange current density of HOR
def i_0_ORR(x_O2, T):
    return 2.47e-4*(x_O2*P_C/P_ref)**0.54*A(67e3,T) # [A/m^2] exchange current density of ORR
def iff(condition, true_val, false_val):
    return np.where(condition, true_val, false_val)
def f(lambda_):
    return lambda_*V_w/(V_m+lambda_*V_w)
def mu(T):
    return 1e-3*np.exp(-3.63148+542.05/(T-144.15))

def BV(i_0,a,T,beta,eta):
    return i_0*a*(np.exp(beta*2*F/(R*T)*eta)-np.exp(-(1-beta)*2*F/(R*T)*eta))
# 1. Vapor sorption mass transfer coefficient [m/s]
def k_ad(lambda_, lambda_eq, T):
    return iff(lambda_ < lambda_eq, 3.53e-5, 1.42e-4) * f(lambda_) * A(20e3, T)
# 2. Reduced saturation
def s_red(s):
    return (s - s_im) / (1 - s_im)
# 3. Evaporation/condensation rate [1/s]
def gamma_ec(x_H2O, x_sat, s, T):
    return 2e6 * iff(x_H2O < x_sat, 5e-4 * s_red(s), 6e-3 * (1 - s_red(s))) * np.sqrt(R * T / (2 * np.pi * M_w))
# 4. Proton conductivity of Nafion [S/m]
def sigma_p(eps_i, lambda_, T):
    return eps_i**1.5 * 116 * np.maximum(0, f(lambda_) - 0.06)**1.5 * A(15e3, T)
# 5. Sorption isotherm [-]
def sorption(RH):
    return 0.043 + 17.81 * RH - 39.85 * RH**2 + 36.0 * RH**3
# 6. Diffusion coefficient of water in Nafion [m^2/s]
def D_lambda(eps_i, lambda_, T):
    numerator = 3.842 * lambda_**3 - 32.03 * lambda_**2 + 67.74 * lambda_
    denominator = lambda_**3 - 2.115 * lambda_**2 - 33.013 * lambda_ + 103.37
    return eps_i**1.5 * (numerator / denominator) * 1e-10 * A(20e3, T)
# 7. Electro-osmotic drag coefficient [-]
def xi(lambda_):
    return 2.5 * lambda_ / 22
# 8. Derivative of capillary pressure-saturation curve [Pa]
def dpds(s):
    return 0.00011 * 44.02 * np.exp(-44.02 * (s - 0.496)) + 278.3 * 8.103 * np.exp(8.103 * (s - 0.496))
# 9. Liquid water transport coefficient [m^2/s]
def D_s(kappa, s, T):
    return kappa * (1e-6 + s_red(s)**3) / mu(T) * dpds(s)