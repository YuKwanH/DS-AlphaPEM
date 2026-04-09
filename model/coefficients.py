from configuration.settings import *


def Csat(T): # saturation concentration (mol/m^3)
    return 101325 * 10 ** (-2.1794 + 0.02953 * (T - 273.15) - 9.1837e-5 * (T - 273.15) ** 2 + 1.4454e-7 * (T - 273.15) ** 3) / (8.314 * T) 

def Dc(P, T):
    coef = 3.242e-5 * (T / 298.15) ** 2.334 * (101325 / P)
    D = coef * epsilon_gdl * ((epsilon_gdl - 0.11) / (1 - 0.11)) ** 0.785 * np.exp(-1.59 * epsilon_c) 
    return D

def h_conv(P, T, Wgc, Hgc):
    Sh = 0.9247 * np.log(Wgc / Hgc) + 2.3787  # Sherwood coefficient.
    return Sh * Dc(P, T) / Hgc

def rho_H2O(T):
    return ((999.83952 + 16.945176 * (T - 273.15) - 7.9870401e-3 * (T - 273.15) ** 2 - 46.170461e-6 * (T - 273.15) ** 3
                 + 105.56302e-9 * (T - 273.15) ** 4 - 280.54253e-12 * (T - 273.15) ** 5) /
                (1 + 16.879850e-3 * (T - 273.15)))

def sigma(T):
    return 235.8e-3 * ((647.15 - T) / 647.15) ** 1.256 * (1 - 0.625 * (647.15 - T) / 647.15)

def K0(epsilon, epsilon_c):
    return epsilon / (8 * np.log(epsilon) ** 2) * (epsilon - 0.11) ** (0.785 + 2) * 4.6e-6 ** 2 / ((1 - 0.11) ** 0.785 * ((0.785 + 1) * epsilon - 0.11) ** 2) * np.exp(-3.60 * epsilon_c)

def nu_l(T):
    mu_l = 2.414 * 10 ** (-5 + 247.8 / (T - 140.0))  # Pa.s. It is the liquid water dynamic viscosity.
    return mu_l / rho_H2O(T)

def lambda_eq(C_v, s, T, Kshape):
    a_w = C_v / Csat(T) + 2 * s  # water activity
    return 0.5 * (0.3 + 10.8 * a_w - 16.0 * a_w ** 2 + 14.1 * a_w ** 3) * (1 - np.tanh(100 * (a_w - 1))) \
               + 0.5 * (9.2 + 8.6 * (1 - np.exp(-Kshape * (a_w - 1)))) * (1 + np.tanh(100 * (a_w - 1)))

def Dw(lambdaa, T):
    lambdaa = np.asarray(lambdaa)
    Dw_low = 3.1 * 1e-7 * lambdaa * (np.exp(0.28 * lambdaa) - 1) * np.exp(-2436 / T)
    Dw_high = 4.17 * 1e-8 * lambdaa * (161 * np.exp(-lambdaa) + 1) * np.exp(-2436 / T)
    return np.where(lambdaa < 3, Dw_low, Dw_high)

def gamma_sorp(C_v, s, lambdaa, T, Hcl, Kshape):
    fv = (lambdaa * M_H2O / rho_H2O(T)) / (M_eq / rho_mem + lambdaa * M_H2O / rho_H2O(T))  
    if lambda_eq(C_v, s, T, Kshape) >= lambdaa:  # type_flow = absorption
        return (1.14e-5 * fv) / Hcl * np.exp(2416 * (1 / 303 - 1 / T))
    else:  # type_flow = desorption
        return (4.59e-5 * fv) / Hcl * np.exp(2416 * (1 / 303 - 1 / T))