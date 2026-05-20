import numpy as np
from model.coefficients import *

# _____________________________________________________Cell voltage_____________________________________________________

def fdrop(x, operating_inputs, parameters):
    """
    x : dict
        The dictionary containing the variables calculated by the solver.
    operating_inputs : dict
        The dictionary containing the operating inputs.
    parameters : dict
        The dictionary containing the parameters.
    Returns
    -------
    float
        The liquid water induced voltage drop function f_drop at time t.
    """

    # Extraction of the variables
    s_ccl = x['s_cgdl_1']
    # Extraction of the operating inputs and the parameters
    Pc_des = operating_inputs['Pc_des']
    a_slim, b_slim, a_switch = parameters['a_slim'], parameters['b_slim'], parameters['a_switch']
    # The liquid water induced voltage drop function f_drop
    slim = a_slim * (Pc_des / 1e5) + b_slim
    s_switch = a_switch * slim

    return 0.5 * (1.0 - np.tanh((4 * s_ccl - 2 * slim - 2 * s_switch) / (slim - s_switch)))


def Rproton(variables, parameters, op):

    # Extraction of the operating inputs and the parameters
    Tccl, Tacl = op['Tfc'], op['Tfc']
    Hmem, Hcl, epsilon_mc, tau = parameters['Hmem'], parameters['Hcl'], parameters['epsilon_mc'], parameters['tau']
    lambda_ccl, lambda_acl = variables['lambda_ccl'], variables['lambda_acl']

    # Recovery of the already calculated variable values at each time step
    Rmem = []
    for i_mem in range(1, parameters['n_mem'] + 1):
        lambda_mem = variables["lambda_mem_" + str(i_mem)]
        Tmem = op['Tfc']
        # The proton resistance
        # The proton resistance at the membrane: Rmem
        if lambda_mem >= 1:
            Rmem += [(Hmem/parameters['n_mem']) / ((0.5139 * lambda_mem - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]
        else:
            Rmem += [(Hmem/parameters['n_mem']) / (0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]

    #  The proton resistance at the cathode catalyst layer : Rccl
    if lambda_ccl >= 1:
        Rccl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_ccl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tccl)))
    else:
        Rccl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tccl)))

    # The proton resistance at the anode catalyst layer : Racl
    if lambda_acl >= 1:
        Racl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_acl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tacl)))
    else:
        Racl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tacl)))

    return Rmem, Rccl, Racl


def Ueq(variables, op):
    # Extraction of the variables
    Tccl = op['Tfc']
    C_H2_acl, C_O2_ccl = variables['C_H2_acl'], variables['C_O2_ccl']
    return (E0 - 8.5e-4 * (Tccl - 298.15) + R * Tccl / (2 * F) * (np.log(R * Tccl * C_H2_acl / Pref) + 0.5 * np.log(R * Tccl * C_O2_ccl / Pref)))


def Ucell(t, variables, operating_inputs, parameters):
    """This function calculates the cell voltage at each time step.

    Parameters
    ----------
    variables : dict
        The dictionary containing the variables calculated by the solver.
    operating_inputs : dict
        The dictionary containing the operating inputs.
    parameters : dict
        The dictionary containing the parameters.

    Returns
    -------
    Ucell_t : list
        The cell voltage at each time step.
    """

    # Extraction of the variables
    Tccl, C_O2_ccl = operating_inputs["Tfc"], variables['C_O2_ccl']
    i_fc = operating_inputs['current_density'](t)
    f_drop = fdrop(variables, operating_inputs, parameters)
    Re, i0_c_ref, kappa_c = parameters['Re'], parameters['i0_c_ref'], parameters['kappa_c']
    OCV = parameters["OCV"]
    # Extraction of the operating inputs and the parameters
    #Ueq_t = Ueq(variables)
    i_fc_t = operating_inputs['current_density'](t)
    eta_c = (1 / f_drop * R * Tccl / (alpha_c * F) * np.log((i_fc) / i0_c_ref * (C_O2ref / C_O2_ccl) ** kappa_c) * np.exp(Eact / R * (1 / 353 - 1 / Tccl)))
    Rmem_t, Rccl_t, Racl_t = Rproton(variables, parameters , operating_inputs)
    Rp = sum(Rmem_t)  #+ Rccl_t + Racl_t
        # The cell voltage OCV = 0.98 according to experimental data
    Ucell_t = OCV - (i_fc_t) * (Rp + Re) - eta_c
    
    return Ucell_t

def eta_ccl(variables, operating_inputs, parameters):

    # Extraction of the variables
    Tccl, C_O2_ccl = operating_inputs['Tfc'], variables['C_O2_ccl']
    i_fc = operating_inputs['current_density'](0)
    f_drop = fdrop(variables, operating_inputs, parameters)
    i0_c_ref, kappa_c = parameters['i0_c_ref'], parameters['kappa_c']
    # Extraction of the operating inputs and the parameters
    eta_c_t = (1 / f_drop * R * Tccl / (alpha_c * F) * np.log((i_fc) / i0_c_ref * (C_O2ref / C_O2_ccl) ** kappa_c) * np.exp(Eact / R * (1 / 353 - 1 / Tccl)))
    
    return eta_c_t


def PtOxideDissolution(theta, Ch):
    """
    Developed by Heather A Baroody et al., "Predicting platinum dissolution and performance degradation under drive cycle operation of polymer electrolyte fuel cells"
    :return:
    """
    return k3 * theta * Ch ** 2


def PtDissolution(Ucell, T, Cpt2, theta):
    """
    Rate of platinum dissolution cm/s
    Initially developed by Darling and Meyers: "Kinetic model of platinum dissolution in PEMFCs"
    :param U_fc: cell voltage applied to the CL
    :return:
    """
    alpha_1 = 0.5
    Ueq_1 = 1.15  # Standard equilibrium potential Ueq_1
    n = 2  # Electron transferred
    # Modelled as Butler-Volmer equation
    Rf = np.exp((alpha_1 * F * n) / (R * T) * (Ucell - Ueq_1))  # Forward
    Rb = Cpt2 / Cpt2_ref * np.exp((-(1 - alpha_1) * F * n) / (R * T) * (Ucell - Ueq_1))  # Reverse
    return k1 * (1 - theta) * Rf - k1_ref * Rb


def PtOxidation(Ucell, T, Ch, theta):
    """
    rate of platinum oxidation
    Initially developed by Harrington then enhanced by Darling "Kinetic model of platinum dissolution in PEMFCs"
    :param Ch:
    :param U_fc: cell voltage applied to the CL
    :return:
    """

    # Standard equilibrium potential Ueq_2
    Ueq_2 = 0.97
    n = 2
    alpha_2 = 0.5
    omega = 27e3
    Ch_ref = 1e-3

    Rf = (k2 * np.exp(-omega * theta / (R * T)) *
          np.exp((alpha_2 * F * n) / (R * T) * (Ucell - Ueq_2)))
    Rb = (k2_ref * theta * (Ch / Ch_ref) ** 2 *
          np.exp((-alpha_2 * F * n) / (R * T) * (Ucell - Ueq_2)))

    return Rf - Rb


def PtDetachment(Ucell, T, r):
    """

    :param Ucell:
    :param T: Temperature
    :param r: Radius
    :return:
    """
    n = 2
    return kdet_ref * Mcc / rho_cc * np.exp((0.5 * F * n) / (R * T) * (Ucell - Ueq_4)) / r


def flourideReleaseRate(MT, U, Tmem, PO2_ca):
    """

    :return:
    """
    # Constant
    A_1 = 1.2e-14  # Fitted constant (gram h-1cm-2)
    alpha_eq = 0.53  # Equivalent transfer coefficient
    e_M0 = 2e-5  # The initial membrane thickness (m)
    E_a = 75e3  # The equivalent activation energy (J/mol)
    T0 = 273.15 + 95
    P0 = 1e5

    return A_1 * (PO2_ca/P0) * (e_M0 / MT) * np.exp(alpha_eq * F * U / (R * Tmem)) * np.exp(-E_a / R * (1 / Tmem - 1 / T0))
