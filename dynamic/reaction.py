import numpy as np
from configuration.settings import *

def PtOxideDissolution(theta, Ch):
    """
    Developed by Heather A Baroody et al., "Predicting platinum dissolution and performance degradation under drive cycle operation of polymer electrolyte fuel cells"
    :return:
    """
    return k3 * theta * Ch ** 2


def PtDissolution(Ucell, T_fc, Cpt2, theta):
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
    Rf = np.exp((alpha_1 * F * n) / (R * T_fc) * (Ucell - Ueq_1))  # Forward
    Rb = Cpt2 / Cpt2_ref * np.exp((-(1 - alpha_1) * F * n) / (R * T_fc) * (Ucell - Ueq_1))  # Reverse
    return k1 * (1 - theta) * Rf - k1_ref * Rb


def PtOxidation(Ucell, T_fc, Ch, theta):
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

    Rf = (k2 * np.exp(-omega * theta / (R * T_fc)) *
          np.exp((alpha_2 * F * n) / (R * T_fc) * (Ucell - Ueq_2)))
    Rb = (k2_ref * theta * (Ch / Ch_ref) ** 2 *
          np.exp((-alpha_2 * F * n) / (R * T_fc) * (Ucell - Ueq_2)))

    return Rf - Rb


def PtDetachment(Ucell, T_fc, r):
    """

    :param Ucell:
    :param T_fc:
    :return:
    """
    n = 2
    return kdet_ref * Mcc / rho_cc * np.exp((0.5 * F * n) / (R * T_fc) * (Ucell - Ueq_4)) / r


def flourideReleaseRate(MT, U, Tfc, PO2_ca):
    """

    :return:
    """
    # Constant
    A_1 = 1.2e-12  # Fitted constant (gram h-1cm-2)
    alpha_eq = 0.53  # Equivalent transfer coefficient
    e_M0 = 2e-5  # The initial membrane thickness (m)
    E_a = 75e3  # The equivalent activation energy (J/mol)
    T0 = 273.15 + 95
    P0 = 1e5

    return A_1 * (PO2_ca/P0) * (e_M0 / MT) * np.exp(alpha_eq * F * U / (R * Tfc)) * np.exp(-E_a / R * (1 / Tfc - 1 / T0))
