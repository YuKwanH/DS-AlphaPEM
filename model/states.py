
# Importing constants' value and functions
from config.settings import *
from model.coefficients import *
from scipy import stats

# _____________________________________________________Cell voltage_____________________________________________________

def calculate_eta_c_intermediate_values(x, operating_inputs, parameters):
    """This function calculates the intermediate values needed for the calculation of the cathode overpotential dynamic
    evolution.

    Parameters
    ----------
    x : dict
        The dictionary containing the variables calculated by the solver.
    operating_inputs : dict
        The dictionary containing the operating inputs.
    parameters : dict
        The dictionary containing the parameters.

    Returns
    -------
    dict
        The dictionary containing the crossover current density i_n at time t, and the liquid water induced voltage drop
        function f_drop at time t.
    """

    # Extraction of the variables
    s_ccl = x['s_ccl']
    # Extraction of the operating inputs and the parameters
    Pc_des = operating_inputs['Pc_des']
    a_slim, b_slim, a_switch = parameters['a_slim'], parameters['b_slim'], parameters['a_switch']

    # The liquid water induced voltage drop function f_drop
    slim = a_slim * (Pc_des / 1e5) + b_slim
    s_switch = a_switch * slim
    f_drop = 0.5 * (1.0 - np.tanh((4 * s_ccl - 2 * slim - 2 * s_switch) / (slim - s_switch)))

    return {'f_drop': f_drop}

def calculate_cell_voltage_intermediate(variables, parameters):

    # Extraction of the operating inputs and the parameters
    Tccl, Tacl = variables['Tccl'], variables['Tacl']
    Hmem, Hcl, epsilon_mc, tau = parameters['Hmem'], parameters['Hcl'], parameters['epsilon_mc'], parameters['tau']
    lambda_ccl, lambda_acl = variables['lambda_ccl'], variables['lambda_acl']
    C_H2_acl, C_O2_ccl = variables["C_H2_acl"], variables["C_O2_ccl"]
    # Recovery of the already calculated variable values at each time step
    # The equilibrium potential
    Ueq = (E0 - 8.5e-4 * (Tccl - 298.15) + R * Tccl / (2 * F) * (np.log(R * Tccl * C_H2_acl / Pref) + 0.5 * np.log(R * Tccl * C_O2_ccl / Pref)))
    Rmem = []
    for i_mem in range(1, parameters['n_mem'] + 1):
        lambda_mem = variables["lambda_mem_" + str(i_mem)]
        Tmem = variables["Tmem_" + str(i_mem)]
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
    # 
    if lambda_acl >= 1:
        Racl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_acl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tacl)))
    else:
        Racl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tacl)))

    # The cell voltage
    return Ueq, Rmem, Rccl, Racl

def calculate_cell_voltage(ECSA, variables, operating_inputs, parameters):
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
    t, lambda_ccl_t = variables['t'],variables['lambda_ccl']
    C_H2_acl_t, C_O2_ccl_t, eta_c_t = variables['C_H2_acl'], variables['C_O2_ccl'], variables['eta_c']
    # Extraction of the operating inputs and the parameters
    Tccl_t = variables['Tccl']
    Hmem, Hcl, epsilon_mc, tau = parameters['Hmem'], parameters['Hcl'], parameters['epsilon_mc'], parameters['tau']
    Re, kappa_co = parameters['Re'], parameters['kappa_co']

    # Initialisation
    n = len(t)
    Ucell_t = [0] * n

    # Loop for having Ucell_t at each time step
    for i in range(n):

        # Recovery of the already calculated variable values at each time step
        lambda_ccl = lambda_ccl_t[i]
        C_H2_acl, C_O2_ccl = C_H2_acl_t[i], C_O2_ccl_t[i]
        eta_c = eta_c_t[i]
        Tccl = Tccl_t[i]

        # Current density value at this time step
        i_fc = operating_inputs['current_density'](t[i])
        # The equilibrium potential
        Ueq = (E0 - 8.5e-4 * (Tccl - 298.15) + R * Tccl / (2 * F) * (np.log(R * Tccl * C_H2_acl / Pref) + 0.5 * np.log(R * Tccl * C_O2_ccl / Pref)))

        # The proton resistance
        # The proton resistance at the membrane: Rmem
        Rmem = []
        for i_mem in range(1, parameters['n_mem'] + 1):
            lambda_mem_t = variables["lambda_mem_" + str(i_mem)]
            lambda_mem = lambda_mem_t[i]
            Tmem = variables["Tmem_" + str(i_mem)][i]
            if lambda_mem >= 1:
                Rmem += [(Hmem/parameters['n_mem']) / ((0.5139 * lambda_mem - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]
            else:
                Rmem += [(Hmem/parameters['n_mem']) / (0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]

        #  The proton resistance at the cathode catalyst layer: Rccl
        if lambda_ccl >= 1:
            Rccl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_ccl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tccl)))
        else:
            Rccl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tccl)))
        #       The total proton resistance
        Rp = sum(Rmem) # + Rccl  # its value is around [4-7]e-6 ohm.m².

        # The cell voltage OCV = 0.98 according to experimental data
        Ucell_t[i] = Ueq - (i_fc) * (Rp + Re) - eta_c
    return Ucell_t

# ____________________________________________Differential equations modules____________________________________________
def dif_eq_int_values(t, x, iload, control_variables, operating_inputs, parameters, I_aux_min = 0):

    # Extraction of the variables
    C_v_agc, C_v_cgc = x['C_v_agc'], x['C_v_cgc']
    C_H2_agc,C_O2_cgc, C_N2 = x['C_H2_agc'], x['C_O2_cgc'], x['C_N2']   
    Paem, Pcsm, Pasm, Pcem = x['Paem'], x['Pcsm'], x['Pasm'], x['Pcem']
    Tfc, Phi_c_des, Phi_a_des = operating_inputs['Tfc'], control_variables['Phi_c_des'], control_variables['Phi_a_des']
    Sa, Sc = operating_inputs['Sa'], operating_inputs['Sc']
    Wcp = x["Wcp"]
    Aact = parameters['Aact']
    # Physical quantities
    # Pressures
    Pagc = (C_v_agc + C_H2_agc) * R * Tfc
    Pcgc = (C_v_cgc + C_O2_cgc + C_N2) * R * Tfc
    Prd = Pasm
    Pcp = Pcsm
    # Humidities
    Phi_agc = C_v_agc / C_v_sat(Tfc)
    Phi_cgc = C_v_cgc / C_v_sat(Tfc)
    Phi_aem = Phi_agc * Paem / Pagc
    Phi_asm = Phi_a_des 
    Phi_csm = Phi_c_des 
    Phi_cem = Phi_cgc * Pcem / Pcgc
    # Oxygen ratio in dry air
    y_cgc = C_O2_cgc / (C_O2_cgc + C_N2)    
    # Molar masses
    Masm = Phi_asm * Psat(Tfc) / Pasm * M_H2O + (1 - Phi_asm * Psat(Tfc) / Pasm) * M_H2
    Maem = Phi_aem * Psat(Tfc) / Paem * M_H2O + (1 - Phi_aem * Psat(Tfc) / Paem) * M_H2
    Mcsm = Phi_csm * Psat(Tfc) / Pcsm * M_H2O + yO2_ext * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_O2 + (1 - yO2_ext) * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_N2
    Mcem = Phi_cem * Psat(Tfc) / Pcem * M_H2O + y_cgc * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_O2 + (1 - y_cgc) * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_N2
    Mext = Phi_ext * Psat(Text) / Pext * M_H2O + yO2_ext * (1 - Phi_ext * Psat(Text) / Pext) * M_O2 +  (1 - yO2_ext) * (1 - Phi_ext * Psat(Text) / Pext) * M_N2
    Magc = C_v_agc * R * Tfc / Pagc * M_H2O +  C_H2_agc * R * Tfc / Pagc * M_H2
    Mcgc = Phi_cgc * Psat(Tfc) / Pcgc * M_H2O +  y_cgc * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_O2 + (1 - y_cgc) * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_N2
    # Physical quantities in the auxiliary system
    # Pressure ratios
    Pr_aem = (Pext / Paem)
    Pr_cem = (Pext / Pcem)
    # Oxygen ratio in dry air
    y_cem = (Pcem - Phi_cem * Psat(Tfc) - C_N2 * R * Tfc) / (Pcem - Phi_cem * Psat(Tfc))
    # Molar masses
    Maem = Phi_aem * Psat(Tfc) / Paem * M_H2O + (1 - Phi_aem * Psat(Tfc) / Paem) * M_H2
    Masm = Phi_asm * Psat(Tfc) / Pasm * M_H2O + (1 - Phi_asm * Psat(Tfc) / Pasm) * M_H2
    Mcem = Phi_cem * Psat(Tfc) / Pcem * M_H2O + y_cem * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_O2 + (1 - y_cem) * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_N2
    Mcsm = Phi_csm * Psat(Tfc) / Pcsm * M_H2O + yO2_ext * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_O2 + (1 - yO2_ext) * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_N2
    
    # Setpoints 
    # The desired air compressor flow rate Wcp_des (kg.s-1)
    Wcp_des = n_cell * Mext * Pext / (Pext - Phi_ext * Psat(Text)) *  1 / yO2_ext * Sc * (iload) / (4 * F) * Aact
    Wrd = n_cell * M_H2 * Sa * (iload) / (2 * F) * Aact
    Wa_inj_des = M_H2O * Phi_a_des * Psat(Tfc) / Prd * (Wrd / M_H2)
    # The desired humidifier flow rate at the cathode side Wc_inj_des (kg.s-1)
    Wv_hum_in = M_H2O * Phi_ext * Psat(Text) / Pext * (Wcp / Mext)  # Vapor flow rate from the outside
    Wc_v_des = M_H2O * Phi_c_des * Psat(Tfc) / Pcp * (Wcp / Mext)  # Desired vapor flow rate
    Wc_inj_des = Wc_v_des - Wv_hum_in  # Desired humidifier flow rate

    return {"Pagc": Pagc, "Pcgc": Pcgc, "Prd": Prd, "Pcp": Pcp, "Pr_aem": Pr_aem, "Pr_cem": Pr_cem,
                 "Mext": Mext, "Masm": Masm, "Maem": Maem, "Mcsm": Mcsm, "Mcem": Mcem, "Magc": Magc, "Mcgc": Mcgc,
                 "Phi_agc": Phi_agc, "Phi_cgc": Phi_cgc, "Phi_aem": Phi_aem, "Phi_asm": Phi_asm, "Phi_csm": Phi_csm, "Phi_cem": Phi_cem,
                 "Wcp_des": Wcp_des, "Wa_inj_des": Wa_inj_des, "Wc_inj_des": Wc_inj_des,
                 "y_cgc": y_cgc, "y_cem": y_cem}


def Cproton_CCL(lambda_w, EW=1.1, rho_mem=0.002):
    """
    Concentration of proton based on the membrane water content(humidity)
    Darling et al. 2023
    :param EW: Equivalent weight of dry membrane.
    :param rho_mem: Membrane dry density
    :return: concentration of H+
    """
    if type(lambda_w) == float:
        rho_H2O = 997  # (kg/m3)
        M_H2O = 18.02e-3  # kg/mol
        return 1 / ((EW / rho_mem) + lambda_w * (M_H2O / rho_H2O))
    else:
        return 0


def initPRD(resolution=100, rmin=1e-8, rmax=1e-6, std=0.549, mu=0.538):
    """
    Initialization the particle radius distribution based on the normal distribution
    :param mu:
    :param std:
    :param rmax:
    :param rmin:
    :param resolution:
    :param dr: the resolution of the particle radius
    :return: the particle radius distribution as an 1D array
    """
    radius = np.linspace(rmin, rmax, resolution)
    pdf_values = stats.norm.pdf(np.log(radius * 1e7), mu, std)
    return pdf_values


def getECSA(prd, radius):
    return 4 * np.pi * np.trapezoid(y=(radius ** 2) * prd, x=radius)

# ______________________________________Function which gives the integration event______________________________________
def event_negative(t, y, operating_inputs, parameters, solver_variable_names, control_variables):
    """This function creates an event that will be checked at each step of solve_ivp integration. The integration stops
    if one of the crucial variables (C_v, lambda, C_O2, C_H2) becomes negative (or smaller than 1e-5).

    Parameters
    ----------
    t : float
        Time (s).
    y : numpy.ndarray
        Numpy list of the solver variables.
    operating_inputs : dict
        Operating inputs of the fuel cell.
    parameters : dict
        Parameters of the fuel cell model.
    solver_variable_names : list
        Names of the solver variables.
    control_variables : dict
        Variables controlled by the user.

    Returns
    -------
    The difference between the minimum value of the crucial variables and 1e-5.
    """

    negative_x = {} # Dictionary to store the crucial variables
    for index, key in enumerate(solver_variable_names):
        if (key.startswith("C_v_")) or (key.startswith("lambda_")) or \
                (key.startswith("C_O2_")) or (key.startswith("C_H2_")):
            negative_x[key] = y[index]
    return min(negative_x.values()) - 1e-5  # 1e-5 is a control param effect to stop the program before
    #                                                        having negative values.