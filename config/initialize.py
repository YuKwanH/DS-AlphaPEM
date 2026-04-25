from config.settings import *
from model.coefficients import *
from model.states import *

"""
Computing parameters
"""
type_fuel_cell = "LEV-200"
type_control = "Phi_des"
type_purge = "constant_purge"

"""
Operating inputs and parameters
"""
operating_inputs = {'current_density': lambda x: 0.1e4, 'Tfc': 343.15, 
                                    'Pa_des': 1.5e5, 'Pc_des': 1.5e5,
                                    'Phi_a_des': 0.0, 'Phi_c_des': 0.5,
                                    'Sa': 1.2, 'Sc': 2.5,}

# Computing parameters
max_step = 0.1
n_gdl = 10
t_purge = 2.4, 15

accessible_physical_parameters = {'Aact': 31e-4, 'Hmem': 1.2e-5, 'Hgc': 8e-4, 'Wgc': 4e-4, 'Lgc': 1.287} #m

current_parameters = {'t_step': (0, 3600, 100, 1.5), 'i_step': (0.5e4, 1.5e4),
                                        'delta_pola': (30, 30, 0.1e4, 60), 'i_max_pola': 1.65e4, # 50A/30e-4
                                        'i_EIS': 1.0e4, 'ratio_EIS': 0.05, 't_EIS': 15, 'f_EIS': (-3, 5, 90, 50)}

undetermined_physical_parameters = {'epsilon_gdl': 0.7, "epsilon_cl": 0.15,
                                                                'epsilon_mc': 0.399,'epsilon_c': 0.299, 
                                                                'e': 4, 'kappa_co': 37.2, 'Re': 2.2e-7, 'tau': 1.01, 
                                                                'i0_c_ref': 10.6, 'kappa_c': 0.1, 'C_scl': 1e8, 
                                                                'a_slim': 0.01, 'b_slim': 0.25, 'a_switch': 0.13,
                                                                "Hcl": 1e-5, "Hgdl": 2.e-4}

computing_parameters = {'max_step': max_step, 'n_gdl': 10,'n_mem':10,'n_group_pt':10,
                                            't_purge': t_purge, 'type_fuel_cell': type_fuel_cell, 'type_control': type_control, 'type_purge': type_purge}

parameters = {**current_parameters, **accessible_physical_parameters,
                          **undetermined_physical_parameters, **computing_parameters}

# Initial conditions
current_density, Tfc = operating_inputs['current_density'], operating_inputs['Tfc']
Pa_des, Pc_des = operating_inputs['Pa_des'], operating_inputs['Pc_des']
Phi_a_des, Phi_c_des = operating_inputs['Phi_a_des'], operating_inputs['Phi_c_des']
Hmem, kappa_co, i0_c_ref, = parameters['Hmem'], parameters['kappa_co'], parameters['i0_c_ref']
kappa_c = parameters['kappa_c']
a_slim, b_slim, a_switch = parameters['a_slim'], parameters['b_slim'], parameters['a_switch']
n_gdl,n_mem = parameters['n_gdl'], parameters['n_mem']

# Mean value of the operating inputs
Phi_des_moy = (Phi_a_des + Phi_c_des) / 2
P_des_moy = (Pa_des + Pc_des) / 2

# Initial fuel cell states
#   Intermediate values
Psat_ini = 101325 * 10 ** (-2.1794 + 0.02953 * (Tfc - 273.15) - 9.1837e-5 * (Tfc - 273.15) ** 2 + 1.4454e-7 * (Tfc - 273.15) ** 3)
slim = a_slim * (Pc_des / 1e5) + b_slim
s_switch = a_switch * slim

def kinetics(i, x, a_slim, b_slim, a_switch, kappa_c, i0_c_ref, operating_inputs):

    """ 
    This function is dedicated to calibrating the kinetic parameters for the cathode catalyst layer, 
    which are used in the calculation of eta_c. It returns the intermediate values needed for calculating eta_c, 
    including Ueq, f_drop, eta_c, Rohm, and Rccl.
    """
    slim = a_slim * (operating_inputs['Pc_des'] / 1e5) + b_slim
    s_switch = a_switch * slim
    i_fc = np.array(x["i_fc"], dtype=float)
    s_ccl = np.array(x["s_ccl"], dtype=float)
    Tccl = np.array(x["Tccl"], dtype=float)
    C_O2_ccl = np.array(x["C_O2_ccl"], dtype=float)
    C_H2_acl = np.array(x["C_H2_acl"], dtype=float)
    Ueq = (E0 - 8.5e-4 * (Tccl - 298.15) + R * Tccl / (2 * F) * (np.log(R * Tccl * C_H2_acl / Pref) + 0.5 * np.log(R * Tccl * C_O2_ccl / Pref)))
    f_drop = 0.5 * (1.0 - np.tanh((4 * s_ccl - 2 * slim - 2 * s_switch) / (slim - s_switch)))
    i0_c = i0_c_ref * np.exp(-Eact / R * (1 / Tccl - 1 / 353))
    eta_c = (1 / f_drop * R * Tccl / (alpha_c * F) * np.log((i_fc) / i0_c * (C_O2ref / C_O2_ccl) ** kappa_c))
    return Ueq, f_drop, eta_c

#   Initial fuel cell states
def init_x(operating_inputs, parameters):
    C_v_ini =  Psat(343.13) / (R * 343.13)  #*Phi_des_moy # mol.m-3. It is the initial vapor concentration.
    C_H2_ini = (P_des_moy - Phi_des_moy * Psat_ini) / (R * Tfc)  # mol.m-3. It is the initial H2 concentration
    C_O2_ini = yO2_ext * (P_des_moy - Phi_des_moy * Psat_ini) / (R * Tfc)  # mol.m-3. It is the initial O2 concentration in the fuel cell.
    C_N2_ini = (1 - yO2_ext) * (P_des_moy - Phi_des_moy * Psat_ini) / (R * Tfc)  # mol.m-3. It is the initial N2  concentration in the fuel cell.
    s_ini = 0.001  # It is the initial liquid water saturation in the fuel cell.
    lambda_mem_ini = lambda_eq(C_v_ini, s_ini, Tfc, Kshape)  # It is the initial water content in the fuel cell.
    i_fc_ini = current_density(0)
    f_drop_ini = 0.5 * (1.0 - np.tanh((4 * s_ini - 2 * slim - 2 * s_switch) / (slim - s_switch)))
    # It is the initial cathode overpotential in the fuel cell.
    eta_c_ini = (1 / f_drop_ini * R * Tfc / (alpha_c * F) * np.log((i_fc_ini) / i0_c_ref * (C_O2ref / C_O2_ini) ** kappa_c) *
                        np.exp(Eact / R * (1 / 353 - 1 / Tfc)))

    # Initial auxiliary system state
    Pasm_ini, Paem_ini = Pa_des, P_des_moy  # Pa. It is the supply/exhaust manifold pressure at the anode side.
    Pcsm_ini, Pcem_ini = Pc_des, P_des_moy  # Pa. It is the supply/exhaust manifold pressure at the cathode side.
    Phi_asm_ini, Phi_aem_ini = Phi_a_des, Phi_des_moy  # It is the supply/exhaust manifold relative humidity
    #     at the anode side.
    Phi_csm_ini, Phi_cem_ini = Phi_c_des, Phi_des_moy  # It is the supply/exhaust manifold relative humidity
    #     at the cathode side.
    Wcp_ini = 0  # kg.s-1. It is the flow rate of the air compressor.
    Wa_inj_ini = 0  # kg.s-1. 3 NL/min
    Wc_inj_ini = 0  # kg.s-1. It is the flow rate of the air compressor at the cathode side.
    Abp_a_ini = 0  # It is the throttle area of the back pressure valve at the anode.
    Abp_c_ini = 0  # It is the throttle area of the back pressure valve at the cathode.

    # Main variable initialization
    C_v_agc, C_v_agdl, C_v_acl, C_v_ccl, C_v_cgdl, C_v_cgc = [C_v_ini/5] * 6
    s_agdl_init = [0.0] * (n_gdl-1)
    s_cgdl_init = [0.01 - 0.01 * i / (n_gdl-1) for i in range(n_gdl-1)]
    s_acl_init = 0.0
    s_ccl_init = 0.01
    s_boundary = 0  # Dirichlet boundary con
    s_boundary = 0  # Dirichlet boundary condition
    C_Pt2_ccl = 0
    C_Pt_mem_init = [0] * (n_mem - 1) + [C_Pt2_ccl/2]
    C_H2_mem_init = [0] + [0] * (n_mem -1)
    C_O2_mem_init = [0] * (n_mem - 1) + [0]
    C_H2_agc, C_H2_agdl, C_H2_acl = C_H2_ini, C_H2_ini, C_H2_ini
    C_O2_ccl, C_O2_cgdl, C_O2_cgc = C_O2_ini, C_O2_ini, C_O2_ini
    C_N2, eta_c = C_N2_ini, eta_c_ini
    Pasm, Paem, Pcsm, Pcem = Pasm_ini, Paem_ini, Pcsm_ini, Pcem_ini
    Phi_asm, Phi_aem, Phi_csm, Phi_cem = Phi_asm_ini, Phi_aem_ini, Phi_csm_ini, Phi_cem_ini
    Wcp, Wa_inj, Wc_inj, Abp_a, Abp_c = Wcp_ini, Wa_inj_ini, Wc_inj_ini, Abp_a_ini, Abp_c_ini
    prd_init = initPRD(resolution=computing_parameters['n_group_pt'])
    theta_CCL = np.zeros(computing_parameters['n_group_pt'])

    # Gathering of the variables initial value into one list
    return  ([C_H2_agc] + [C_H2_agdl] * n_gdl + [C_H2_acl] + C_H2_mem_init +
                    C_O2_mem_init + [C_O2_ccl] + [C_O2_cgdl] * n_gdl + [C_O2_cgc] + [C_N2] +
                    [C_v_agc] + [C_v_agdl] * n_gdl + [C_v_acl, C_v_ccl+1] + [C_v_cgdl] * n_gdl + [C_v_cgc] +
                    [0] + s_agdl_init + [0, s_ccl_init] + s_cgdl_init + [s_boundary] +
                    [2] + [lambda_mem_ini] * n_mem + [lambda_mem_ini] +
                    [eta_c, Pasm, Paem, Pcsm, Pcem, Phi_asm, Phi_aem, Phi_csm, Phi_cem] +
                    [Wcp, Wa_inj, Wc_inj, Abp_a, Abp_c] + C_Pt_mem_init +
                    [C_Pt2_ccl, Hmem] + prd_init.tolist() + theta_CCL.tolist() +
                    [operating_inputs["Tfc"]] * 32)

# Displaying settings
# Display settings
Hgdl, Hcl, Hmem, n_gdl, n_mem = parameters['Hgdl'], parameters['Hcl'], parameters['Hmem'], parameters['n_gdl'], parameters['n_mem']
profile_1d = {}
profile_pola = {}

borders = [-2*Hgdl/n_gdl, 0, Hgdl, Hgdl + Hcl, Hgdl + Hcl + Hmem, Hgdl + Hcl*2 + Hmem, Hgdl*2 + Hcl*2 + Hmem, Hgdl*2.2 + Hcl*2 + Hmem]
nodes = [-Hgdl/n_gdl] + np.linspace(borders[1], borders[2], n_gdl).tolist() + [borders[2] + Hcl/2] + np.linspace(borders[3], borders[4], n_mem).tolist() + [borders[4] + Hcl/2] + np.linspace(borders[5], borders[6], n_gdl).tolist() + [borders[6]+Hgdl/n_gdl]
nodes_postfix = ["agdl_" + str(i+1) for i in range(n_gdl)] + ["acl"] + ["mem_" + str(i+1) for i in range(n_mem)] + ["ccl"] + ["cgdl_" + str(i+1) for i in range(n_gdl)]
nodes_names_vp = [f"C_v_agdl_{i+1}" for i in range(n_gdl)] + ["C_v_acl"] + ["C_v_ccl"] + [f"C_v_cgdl_{i+1}" for i in range(n_gdl)]
nodes_name_O2 = [f"C_O2_mem_{i+1}" for i in range(n_mem)] + ["C_O2_ccl"] + [f"C_O2_cgdl_{i+1}" for i in range(n_gdl)]
nodes_names_H2 = [f"C_H2_agdl_{i+1}" for i in range(n_gdl)] + ["C_H2_acl"] + [f"C_H2_mem_{i+1}" for i in range(n_mem)]
nodes_names_s = [f"s_agdl_{i+1}" for i in range(n_gdl)] + ["s_acl"] + ["s_ccl"] + [f"s_cgdl_{i+1}" for i in range(n_gdl)]
nodes_lambda = ["lambda_acl"] + [f"lambda_mem_{i+1}" for i in range(n_mem)] + ["lambda_ccl"]
nodes_T = ["Tagdl_" + str(i+1) for i in range(n_gdl)] + ["Tacl"] + ["Tmem_" + str(i+1) for i in range(n_mem)] + ["Tccl"] + ["Tcgdl_" + str(i+1) for i in range(n_gdl)]

def has_species_value(profile_key, postfix):
    if profile_key == "v":
        return postfix.startswith("agdl_") or postfix == "acl" or postfix == "ccl" or postfix.startswith("cgdl_")
    if profile_key == "O2":
        return postfix.startswith("mem_") or postfix == "ccl" or postfix.startswith("cgdl_")
    if profile_key == "H2":
        return postfix.startswith("agdl_") or postfix == "acl" or postfix.startswith("mem_")
    if profile_key == "saturation":
        return postfix.startswith("agdl_") or postfix == "acl" or postfix == "ccl" or postfix.startswith("cgdl_")
    if profile_key == "lambda":
        return postfix == "acl" or postfix.startswith("mem_") or postfix == "ccl"
    if profile_key == "T":
        return True
    return False

def expand_profile_on_nodes(profile_key, compact_values):
    expanded_values = []
    idx = 0
    for postfix in nodes_postfix:
        if has_species_value(profile_key, postfix):
            expanded_values.append(compact_values[idx])
            idx += 1
        else:
            expanded_values.append(0.0)
    return expanded_values



# Colormap: Temperature condition (50, 60, 70 °C)
temperature_values = [50, 60, 70]
colormap_temp = {
    50: '#1f77b4',   # blue
    60: '#ff7f0e',   # orange
    70: '#d62728'    # red
}

# Linemap: Pressure condition (PA = PC always equal)
pressure_values = [1.3, 1.4, 1.5]  # in bar (130, 140, 150 kPa)
linemap_pressure = {
    1.3: '-',      # solid
    1.4: '--',     # dashed
    1.5: ':'       # dotted
}

# Markermap: RH condition (0 and 0.5)
rh_values = [0, 0.5]
markermap_rh = {
    0: 'x',        # cross
    0.5: 'o'       # circle
}

# Function to get plotting properties from condition key
def get_plot_properties(cond_key):
    """
    Extract temperature, pressure, and RH from condition key string.
    Key format: "RHA0/RHC{RH}_P{P}_T{T}"
    Returns: dict with 'color', 'linestyle', 'marker', 'label'
    """
    import re
    pattern = r'RHA\d+/RHC([\d.]+)_P([\d.]+)_T(\d+)'
    match = re.match(pattern, cond_key)

    if not match:
        return {'color': 'black', 'linestyle': '-', 'marker': 'o', 'label': cond_key}

    rh = float(match.group(1))
    p = float(match.group(2))
    t = int(match.group(3))

    return {
        'color': colormap_temp.get(t, 'black'),
        'linestyle': linemap_pressure.get(p, '-'),
        'marker': markermap_rh.get(rh, 'o'),
        'label': cond_key
    }