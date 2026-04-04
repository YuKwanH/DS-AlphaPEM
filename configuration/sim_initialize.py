from configuration.settings import *
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
Tfc = 273.15 + 60 # K
Pa_des = 1.3e5 # Pa
Pc_des = 1.3e5 # Pa
Sa = 1.2 # Stoichiometric ratio of hydrogen
Sc = 2.5 # Stoichiometric ratio of oxygen
Phi_a_des = 0.0 # Desired anode RH
Phi_c_des = 0.5 # Desired cathode RH
i_max = 1.6e4 # Maximum current density in A/m²

# Computing parameters
max_step = 0.1
n_gdl = 10
t_purge = 2.4, 15
                                                                 
operating_inputs = {'current_density': lambda x: 0.1e4, 'Tfc': Tfc, 'Pa_des': Pa_des, 'Pc_des': Pc_des,
                                    'Sa': Sa, 'Sc': Sc, 'Phi_a_des': Phi_a_des, 'Phi_c_des': Phi_c_des}

accessible_physical_parameters = {'Aact': 30e-4, 'Hmem': 2.5e-5, 'Hgc': 8e-4, 'Wgc': 4e-4, 'Lgc': 1.287} #m

current_parameters = {'t_step': (0, 3600, 100, 1.5), 'i_step': (0.5e4, 1.5e4),
                                        'delta_pola': (30, 30, 0.1e4, 60), 'i_max_pola': i_max,
                                        'i_EIS': 1.0e4, 'ratio_EIS': 0.05, 't_EIS': 15, 'f_EIS': (-3, 5, 90, 50)}

undetermined_physical_parameters = {'epsilon_gdl': 0.6, "epsilon_cl": 0.15,
                                                                'epsilon_mc': 0.399,'epsilon_c': 0.299, 
                                                                'e': 5, 'kappa_co': 37.2, 'Re': 2.2e-7, 'tau': 1.01, 
                                                                'i0_c_ref': 10.6, 'kappa_c': 0.1, 'C_scl': 1e8, 
                                                                'a_slim': 0.2, 'b_slim': 0.3, 'a_switch': 0.3,
                                                                "Hcl": 1e-5, "Hgdl": 4.0e-4}

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

#   Initial fuel cell states
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
Wa_inj_ini = 0  # kg.s-1. It is the flow rate of the air compressor at the anode side.
Wc_inj_ini = 0  # kg.s-1. It is the flow rate of the air compressor at the cathode side.
Abp_a_ini = 0  # It is the throttle area of the back pressure valve at the anode.
Abp_c_ini = 0  # It is the throttle area of the back pressure valve at the cathode.

# Main variable initialization
C_v_agc, C_v_agdl, C_v_acl, C_v_ccl, C_v_cgdl, C_v_cgc = [C_v_ini/5] * 6
s_agdl, s_acl, s_ccl, s_cgdl = [s_ini] * 4
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
initial_variable_values =  ([C_H2_agc] + [C_H2_agdl] * n_gdl + [C_H2_acl] + C_H2_mem_init +
                                                C_O2_mem_init + [C_O2_ccl] + [C_O2_cgdl] * n_gdl + [C_O2_cgc] + [C_N2] +
                                                [C_v_agc] + [C_v_agdl] * n_gdl + [C_v_acl, C_v_ccl+1] + [C_v_cgdl] * n_gdl + [C_v_cgc] +
                                                [s_boundary] + [s_agdl] * (n_gdl - 1) + [s_acl, s_ccl] + [s_cgdl] * (n_gdl - 1) + [s_boundary] +
                                                [lambda_mem_ini] + [lambda_mem_ini] * n_mem + [lambda_mem_ini] +
                                                [eta_c, Pasm, Paem, Pcsm, Pcem, Phi_asm, Phi_aem, Phi_csm, Phi_cem] +
                                                [Wcp, Wa_inj, Wc_inj, Abp_a, Abp_c] + C_Pt_mem_init +
                                                [C_Pt2_ccl, Hmem] + prd_init.tolist() + theta_CCL.tolist() +
                                                [operating_inputs["Tfc"]] * 32)