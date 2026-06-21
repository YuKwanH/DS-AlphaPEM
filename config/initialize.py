from model.coefficients import *
from model.inst_values import *


accessible_physical_parameters = {'Aact': 31e-4, 'Hmem': 1.2e-5, 'Hgc': 8e-4, 'Wgc': 4e-4, 'Lgc': 1.287} #m

current_parameters = {'t_step': (0, 3600, 100, 1.5), 'i_step': (0.5e4, 1.5e4),
                                        'delta_pola': (30, 30, 0.1e4, 60), 'i_max_pola': 1.65e4, # 50A
                                        'i_EIS': 1.0e4, 'ratio_EIS': 0.05, 't_EIS': 15, 'f_EIS': (-3, 5, 90, 50)}
 
operating_inputs = {'current_density': lambda x: 0.1e4, 'Tfc': 353.15, 
                                    'Pa_des': 1.8e5, 'Pc_des': 1.8e5,
                                    'Phi_a_des': 0.0, 'Phi_c_des': 0.85,
                                    'Sa': 1.2, 'Sc': 2.,
                                    'Imin_aux': 10}

undetermined_physical_parameters = {'epsilon_gdl': 0.6, "epsilon_cl": 0.35,
                                                                        'epsilon_mc': 0.399,'epsilon_c': 0.189, 
                                                                        'e': 4, 'kappa_co': 37.2, 'Re': 2.2e-7, 'tau': 1.01, 
                                                                        'i0_c_ref': 2.16, 'kappa_c': 2.9, 'C_scl': 1e8, 
                                                                        'a_slim': 0.4, 'b_slim': 0.5, 'a_switch': 0.5,
                                                                        "Hcl": 1e-5, "Hgdl": 2.e-4, "OCV": 0.95}

computing_parameters = {'max_step': 0.1, 'n_gdl': 10,'n_mem':10,'n_group_pt':50,
                                            't_purge': (2.4, 15), 'type_fuel_cell': "LEV-200", 'type_control': "Phi_des", 'type_purge': "constant_purge",
                                            'aux_system': True}  # True: simulate the auxiliary system (compressor / BoP); False: ideal fixed supply

parameters = {**current_parameters, **accessible_physical_parameters,
                          **undetermined_physical_parameters, **computing_parameters}

#   Initial fuel cell states
def init_x(operating_inputs, parameters):
    # Initial conditions
    current_density, Tfc = operating_inputs['current_density'], operating_inputs['Tfc']
    Pa_des, Pc_des = operating_inputs['Pa_des'], operating_inputs['Pc_des']
    Phi_a_des, Phi_c_des = operating_inputs['Phi_a_des'], operating_inputs['Phi_c_des']
    Hmem, i0_c_ref, = parameters['Hmem'], parameters['i0_c_ref']
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
    C_v_ini =  Psat(Tfc) / (R * Tfc)  #*Phi_des_moy # mol.m-3. It is the initial vapor concentration.
    C_H2_ini = (P_des_moy - Phi_des_moy * Psat_ini) / (R * Tfc)  # mol.m-3. It is the initial H2 concentration
    C_O2_ini = yO2_ext * (P_des_moy - Phi_des_moy * Psat_ini) / (R * Tfc)  # mol.m-3. It is the initial O2 concentration in the fuel cell.
    C_N2_ini = (1 - yO2_ext) * (P_des_moy - Phi_des_moy * Psat_ini) / (R * Tfc)  # mol.m-3. It is the initial N2  concentration in the fuel cell.
    s_ini = 0.001  # It is the initial liquid water saturation in the fuel cell.
    lambda_mem_ini = lambda_eq(C_v_ini, s_ini, Tfc, Kshape)  # It is the initial water content in the fuel cell.
    i_fc_ini = current_density(0)
    f_drop_ini = 0.5 * (1.0 - np.tanh((4 * s_ini - 2 * slim - 2 * s_switch) / (slim - s_switch)))
    # It is the initial cathode overpotential in the fuel cell.
    eta_c_ini = (1 / f_drop_ini * R * Tfc / (alpha_c * F) * np.log((i_fc_ini) / i0_c_ref * (C_O2ref / C_O2_ini) ** kappa_c) * np.exp(Eact / R * (1 / 353 - 1 / Tfc)))

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
    s_boundary = 0  # Dirichlet boundary con
    C_Pt2_ccl = 0
    C_Pt_mem_init = [0] * (n_mem - 1) + [C_Pt2_ccl/2]
    C_H2_mem_init = [0] + [0] * (n_mem -1)
    C_O2_mem_init = [0] * (n_mem - 1) + [0]
    C_H2_agc, C_H2_agdl, C_H2_acl = C_H2_ini, C_H2_ini, C_H2_ini
    C_O2_ccl, C_O2_cgdl, C_O2_cgc = C_O2_ini, C_O2_ini, C_O2_ini
    C_N2 = C_N2_ini
    Wcp, Wa_inj, Wc_inj, Abp_a, Abp_c = Wcp_ini, Wa_inj_ini, Wc_inj_ini, Abp_a_ini, Abp_c_ini
    prd_init = initPRD(resolution=computing_parameters['n_group_pt'])
    theta_CCL = np.zeros(computing_parameters['n_group_pt'])

    # Gathering of the variables initial value into one list
    x0 = ([C_H2_agc] + [C_H2_agdl] * n_gdl + [C_H2_acl] + C_H2_mem_init +
                C_O2_mem_init + [C_O2_ccl] + [C_O2_cgdl] * n_gdl + [C_O2_cgc] + [C_N2] +
                [C_v_agc] + [C_v_agdl] * n_gdl + [C_v_acl, C_v_ccl+1] + [C_v_cgdl] * n_gdl + [C_v_cgc] +
                [0] + s_agdl_init + [0.0, 0.01] + s_cgdl_init + [s_boundary] +
                [2] + [lambda_mem_ini] * n_mem + [lambda_mem_ini] + 
                [Wcp, Wa_inj, Wc_inj] + C_Pt_mem_init +
                [C_Pt2_ccl, Hmem] + prd_init.tolist() + theta_CCL.tolist())
    return x0


# ---------------------------------------------------------------------------
# Initial state for the 1-D dynamic model PEMFC_dyn (model/dynamic.py)
# ---------------------------------------------------------------------------
def init_x_dyn(operating_inputs, parameters):
    """Initial state vector for PEMFC_dyn (1-D model + balance-of-plant).

    Returns a list of length **181** under the default mesh
    (n_gdl = 10, n_mem = 10, n_pt_micro = 10 -- the last is hard-coded in
    PEMFC_dyn.micro_parameters), matching the order of
    ``PEMFC_dyn.solver_variable_names``. ``config.initialize.init_x`` returns
    a 218-element vector sized for the dual-scale model and is NOT
    compatible with PEMFC_dyn; this function exists to produce the
    right-sized vector for the dynamic model file.

    Variables are seeded from the operating-condition design point in
    exactly the same way as ``init_x``: ideal-gas reactant concentrations,
    equilibrium membrane water content, manifold pressures at the design
    pressure, BoP flows at rest (the BoP controller spins them up in the
    first few solver steps).
    """
    n_gdl = int(parameters['n_gdl'])
    n_mem = int(parameters['n_mem'])
    n_pt  = 10                       # PEMFC_dyn hardcodes n_group_ptParticle
    Hmem  = float(parameters['Hmem'])

    Tfc       = float(operating_inputs['Tfc'])
    Pa_des    = float(operating_inputs['Pa_des'])
    Pc_des    = float(operating_inputs['Pc_des'])
    Phi_a_des = float(operating_inputs['Phi_a_des'])
    Phi_c_des = float(operating_inputs['Phi_c_des'])

    Phi_des_moy = 0.5 * (Phi_a_des + Phi_c_des)
    P_des_moy   = 0.5 * (Pa_des + Pc_des)

    Psat_fc = Psat(Tfc)
    C_v_ini  = max(Phi_des_moy * Psat_fc / (R * Tfc), 1.0)
    C_H2_ini = max((Pa_des - Phi_a_des * Psat_fc) / (R * Tfc), 1.0)
    C_O2_dry = max((Pc_des - Phi_c_des * Psat_fc) / (R * Tfc), 1.0)
    C_O2_ini = yO2_ext * C_O2_dry
    C_N2_ini = (1.0 - yO2_ext) * C_O2_dry
    s_ini = 0.001
    lambda_ini = lambda_eq(C_v_ini, s_ini, Tfc, Kshape)

    x = []
    # C_H2 : agc, agdl*n_gdl, acl, mem*n_mem                          (22)
    x.append(C_H2_ini)
    x += [C_H2_ini] * n_gdl
    x.append(C_H2_ini)
    x += [0.1 * C_H2_ini] * n_mem
    # C_O2 + C_N : mem*n_mem, ccl, cgdl*n_gdl, cgc, C_N               (23)
    x += [0.1 * C_O2_ini] * n_mem
    x.append(C_O2_ini)
    x += [C_O2_ini] * n_gdl
    x.append(C_O2_ini)
    x.append(C_N2_ini)
    # C_v : agc, agdl*n_gdl, acl, ccl, cgdl*n_gdl, cgc                (24)
    x.append(C_v_ini)
    x += [C_v_ini] * n_gdl
    x.append(C_v_ini)
    x.append(C_v_ini)
    x += [C_v_ini] * n_gdl
    x.append(C_v_ini)
    # Liquid saturation : agdl*n_gdl, acl, ccl, cgdl*n_gdl            (22)
    x += [s_ini] * n_gdl
    x.append(s_ini)
    x.append(s_ini)
    x += [s_ini] * n_gdl
    # Lambda : acl, ccl, mem*n_mem                                    (12)
    x.append(lambda_ini)
    x.append(lambda_ini)
    x += [lambda_ini] * n_mem
    # Cathode overpotential                                            (1)
    x.append(0.3)
    # Manifold pressures (Pasm, Paem, Pcsm, Pcem)                      (4)
    x += [Pa_des, P_des_moy, Pc_des, P_des_moy]
    # Manifold RHs (Phi_asm, Phi_aem, Phi_csm, Phi_cem)                (4)
    x += [Phi_a_des, Phi_des_moy, Phi_c_des, Phi_des_moy]
    # BoP setpoints at rest -- controller spins them up
    # (Wcp, Wa_inj, Wc_inj, Abp_a, Abp_c)                              (5)
    x += [0.0, 0.0, 0.0, 0.0, 0.0]
    # Degradation : C_Pt2_mem*n_mem, C_Pt2_ccl, delta_mem,
    #               S_N_ccl*n_pt, theta_ccl*n_pt                      (32)
    x += [0.0] * n_mem
    x.append(0.0)
    x.append(Hmem)
    x += [1.0] * n_pt
    x += [0.0] * n_pt
    # Temperatures : Tagdl*n_gdl, Tacl, Tmem*n_mem, Tccl, Tcgdl*n_gdl (32)
    x += [Tfc] * n_gdl
    x.append(Tfc)
    x += [Tfc] * n_mem
    x.append(Tfc)
    x += [Tfc] * n_gdl
    return x


# ---------------------------------------------------------------------------
# Dispatcher: pick the right initial state for the requested model
# ---------------------------------------------------------------------------
def init_x_for(model_variant, operating_inputs, parameters):
    """Return an initial state vector sized for the requested model.

    Each PEMFC model in ``model/`` has a different state-vector size:

        ===================  ===========  ============================
        model_variant value  N_states     Source
        ===================  ===========  ============================
        'dual-scale'         218          init_x (this module)
        '0D'                 111          PEMFC_0D.default_initial_state
        'dynamic'            181          init_x_dyn (this module)
        'static'             None         PEMFC_stat is algebraic
        ===================  ===========  ============================

    The caller is responsible for passing the result into the right
    constructor (or, for 'static', skipping the initial-state step
    entirely). All variants are seeded from the same operating-condition
    design point (Tfc, Pa_des, Pc_des, Phi_*_des).

    The model_variant string is case-insensitive and tolerant of
    underscores / hyphens / spaces; valid values include:

        * 'dual-scale', 'dual_scale', 'dualscale', 'PEMFC'
        * '0D', 'zero-D', 'PEMFC_0D', 'lumped'
        * 'dynamic', 'dyn', 'PEMFC_dyn'
        * 'static', 'PEMFC_stat', 'polar'
    """
    if not model_variant:
        model_variant = 'dual-scale'
    name = str(model_variant).lower().replace('_', '').replace('-', '').replace(' ', '')

    if name in ('dualscale', 'pemfc', 'default'):
        return init_x(operating_inputs, parameters)
    if name in ('0d', 'zerod', 'pemfc0d', 'lumped'):
        # Local import to avoid a top-level config -> model.dualscale dependency
        # at import time (config/initialize is imported very early on startup).
        from model.dualscale import PEMFC_0D
        return PEMFC_0D.default_initial_state(parameters, operating_inputs)
    if name in ('dynamic', 'dyn', 'pemfcdyn', 'bop'):
        return init_x_dyn(operating_inputs, parameters)
    if name in ('static', 'pemfcstat', 'polar', 'steadystate'):
        return None                  # PEMFC_stat is an algebraic solver

    raise ValueError(
        f"Unknown model_variant {model_variant!r}. "
        "Valid choices: 'dual-scale', '0D', 'dynamic', 'static'."
    )
