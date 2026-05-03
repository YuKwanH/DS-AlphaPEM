import scipy.stats as stats
from scipy.integrate import trapezoid
# Importing constants' value and functions
from model.coefficients import *
from model.kinetic_eq import *

# ____________________________________________Differential equations modules____________________________________________
def dif_eq_int_values(t, x, operating_inputs, parameters):

    # Extraction of the variables
    iload = operating_inputs['current_density'](t)
    C_v_agc, C_v_cgc = x['C_v_agc'], x['C_v_cgc']
    C_H2_agc,C_O2_cgc, C_N2 = x['C_H2_agc'], x['C_O2_cgc'], x['C_N2']   
    Pasm, Paem, Pcsm, Pcem =  x['Pasm'], x['Paem'], x['Pcsm'], x['Pcem']
    Phi_aem, Phi_csm, Phi_asm, Phi_cem = x['Phi_aem'], x['Phi_csm'], x['Phi_asm'], x['Phi_cem']
    Tfc, Phi_c_des, Phi_a_des = operating_inputs['Tfc'], operating_inputs['Phi_c_des'], operating_inputs['Phi_a_des']
    Sa, Sc = operating_inputs['Sa'], operating_inputs['Sc']
    n_gdl, n_group_pt = parameters['n_gdl'], parameters['n_group_pt']
    r_m = parameters['r_m']
    Wcp = x["Wcp"]
    Aact = parameters['Aact']
    # Physical quantities
    # Pressures
    Pagc = (C_v_agc + C_H2_agc) * R * Tfc
    Pcgc = (C_v_cgc + C_O2_cgc + C_N2) * R * Tfc
    Prd = Pasm
    Pcp = Pcsm
    # Pressures in the stack
    Pagdl = [(x[f'C_v_agdl_{i}'] + x[f'C_H2_agdl_{i}']) * R * x[f"Tagdl_{i}"] for i in range(1, n_gdl + 1)]
    Pacl = (x["C_v_acl"] + x["C_H2_acl"]) * R * x['Tacl']
    Pccl = (x["C_v_ccl"] + x["C_O2_ccl"] + x["C_N2"]) * R * x['Tccl']
    Pcgdl = [(x[f'C_v_cgdl_{i}'] + x[f'C_O2_cgdl_{i}'] + x["C_N2"]) * R * x[f"Tcgdl_{i}"] for i in range(1, n_gdl + 1)]
    # Humidities
    Phi_agc = C_v_agc / C_v_sat(Tfc)
    Phi_cgc = C_v_cgc / C_v_sat(Tfc)
    # Oxygen ratio in dry air
    y_cgc = C_O2_cgc / (C_O2_cgc + C_N2)    
    # Molar masses
    Masm = Phi_asm * Psat(Tfc) / Pasm * M_H2O + (1 - Phi_asm * Psat(Tfc) / Pasm) * M_H2
    Maem = Phi_aem * Psat(Tfc) / Paem * M_H2O + (1 - Phi_aem * Psat(Tfc) / Paem) * M_H2
    Mcsm = Phi_csm * Psat(Tfc) / Pcsm * M_H2O + yO2_ext * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_O2 + (1 - yO2_ext) * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_N2
    Mcem = Phi_cem * Psat(Tfc) / Pcem * M_H2O + y_cgc * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_O2 + (1 - y_cgc) * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_N2
    Mext = (Phi_ext * Psat(Text) / Pext * M_H2O + 
                    yO2_ext * (1 - Phi_ext * Psat(Text) / Pext) * M_O2 + 
                    (1 - yO2_ext) * (1 - Phi_ext * Psat(Text) / Pext) * M_N2)
    Magc = C_v_agc * R * Tfc / Pagc * M_H2O +  C_H2_agc * R * Tfc / Pagc * M_H2
    Mcgc = (Phi_cgc * Psat(Tfc) / Pcgc * M_H2O + 
                    y_cgc * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_O2 + 
                    (1 - y_cgc) * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_N2)
    # Physical quantities in the auxiliary system
    # Pressure ratios
    Pr_aem = (Pext / Paem)
    Pr_cem = (Pext / Pcem)
    # Oxygen ratio in dry air
    y_cem = ((Pcem - Phi_cem * Psat(Tfc) - C_N2 * R * Tfc) / 
             (Pcem - Phi_cem * Psat(Tfc)))
    # Molar masses
    Masm = Phi_asm * Psat(Tfc) / Pasm * M_H2O + (1 - Phi_asm * Psat(Tfc) / Pasm) * M_H2
    Maem = Phi_aem * Psat(Tfc) / Paem * M_H2O + (1 - Phi_aem * Psat(Tfc) / Paem) * M_H2

    Mcsm = (Phi_csm * Psat(Tfc) / Pcsm * M_H2O + 
                    yO2_ext * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_O2 + 
                    (1 - yO2_ext) * (1 - Phi_csm * Psat(Tfc) / Pcsm) * M_N2)
    Mcem = (Phi_cem * Psat(Tfc) / Pcem * M_H2O + 
                    y_cem * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_O2 + 
                    (1 - y_cem) * (1 - Phi_cem * Psat(Tfc) / Pcem) * M_N2)
    
    # eletrochemical kinetics
    i_fc = operating_inputs['current_density'](t)
    fd = fdrop(x=x, operating_inputs=operating_inputs, parameters=parameters)
    U = Ucell(t=t, variables=x, operating_inputs=operating_inputs, parameters=parameters)
    PRD = np.zeros(n_group_pt)
    theta_ccl = np.zeros(n_group_pt)
    for i in range(n_group_pt):
        PRD[i] = x[f"S_N_ccl_{i + 1}"]
        theta_ccl[i] = x[f'theta_ccl_{i + 1}']
    ECSA = (getECSA(PRD, radius= r_m) / getECSA(parameters["prd0"], radius= r_m))
    # CCL kinetic
    C_Proton = Cproton_CCL(lambda_w=x["lambda_ccl"])
    kdis = PtDissolution(U, x["Tccl"], x["C_Pt2_ccl"], theta_ccl)
    kox = PtOxidation(U, x["Tccl"], C_Proton, theta_ccl)
    kcdis = PtOxideDissolution(theta_ccl, C_Proton)
    kdet = PtDetachment(U, x["Tccl"], r_m)
    
    # Setpoints 
    # The desired air compressor flow rate Wcp_des (kg.s-1)
    Wcp_des = (n_cell * Mext * Pext / (Pext - Phi_ext * Psat(Text)) * 1 / yO2_ext * Sc * (iload) / (4 * F) * Aact)
    Wrd = n_cell * M_H2 * Sa * (iload) / (2 * F) * Aact
    Wa_inj_des = M_H2O * Phi_a_des * Psat(Tfc) / Prd * (Wrd / M_H2)
    # The desired humidifier flow rate at the cathode side Wc_inj_des (kg.s-1)
    Wv_hum_in = M_H2O * Phi_ext * Psat(Text) / Pext * (Wcp / Mext)  # Vapor flow rate from the outside
    Wc_v_des = M_H2O * Phi_c_des * Psat(Tfc) / Pcp * (Wcp / Mext)  # Desired vapor flow rate
    Wc_inj_des = Wc_v_des - Wv_hum_in  # Desired humidifier flow rate

    result = {"Pagc": Pagc, "Pcgc": Pcgc, "Pacl": Pacl, "Pagdl": Pagdl, "Pccl": Pccl, "Pcgdl": Pcgdl,
                 "Prd": Prd, "Pcp": Pcp, "Pr_aem": Pr_aem, "Pr_cem": Pr_cem,
                 "Mext": Mext, "Masm": Masm, "Maem": Maem, "Mcsm": Mcsm, "Mcem": Mcem, "Magc": Magc, "Mcgc": Mcgc,
                 "Phi_agc": Phi_agc, "Phi_cgc": Phi_cgc,
                 "Wcp_des": Wcp_des, "Wa_inj_des": Wa_inj_des, "Wc_inj_des": Wc_inj_des,
                 "y_cgc": y_cgc, "y_cem": y_cem, "f_drop": fd, 'i_fc': i_fc, "Ucell": U,
                 'ECSA': ECSA, 'kox': kox, 'kcdis': kcdis, 'kdet': kdet, 'kdis': kdis, 'prd': PRD, 'theta_ccl': theta_ccl}
    
    return result


def calculate_flows(t, x ,operating_inputs, parameters, Pagc, Pcgc, Pacl,Pagdl, Pccl, Pcgdl,
                                 Pr_aem, Pr_cem, Mext, Masm, Maem, Mcsm, Mcem, Magc, Mcgc,
                                 Phi_agc, Phi_cgc, y_cgc, **kwargs):

    # Mapping macro-scale variables
    iload = operating_inputs["current_density"](t)
    Iload = iload * parameters["Aact"]
    # Mapping constant parameters
    Hcl, Hgdl, Hgc, Hmem, Wgc,Lgc, Aact = parameters["Hcl"], parameters["Hgdl"], parameters["Hgc"], parameters["Hmem"], parameters["Wgc"], parameters["Lgc"], parameters["Aact"]
    n_gdl, n_mem = parameters["n_gdl"], parameters["n_mem"]
    epsilon_gdl, epsilon_cl, epsilon_c = parameters["epsilon_gdl"], parameters["epsilon_cl"], parameters["epsilon_c"]
    e = parameters["e"]
    # Operating inputs
    Pa_des, Pc_des = operating_inputs['Pa_des'], operating_inputs["Pc_des"]
    Sa = operating_inputs['Sa']
    Tfc = operating_inputs["Tfc"]
    Imin_aux = operating_inputs["Imin_aux"]
    k_purge = 2.5
    
    # Mean values ...
    #       ... of the saturated liquid water variable
    s_agdl_agdl = [None] + [x[f's_agdl_{i}'] / 2 + x[f's_agdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
    s_agdl_acl = x[f's_agdl_{n_gdl}'] / 2 + x['s_acl'] / 2
    s_ccl_cgdl = x['s_ccl'] / 2 + x['s_cgdl_1'] / 2
    s_cgdl_cgdl = [None] + [x[f's_cgdl_{i}'] / 2 + x[f's_cgdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
    #       ... of the porosity and the contact angle
    epsilon_mean = epsilon_gdl / 2 + epsilon_cl / 2
    theta_c_mean = theta_c_gdl / 2 + theta_c_cl / 2
    #       ... of the pressure
    Pagdl_agdl = [Pa_des] * n_gdl
    Pagdl_acl = Pagdl[-1] / 2 + Pacl / 2
    Pccl_cgdl = Pccl / 2 + Pcgdl[0] / 2
    Pcgdl_cgdl = [Pa_des] * n_gdl
    
    # Inlet and outlet flows (mol/s) or (kg/s)
    # Anode inlet
    if Iload < Imin_aux and Imin_aux > 0:
        Wrd = n_cell * M_H2 * Sa * (Imin_aux / Aact ) / (2 * F) * Aact  # kg.s-1
    else:
        Wrd = n_cell * M_H2 * Sa * (iload) / (2 * F) * Aact  # kg.s-1
    Wasm_in = Wrd + x['Wa_inj']  # kg.s-1
    Wasm_out =  (x['Pasm'] - Pagc)  * Ksm_out  # kg.s-1
    Ja_in = Wasm_out / (Hgc * Wgc * Masm)  # mol.m-2.s-1
    Jv_a_in = x['Phi_asm'] * Psat(Tfc) / x['Pasm'] * Ja_in
    J_H2_in = (1 - x['Phi_asm'] * Psat(Tfc) / x['Pasm']) * Ja_in
    Wv_asm_in = x['Wa_inj'] / M_H2O
    # Anode outlet
    Waem_in = Kem_in * (Pagc - x["Paem"])  # kg.s-1
    Ware = 0  # kg.s-1
    Waem_out = C_D * x['Abp_a'] * x["Paem"] / np.sqrt(R * Tfc) * Pr_aem ** (1 / gamma_H2) * \
                          np.sqrt(Magc * 2 * gamma_H2 / (gamma_H2 - 1) * (1 - Pr_aem ** ((gamma_H2 - 1) / gamma_H2)))
    Ja_out = Waem_in / (Hgc * Wgc * Magc)  # mol.m-2.s-1
    Jv_a_out = Phi_agc * Psat(Tfc) / Pagc * Ja_out
    J_H2_out = (1 - Phi_agc * Psat(Tfc) / Pagc) * Ja_out
    Wv_aem_out = x['Phi_aem'] * Psat(Tfc) / x['Paem'] * (Waem_out / Maem)
    # Cathode inlet         
    Wcsm_in = x['Wcp'] + x['Wc_inj']  # kg.s-1
    Wcsm_out = (x['Pcsm'] - Pcgc) * Ksm_out  # kg.s-1
    Jc_in = Wcsm_out / (Hgc * Wgc * Mcsm)  # mol.m-2.s-1
    J_O2_in = yO2_ext * (1 - x['Phi_csm'] * Psat(Tfc) / x['Pcsm']) * Jc_in
    Jv_c_in = x['Phi_csm'] * Psat(Tfc) / x['Pcsm'] * Jc_in
    Wv_csm_in = Phi_ext * Psat(Text) / Pext * (x['Wcp'] / Mext) + x['Wc_inj'] / M_H2O
    J_N2_in = (1 - yO2_ext) * (1 - x['Phi_csm'] * Psat(Tfc) / x['Pcsm']) * Jc_in
    # Cathode outlet
    Wcem_in = Kem_in * (Pcgc - x["Pcem"])  # kg.s-1
    Wcem_out = C_D * x['Abp_c'] * x["Pcem"] / np.sqrt(R * Tfc) * Pr_cem ** (1 / gamma) * \
                            np.sqrt(Mcgc * 2 * gamma / (gamma - 1) * (1 - Pr_cem ** ((gamma - 1) / gamma)))  # kg.s-1
    Jc_out = Wcem_in / (Hgc * Wgc * Mcgc)  # mol.m-2.s-1
    J_O2_out = y_cgc * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * Jc_out
    Jv_c_out = Phi_cgc * Psat(Tfc) / Pcgc * Jc_out
    Wv_cem_out = x['Phi_cem'] * Psat(Tfc) / x['Pcem'] * (Wcem_out / Mcem)
    J_N2_out = (1 - y_cgc) * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * Jc_out
    
    #________________________________________Dissolved water flows (mol.m-2.s-1)_______________________________________
    # Anode side
    J_lambda_mem_acl = 2.5 / 22 * iload / F * x['lambda_acl'] - \
                                            rho_mem / M_eq * Dw(x['lambda_acl'], x['Tacl']) * (x['lambda_mem_1'] - x['lambda_acl']) / (Hmem/ n_mem + Hcl/10)
    # Cathode side
    J_lambda_mem_ccl = 2.5 / 22 * iload / F * x['lambda_ccl'] - \
                                            rho_mem / M_eq * Dw(x['lambda_ccl'], x['Tccl']) * (x['lambda_ccl'] - x[f'lambda_mem_{n_mem}']) / (Hmem/ n_mem + Hcl/5)
    # Membrane internal
    J_lambda_mem = [0] * (n_mem-1)
    for i in range(n_mem-1):
        J_lambda_mem[i] = 2.5 / 22 * iload / F * x[f'lambda_mem_{i+1}'] - \
                                            rho_mem / M_eq * Dw(x[f'lambda_mem_{i+1}'], x[f"Tmem_{i+1}"]) * (x[f'lambda_mem_{i+2}'] - x[f'lambda_mem_{i+1}']) / (Hmem / n_mem)

    # _____________________________________________Vapor flows (mol.m-2.s-1)____________________________________________
    # Convective vapor flows
    #   Anode side
    Jv_agc_agdl = h_a(Pagc, Tfc, Wgc, Hgc) * (x['C_v_agc'] - x['C_v_agdl_1']) #* Hcodi_a(iload)
    #   Cathode side
    Jv_cgdl_cgc = h_c(Pcgc, Tfc, Wgc, Hgc) * (x[f'C_v_cgdl_{n_gdl}'] - x['C_v_cgc'])  #* Hcodi_c(iload)
    # Conductive vapor flows
    #   Anode side
    Jv_agdl_agdl = [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jv_agdl_agdl[i-1] = - Da_eff(s_agdl_agdl[i], epsilon_gdl, Pagdl_agdl[i], x[f"Tagdl_{i}"], epsilon_c, epsilon_gdl) * \
                                        (x[f'C_v_agdl_{i + 1}'] - x[f'C_v_agdl_{i}']) / (Hgdl / n_gdl)
    Jv_agdl_acl = - 2 * Da_eff(s_agdl_acl, epsilon_mean, Pagdl_acl, (x[f"Tagdl_{n_gdl}"] + x['Tacl']) / 2, epsilon_c, epsilon_gdl) * \
                                (x["C_v_acl"] - x[f'C_v_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl/2)
    #   Cathode side
    Jv_cgdl_cgdl = [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jv_cgdl_cgdl[i-1] = - Dc_eff(s_cgdl_cgdl[i], epsilon_gdl, Pcgdl_cgdl[i], x[f"Tcgdl_{i}"], epsilon_c, epsilon_gdl) * \
                                        (x[f'C_v_cgdl_{i + 1}'] - x[f'C_v_cgdl_{i}']) / (Hgdl / n_gdl)
    Jv_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, (x[f"Tcgdl_1"] + x['Tccl']) / 2, epsilon_c, epsilon_gdl) * \
                            (x['C_v_cgdl_1'] - x["C_v_ccl"]) / (Hgdl / n_gdl + Hcl/2)
    
    # saturation front 
    nu_g =  1.881e-5
    ## Anode side
    Jwater = np.mean(Jv_agc_agdl)
    if Jwater < 0: # AGC <- ACL
        # ------------------- Regime M ------------------- #
        if x["C_v_acl"] > C_v_sat(x["Tacl"]) and x["C_v_agc"] <= C_v_sat(Tfc):
            s_agdl_avg = np.mean([x[f's_agdl_{i}'] for i in range(1, n_gdl + 1)])
            s_front_agdl = Hgdl -(C_v_sat(Tfc) - x["C_v_agc"]) * (Da_eff(s_agdl_avg,epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) / Jwater)
        # ------------------- Regime L ------------------- #
        elif x["C_v_agc"] > C_v_sat(Tfc) and x["C_v_acl"] > C_v_sat(x['Tacl']):
            mliquid = M_H2O * (-Jwater + (Jv_a_in - Jv_a_out) * Hgc/Lgc)
            ans1 = (mliquid * Lgc * nu_l(Tfc)/ (Hgc * rho_H2O(Tfc) * nu_g)) ** (1/3)
            s_agdl_inter = ans1 / (ans1 + 1)
            s_front_agdl = 0
        # ------------------- Regime V ------------------- #
        else:
            s_front_agdl = Hgdl
    else: # AGC -> ACL
        # ------------------- Regime M ------------------- #
        if  x["C_v_acl"] < C_v_sat(x['Tacl'])  and x['C_v_agc'] > C_v_sat(Tfc):
            mliquid = M_H2O * (-Jwater + (Jv_a_in - Jv_a_out) * Hgc/Lgc)
            ans1 = (mliquid * Lgc * nu_l(Tfc)/ (Hgc * rho_H2O(Tfc) * nu_g)) ** (1/3)
            s_agdl_inter = ans1 / (ans1 + 1)
            rhs = (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5)
            s_front_agdl = Hgdl - (0.35425 *s_agdl_inter ** 4 - 0.848 *  s_agdl_inter**5 + 0.6315 *  s_agdl_inter ** 6) * rhs / (M_H2O * -Jwater)
        elif  x["C_v_acl"] < C_v_sat(x['Tacl'])  and x['C_v_agc'] < C_v_sat(Tfc):
            s_front_agdl =Hgdl
        else:
            s_front_agdl = 0
    
    # Cathode side
    Jwater = np.mean(Jv_cgdl_cgdl)
    if Jwater > 0: # CCL -> CGC
        # ------------------- Regime M ------------------- #
        if  x["C_v_ccl"] >= C_v_sat(x['Tccl']) and x['C_v_cgc'] <= C_v_sat(Tfc):
            s_cgdl_avg = np.mean([x[f's_cgdl_{i}'] for i in range(1, n_gdl + 1)])
            s_front_cgdl = (C_v_sat(Tfc) - x["C_v_cgdl_10"]) * Dc_eff(s_cgdl_avg, epsilon_c, Pc_des, Tfc, epsilon_c, epsilon_gdl) * epsilon_gdl **1.5 / (Jwater)
        # ------------------- Regime L ------------------- #
        elif x['C_v_cgc'] > C_v_sat(Tfc) and x['C_v_ccl'] > C_v_sat(x['Tccl']):
            s_front_cgdl = Hgdl
            mliquid = M_H2O * (Jwater + (Jv_c_in - Jv_c_out) * Hgc/Lgc)
            ans1 = (mliquid * Lgc * nu_l(Tfc)/ (Hgc * rho_H2O(Tfc) * nu_g)) ** (1/3)
            s_cgdl_inter = ans1 / (ans1 + 1)
        else:
            s_front_cgdl = 0
    else: # CCL <- CGC
        mliquid = M_H2O * (Jwater + (Jv_c_in - Jv_c_out) * Hgc/Lgc)
        ans1 = (mliquid * Lgc * nu_l(Tfc)/ (Hgc * rho_H2O(Tfc) * nu_g)) ** (1/3)
        s_cgdl_inter = ans1 / (ans1 + 1)
        # ------------------- Regime M ------------------- #
        if  x["C_v_ccl"] <= C_v_sat(x['Tccl'])  and x['C_v_cgc'] >= C_v_sat(Tfc):
            rhs = (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5)
            s_front_cgdl = (0.35425 *s_cgdl_inter ** 4 - 0.848 *  s_cgdl_inter**5 + 0.6315 *  s_cgdl_inter ** 6) * rhs / (M_H2O * Jwater)
        elif  x["C_v_ccl"] < C_v_sat(x['Tccl'])  and x['C_v_cgc'] < C_v_sat(Tfc):
            s_cgdl_inter = 0
            s_front_cgdl = 0
        else:
            s_front_cgdl = Hgdl

    s_front_agdl = max(0, min(Hgdl, s_front_agdl))
    s_front_cgdl = max(0, min(Hgdl, s_front_cgdl))

    #_____________________________________Liquid water flows (kg.m-2.s-1)__________________________________________
    if s_front_agdl == 0:
        Jl_agdl_agc =  -s_front_agdl **3 / (1 - s_front_agdl) * rho_H2O(x["Tagdl_1"]) *(1/1298)  *4.8 * 1e-5/3e-4
    else:
        Jl_agdl_agc = 0
    # Anode side
    Jl_agdl_agdl = [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jl_agdl_agdl[i-1] = - sigma(x[f"Tagdl_{i}"]) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l(x[f"Tagdl_{i}"]) * abs(np.cos(theta_c_gdl)) * \
                                        (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * (s_agdl_agdl[i] ** e) * (1.417 - 4.24 * s_agdl_agdl[i] + 3.789 * s_agdl_agdl[i] ** 2) * \
                                        (x[f's_agdl_{i + 1}'] - x[f's_agdl_{i}']) / (Hgdl / n_gdl)
    Jl_agdl_acl = - 2 * sigma((x[f"Tagdl_{n_gdl}"] + x['Tacl']) / 2) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l((x[f"Tagdl_{n_gdl}"] + x['Tacl']) / 2) * abs(np.cos(theta_c_mean)) * \
                            (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * (s_agdl_acl ** e) * (1.417 - 4.24 * s_agdl_acl + 3.789 * s_agdl_acl ** 2) * \
                            (x["s_acl"] - x[f's_agdl_{n_gdl}']) / (Hgdl / n_gdl)
    
    # Cathode side
    if s_front_cgdl == Hgdl:
        Jl_cgdl_cgc = -s_cgdl_inter **3 / (1 - s_cgdl_inter) * rho_H2O(x["Tcgdl_{}".format(n_gdl)]) *(1/1298)  *4.8 * 1e-5/3e-4
    else:
        Jl_cgdl_cgc = 0
    Jl_cgdl_cgdl = [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jl_cgdl_cgdl[i-1] = - sigma(x[f"Tcgdl_{i}"]) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l(x[f"Tcgdl_{i}"]) * abs(np.cos(theta_c_gdl)) * \
                                        (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * (s_cgdl_cgdl[i] ** e) * (1.417 - 4.24 * s_cgdl_cgdl[i] + 3.789 * s_cgdl_cgdl[i] ** 2) * \
                                        (x[f's_cgdl_{i + 1}'] - x[f's_cgdl_{i}']) / (Hgdl / n_gdl)
    Jl_ccl_cgdl = - 2 * sigma((x["Tcgdl_1"] + x['Tccl']) / 2) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l((x["Tcgdl_1"] + x['Tccl']) / 2) * abs(np.cos(theta_c_mean)) * \
                            (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 *(s_ccl_cgdl ** e) * (1.417 - 4.24 * s_ccl_cgdl + 3.789 * s_ccl_cgdl ** 2) * \
                            (x['s_cgdl_1'] - x['s_ccl']) / (Hgdl / n_gdl + Hcl)
    
    # __________________________________________H2 and O2 flows (mol.m-2.s-1)___________________________________________
    # Hydrogen and oxygen consumption
    # Anode side
    S_H2_acl = - iload / (2 * F * Hcl) 
    # Cathode side
    S_O2_ccl = - iload / (4 * F * Hcl) 
    # Conductive-convective H2 and O2 flows
    #   Anode side
    J_H2_agc_agdl = h_a(Pagc, Tfc, Wgc, Hgc) * (x['C_H2_agc'] - x['C_H2_agdl_1'])
    #   Cathode side
    J_O2_cgdl_cgc = h_c(Pcgc, Tfc, Wgc, Hgc) * (x[f'C_O2_cgdl_{n_gdl}'] - x['C_O2_cgc'])

    # Conductive H2 and O2 flows
    # Anode side
    J_H2_agdl_agdl = [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        J_H2_agdl_agdl[i-1] = - Da_eff(s_agdl_agdl[i], epsilon_gdl, Pagdl_agdl[i], x[f"Tagdl_{i}"], epsilon_c, epsilon_gdl) * \
                                            (x[f'C_H2_agdl_{i + 1}'] - x[f'C_H2_agdl_{i}']) / (Hgdl / n_gdl)
    J_H2_agdl_acl = - 2 * Da_eff(s_agdl_acl, epsilon_mean, Pagdl_acl, (x[f"Tagdl_{n_gdl}"] + x['Tacl']) / 2, epsilon_c, epsilon_gdl) * \
                                (x["C_H2_acl"] - x[f'C_H2_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl/2)
    #   Cathode side
    J_O2_cgdl_cgdl = [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        J_O2_cgdl_cgdl[i-1] = - Dc_eff(s_cgdl_cgdl[i], epsilon_gdl, Pcgdl_cgdl[i], x[f"Tcgdl_{i}"], epsilon_c, epsilon_gdl) * \
                                            (x[f'C_O2_cgdl_{i + 1}'] - x[f'C_O2_cgdl_{i}']) / (Hgdl / n_gdl)
    J_O2_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, (x[f"Tcgdl_1"] + x['Tccl']) / 2, epsilon_c, epsilon_gdl) * \
                                (x['C_O2_cgdl_1'] - x["C_O2_ccl"]) / (Hgdl / n_gdl + Hcl/2)   

    # __________________________________________Water generated (mol.m-3.s-1)___________________________________________
    # Water produced in the membrane at the CL through the chemical reaction and crossover
    #   Anode side
    Sp_acl = 0
    #   Cathode side
    Sp_ccl = iload / (2 * F * Hcl)

    # Water sorption in the CL:
    #   Anode side
    S_sorp_acl = gamma_sorp(x['C_v_acl'], x['s_acl'], x['lambda_acl'], x['Tacl'], Hcl, Kshape) * rho_mem / M_eq * \
                            (lambda_eq(x['C_v_acl'], x['s_acl'], x['Tacl'], Kshape) - x['lambda_acl'])
    #   Cathode side
    S_sorp_ccl = gamma_sorp(x['C_v_ccl'], x['s_ccl'], x['lambda_ccl'], x['Tccl'], Hcl, Kshape) * rho_mem / M_eq * \
                            (lambda_eq(x['C_v_ccl'], x['s_ccl'], x['Tccl'], Kshape) - x['lambda_ccl'])

    # Liquid water generated through vapor condensation or degenerated through evaporation
    #   Anode side
    Sl_agdl = [Svl(x[f's_agdl_{i}'], x[f'C_v_agdl_{i}'], x[f'C_v_agdl_{i}'] + x[f'C_H2_agdl_{i}'], epsilon_gdl, x[f"Tagdl_{i}"], gamma_cond, gamma_evap)
                        for i in range(1, n_gdl + 1)]
    Sl_acl = Svl(x['s_acl'], x['C_v_acl'], x['C_v_acl'] + x['C_H2_acl'], epsilon_cl, x['Tacl'], gamma_cond, gamma_evap)
    #   Cathode side
    Sl_cgdl = [Svl(x[f's_cgdl_{i}'], x[f'C_v_cgdl_{i}'], Pc_des / (R * x[f"Tcgdl_{i}"]), epsilon_gdl, x[f"Tcgdl_{i}"], gamma_cond, gamma_evap) 
                        for i in range(1, n_gdl + 1)]
    Sl_ccl = Svl(x['s_ccl'], x['C_v_ccl'], Pc_des / (R * x['Tccl']), epsilon_cl, x['Tccl'], gamma_cond, gamma_evap)

    # Vapor generated through liquid water evaporation or degenerated through condensation
    #   Anode side
    Sv_agdl = [-S for S in Sl_agdl]
    Sv_acl = - Sl_acl
    #   Cathode side
    Sv_cgdl = [-S for S in Sl_cgdl]
    Sv_ccl = - Sl_ccl
    J_O2_mem = np.zeros(n_mem).tolist()
    J_H2_mem = np.zeros(n_mem).tolist()
    J_O2_mem_ccl = 0
    J_H2_acl_mem = 0

    for i_node in np.arange(1, n_mem):
        x_H2O = (x[f"lambda_mem_{i_node+1}"] * R_H2O) / (R_iono + x[f"lambda_mem_{i_node+1}"] * R_H2O)
        k_H2_mem = (0.29 + 2.2 * x_H2O) * 1e-15 * np.exp(2.1e4 / R * (1 / 303 - 1 / x[f"Tmem_{i_node+1}"]))
        k_O2_mem = (0.11 + 1.9 * x_H2O) * 1e-15 * np.exp(2.2e4 / R * (1 / 303 - 1 / x[f"Tmem_{i_node+1}"]))
        D_H2 = 2.584 * np.exp(170 / x[f"Tmem_{i_node+1}"]) * k_O2_mem
        D_O2 = 1348 * np.exp(-666 / x[f"Tmem_{i_node+1}"]) * k_H2_mem
        if i_node == 1:
            J_O2_mem[0] = -D_O2 * (x["C_O2_mem_1"] - 0) / (Hmem / n_mem)
            J_H2_mem[0] = -D_H2 * (x[f"C_H2_mem_1"] - 0) / (Hmem / n_mem)-D_H2 * (x[f"C_H2_mem_{i_node + 1}"] - x[f"C_H2_mem_{i_node}"]) / (Hmem / n_mem)
            J_H2_acl_mem = -D_H2 * (x[f"C_H2_mem_1"] - x['C_H2_acl']) / (Hmem / n_mem + Hcl/2)
        elif i_node == n_mem - 1:
            J_H2_mem[i_node] = -D_H2 * (0 - x[f"C_H2_mem_{n_mem}"]) / (Hmem / n_mem)
            J_O2_mem_ccl = -D_O2 * (x['C_O2_ccl'] - x[f"C_O2_mem_{n_mem}"]) / (Hmem / n_mem + Hcl/2)
            J_O2_mem[i_node] = -D_O2 * (x[f"C_O2_mem_{i_node + 1}"] - x[f"C_O2_mem_{i_node}"]) / (Hmem / n_mem)

        else:
            J_O2_mem[i_node] = -D_O2 * (x[f"C_O2_mem_{i_node + 1}"] - x[f"C_O2_mem_{i_node}"]) / (Hmem / n_mem)
            J_H2_mem[i_node] = -D_H2 * (x[f"C_H2_mem_{i_node + 1}"] - x[f"C_H2_mem_{i_node}"]) / (Hmem / n_mem)

    JT_ccl_cgdl = (k_CL+k_GDL)/2 * (x["Tccl"] - x["Tcgdl_1"]) / ((Hcl/2+Hgdl/n_gdl))
    JT_agdl_acl = (k_CL+k_GDL)/2 * (x[f"Tagdl_{n_gdl}"] - x["Tacl"]) / ((Hgdl/n_gdl+Hcl/2))
    JT_mem_ccl = (k_CL+k_PEM)/2 * (x[f"Tmem_{n_mem}"] - x["Tccl"]) / ((Hmem/n_mem+Hcl/2))
    JT_acl_mem = (k_CL+k_PEM)/2 * (x["Tacl"] - x["Tmem_1"]) / ((Hmem/n_mem+Hcl/2))
    JT_agc_agdl =  k_GDL * (Tfc - x["Tagdl_1"]) / (Hgdl/n_gdl + Hgc/2)
    JT_cgdl_cgc = k_GDL * (x[f"Tcgdl_{n_gdl}"] - Tfc) / (Hgdl/n_gdl + Hgc/2)

    Sr_acl = - deltaS_HOR * iload/(2*F)* x["Tacl"]
    Sr_ccl = - deltaS_OOR * iload/(4*F) * x["Tccl"] + deltaS_HOR * iload/(2*F)* x["Tccl"]
    Sad_acl =  0#massflow["S_sorp_acl"] * 42e3
    Sad_ccl =  0#massflow["S_sorp_ccl"] * 42e3
    Sec_acl = 0#massflow["Sv_acl"] * 42e3
    Sec_ccl = 0#massflow["Sv_ccl"] * 42e3
    Rmem, Racl, Rccl = Rproton(variables=x, parameters=parameters)
    Sre_acl = Racl * iload**2
    Sre_ccl = Rccl * iload**2
    Sre_mem =  np.array(Rmem) * iload**2 
    Sec_agdl = 0
    Sec_cgdl = 0

    JT_cgdl = np.zeros(n_gdl-1)
    JT_agdl = np.zeros(n_gdl-1)
    JT_mem = np.zeros(n_mem-1)
    for i in range(1, n_gdl):
        JT_agdl[i-1] = k_GDL * (x[f"Tagdl_{i}"] - x[f"Tagdl_{i+1}"]) / (Hgdl/n_gdl)
    for i in range(1, n_gdl):
        JT_cgdl[i-1] = k_GDL * (x[f"Tcgdl_{i}"] - x[f"Tcgdl_{i+1}"]) / (Hgdl/n_gdl)
    for i in range(1, n_mem):
        JT_mem[i-1] = k_PEM * (x[f"Tmem_{i}"] - x[f"Tmem_{i+1}"]) / (Hmem/n_mem)
    
    # Platinum ion dynamic
    J_Pt2_mem = np.zeros(n_mem + 1).tolist()
    # J_Pt2_mem[0] = D_Pt2 * (x["C_Pt2_mem_1"] - c_Pt2_AMI) * epsilon_cm **1.5 / (Hmem / n_mem + Hcl)
    # for i_node in np.arange(1, n_mem):
    #     J_Pt2_mem[i_node] = D_Pt2 * (sv[f"C_Pt2_mem_{i_node + 1}"] - sv[f"C_Pt2_mem_{i_node}"]) * epsilon_cm **1.5 / (Hmem / n_mem)
    # J_Pt2_mem[n_mem] = D_Pt2 * (c_Pt2_CMI - sv[f"C_Pt2_mem_{n_mem}"]) * epsilon_cm ** 1.5 / (Hmem / n_mem + Hcl)

    return {'Jv_a_in': Jv_a_in, 'Jv_a_out': Jv_a_out, 'Jv_c_in': Jv_c_in, 'Jv_c_out': Jv_c_out, 'J_H2_in': J_H2_in,
                'J_H2_out': J_H2_out, 'J_O2_in': J_O2_in, 'J_O2_out': J_O2_out, 'J_N2_in': J_N2_in, 'J_N2_out': J_N2_out,
                'Jv_agc_agdl': Jv_agc_agdl, 'Jv_agdl_agdl': Jv_agdl_agdl, 'Jv_agdl_acl': Jv_agdl_acl,
                'Jv_cgdl_cgc': Jv_cgdl_cgc, 'Jv_cgdl_cgdl': Jv_cgdl_cgdl, 'Jv_ccl_cgdl': Jv_ccl_cgdl, 
                'Jl_agdl_acl': Jl_agdl_acl, 'Jl_agdl_agdl': Jl_agdl_agdl, 'Jl_agdl_agc': Jl_agdl_agc,
                'Jl_ccl_cgdl': Jl_ccl_cgdl, 'Jl_cgdl_cgdl': Jl_cgdl_cgdl, 'Jl_cgdl_cgc': Jl_cgdl_cgc,
                's_front_agdl': s_front_agdl, 's_front_cgdl': s_front_cgdl,
                'J_lambda_mem_acl': J_lambda_mem_acl, 'J_lambda_mem_ccl': J_lambda_mem_ccl, 'J_lambda_mem': J_lambda_mem,
                'J_H2_agc_agdl': J_H2_agc_agdl, 'J_H2_agdl_agdl': J_H2_agdl_agdl, 'J_H2_agdl_acl': J_H2_agdl_acl,
                'J_H2_acl_mem': J_H2_acl_mem, 'J_H2_mem': J_H2_mem,
                'J_O2_ccl_cgdl': J_O2_ccl_cgdl, 'J_O2_cgdl_cgdl': J_O2_cgdl_cgdl, 'J_O2_cgdl_cgc': J_O2_cgdl_cgc,
                'J_O2_mem_ccl': J_O2_mem_ccl, 'J_O2_mem': J_O2_mem,
                'JT_ccl_cgdl': JT_ccl_cgdl, 'JT_agdl_acl': JT_agdl_acl, 'JT_agdl': JT_agdl, 'JT_cgdl': JT_cgdl,
                'JT_mem_ccl': JT_mem_ccl, 'JT_acl_mem': JT_acl_mem, 'JT_mem': JT_mem,
                'JT_agc_agdl': JT_agc_agdl, 'JT_cgdl_cgc': JT_cgdl_cgc,
                'J_Pt2_mem': J_Pt2_mem,
                'S_sorp_acl': S_sorp_acl, 'S_sorp_ccl': S_sorp_ccl, 'Sp_acl': Sp_acl, 'Sp_ccl': Sp_ccl,
                'S_H2_acl': S_H2_acl, 'S_O2_ccl': S_O2_ccl, 
                'Sv_cgdl': Sv_cgdl, 'Sv_agdl': Sv_agdl, 'Sv_acl': Sv_acl, 'Sv_ccl': Sv_ccl,
                'Sl_agdl': Sl_agdl, 'Sl_acl': Sl_acl, 'Sl_ccl': Sl_ccl, 'Sl_cgdl': Sl_cgdl,
                'Sre_acl': Sre_acl, 'Sre_ccl': Sre_ccl, 'Sre_mem': Sre_mem,
                'Sr_acl': Sr_acl, 'Sr_ccl': Sr_ccl,
                'Sec_agdl': Sec_agdl, 'Sec_cgdl': Sec_cgdl, 'Sec_acl': Sec_acl, 'Sec_ccl': Sec_ccl,
                'Sad_acl': Sad_acl, 'Sad_ccl': Sad_ccl,
                'Wasm_in': Wasm_in, 'Wasm_out': Wasm_out, 'Waem_in': Waem_in,
                'Waem_out': Waem_out, 'Wcsm_in': Wcsm_in, 'Wcsm_out': Wcsm_out, 'Wcem_in': Wcem_in, 'Wcem_out': Wcem_out,
                'Wrd': Wrd, 'Ware': Ware, 'Wv_asm_in': Wv_asm_in, 'Wv_aem_out': Wv_aem_out, 'Wv_csm_in': Wv_csm_in,
                'Wv_cem_out': Wv_cem_out}


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
    return 4 * np.pi * trapezoid(y=(radius ** 2) * prd, x=radius)

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