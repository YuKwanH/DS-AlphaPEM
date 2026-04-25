from modules.kinetic_eq import *
from config.settings import *
from model.coefficients import Psat, rho_H2O


def dxdt_AGC(dif,Jv_a_in, Jv_a_out, Lgc, Jv_agc_agdl, Hgc, J_H2_in, J_H2_out, J_H2_agc_agdl, **kwargs):
    dif['dC_v_agc / dt'] = (Jv_a_in - Jv_a_out) / Lgc - Jv_agc_agdl / Hgc
    dif['dC_H2_agc / dt'] = (J_H2_in - J_H2_out) / Lgc - J_H2_agc_agdl / Hgc
    

def dxdt_AGDL(dif,sv, Hgdl, epsilon_gdl, n_gdl, Jl_agdl_agdl, Jl_agdl_acl, Sl_agdl, Jv_agc_agdl,
                                    Jv_agdl_agdl, Jv_agdl_acl, Sv_agdl, J_H2_agc_agdl, J_H2_agdl_agdl, J_H2_agdl_acl, **kwargs):

    #dif['ds_agdl_1 / dt'] = 0  # Dirichlet boundary condition. s_agdl_1 is initialized to 0 and remains constant.
    for i in range(2, n_gdl):
        dif[f'ds_agdl_{i} / dt'] = 1 / (rho_H2O(sv[f"Tcgdl_{i}"]) * epsilon_gdl) * ((Jl_agdl_agdl[i - 1] - Jl_agdl_agdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[i])
    dif[f'ds_agdl_{n_gdl} / dt'] = 1 / (rho_H2O(sv[f"Tcgdl_{n_gdl}"]) * epsilon_gdl) * ((Jl_agdl_agdl[n_gdl - 1] - Jl_agdl_acl) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[n_gdl])

    dif['dC_v_agdl_1 / dt'] = 1 / (epsilon_gdl * (1 - sv['s_agdl_1'])) * ((Jv_agc_agdl - Jv_agdl_agdl[1]) / (Hgdl / n_gdl) + Sv_agdl[1])
    for i in range(2, n_gdl):
        dif[f'dC_v_agdl_{i} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_agdl_{i}'])) * ((Jv_agdl_agdl[i - 1] - Jv_agdl_agdl[i]) / (Hgdl / n_gdl) + Sv_agdl[i])
    dif[f'dC_v_agdl_{n_gdl} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_agdl_{n_gdl}'])) * ((Jv_agdl_agdl[n_gdl - 1] - Jv_agdl_acl) / (Hgdl / n_gdl) + Sv_agdl[n_gdl])

    dif['dC_H2_agdl_1 / dt'] = 1 / (epsilon_gdl * (1 - sv['s_agdl_1'])) * (J_H2_agc_agdl - J_H2_agdl_agdl[1]) / (Hgdl / n_gdl)
    for i in range(2, n_gdl):
        dif[f'dC_H2_agdl_{i} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_agdl_{i}'])) * (J_H2_agdl_agdl[i - 1] - J_H2_agdl_agdl[i]) / (Hgdl / n_gdl)
    dif[f'dC_H2_agdl_{n_gdl} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_agdl_{n_gdl}'])) * (J_H2_agdl_agdl[n_gdl - 1] - J_H2_agdl_acl) / (Hgdl / n_gdl)


def dxdt_CGC(dif,J_O2_in, J_O2_out, Jv_c_out, Jv_c_in, Jv_cgdl_cgc, J_O2_cgdl_cgc, Hgc, Lgc, **kwargs):
    dif['dC_v_cgc / dt'] = (Jv_c_in - Jv_c_out) / Lgc + Jv_cgdl_cgc / Hgc
    dif['dC_O2_cgc / dt'] = (J_O2_in - J_O2_out) / Lgc + J_O2_cgdl_cgc / Hgc


def dxdt_CGDL(dif,sv, Jv_ccl_cgdl, Jv_cgdl_cgdl, Sv_cgdl, Jv_cgdl_cgc, epsilon_gdl, J_O2_ccl_cgdl,
                                    J_O2_cgdl_cgdl, J_O2_cgdl_cgc, Jl_ccl_cgdl, Jl_cgdl_cgdl, Sl_cgdl, n_gdl, Hgdl, **kwargs):
    dif['ds_cgdl_1 / dt'] = 1 / (rho_H2O(sv["Tcgdl_1"]) * epsilon_gdl) * ((Jl_ccl_cgdl - Jl_cgdl_cgdl[1]) / (Hgdl / n_gdl) + M_H2O * Sl_cgdl[1])
    for i in range(2, n_gdl):
        dif[f'ds_cgdl_{i} / dt'] = 1 / (rho_H2O(sv[f"Tcgdl_{i}"]) * epsilon_gdl) * ((Jl_cgdl_cgdl[i - 1] - Jl_cgdl_cgdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_cgdl[i])
    dif[f'ds_cgdl_{n_gdl} / dt'] = 0 
    dif['dC_v_cgdl_1 / dt'] = 1 / (epsilon_gdl * (1 - sv['s_cgdl_1'])) * ((Jv_ccl_cgdl - Jv_cgdl_cgdl[1]) / (Hgdl / n_gdl) + Sv_cgdl[1])
    for i in range(2, n_gdl):
        dif[f'dC_v_cgdl_{i} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_cgdl_{i}'])) * ((Jv_cgdl_cgdl[i - 1] - Jv_cgdl_cgdl[i]) / (Hgdl / n_gdl) + Sv_cgdl[i])
    dif[f'dC_v_cgdl_{n_gdl} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_cgdl_{n_gdl}'])) * ((Jv_cgdl_cgdl[n_gdl - 1] - Jv_cgdl_cgc) / (Hgdl / n_gdl) + Sv_cgdl[n_gdl])
    dif['dC_O2_cgdl_1 / dt'] = 1 / (epsilon_gdl * (1 - sv['s_cgdl_1'])) * (J_O2_ccl_cgdl - J_O2_cgdl_cgdl[1]) / (Hgdl / n_gdl)
    for i in range(2, n_gdl):
        dif[f'dC_O2_cgdl_{i} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_cgdl_{i}'])) * (J_O2_cgdl_cgdl[i - 1] - J_O2_cgdl_cgdl[i]) / (Hgdl / n_gdl)
    dif[f'dC_O2_cgdl_{n_gdl} / dt'] = 1 / (epsilon_gdl * (1 - sv[f's_cgdl_{n_gdl}'])) * (J_O2_cgdl_cgdl[n_gdl - 1] - J_O2_cgdl_cgc) / (Hgdl / n_gdl)


def dxdt_ACL(dif,sv, Jl_agdl_acl, Hcl, Sl_acl, Jv_agdl_acl, J_H2_agdl_acl, S_sorp_acl, S_H2_acl,
                                Sv_acl, epsilon_cl, Sp_acl, J_H2_acl_mem, **kwargs):
    dif['ds_acl / dt'] = 1 / (rho_H2O(sv["Tacl"]) * epsilon_cl) * (Jl_agdl_acl / Hcl + M_H2O * Sl_acl)
    dif['dC_v_acl / dt'] = 1 / (epsilon_cl * (1 - sv['s_acl'])) * (Jv_agdl_acl / Hcl - S_sorp_acl + Sv_acl + + Sp_acl)
    dif['dC_H2_acl / dt'] = 1 / (epsilon_cl * (1 - sv['s_acl'])) * ((J_H2_agdl_acl - J_H2_acl_mem)/ Hcl + S_H2_acl)


def dxdt_MEM(dif, J_lambda_mem_acl, J_lambda_mem_ccl, J_O2_mem, J_H2_mem, J_lambda_mem,
                                   J_O2_mem_ccl, J_H2_acl_mem, J_Pt2_mem,S_Pt2_mem, Sp_ccl,
                                   epsilon_mc, Hcl, Hmem, S_sorp_acl, S_sorp_ccl, Tfc, Ucell,C_O2_ccl, n_mem,**kwargs):

    dif['dlambda_acl / dt'] = M_eq / (rho_mem * epsilon_mc) * (-J_lambda_mem_acl / Hcl + S_sorp_acl)
    dif['dlambda_ccl / dt'] = M_eq / (rho_mem * epsilon_mc) * (J_lambda_mem_ccl / Hcl + S_sorp_ccl)
    dif['dlambda_mem_1 / dt'] = M_eq / rho_mem * (J_lambda_mem_acl - J_lambda_mem[0]) / Hmem
    for i in range(2, n_mem):
         dif[f'dlambda_mem_{i} / dt'] = M_eq / rho_mem * (J_lambda_mem[i-2] - J_lambda_mem[i-1]) / Hmem
    dif[f'dlambda_mem_{n_mem} / dt'] = M_eq / rho_mem * (J_lambda_mem[-1] - J_lambda_mem_ccl) / Hmem

    dif['ddelta_mem / dt'] = 0 #-20.8 / (0.82 * 1980e3) * flourideReleaseRate(MT = Hmem, U = Ucell, Tfc = Tfc, PO2_ca = C_O2_ccl * R * Tfc)

    #Crossover fluxes of O2 and H2 in the membrane
    dif[f'dC_H2_mem_{1} / dt'] = (J_H2_acl_mem - J_H2_mem[0]) / (Hmem / n_mem+Hcl)
    dif[f'dC_O2_mem_{n_mem} / dt'] = (J_O2_mem[n_mem - 1] - J_O2_mem_ccl) / (Hmem / n_mem+ Hcl)
    for i in np.arange(2, n_mem):
        dif[f'dC_O2_mem_{i} / dt'] = (J_O2_mem[i - 1] - J_O2_mem[i]) / (Hmem / n_mem)
        dif[f'dC_H2_mem_{i} / dt'] = (J_H2_mem[i - 1] - J_H2_mem[i]) / (Hmem / n_mem) # + S_Pt2_mem[i - 1]
        dif[f'dC_Pt2_mem_{i} / dt'] = 0 #((J_Pt2_mem[i-1] - J_Pt2_mem[i]) / (Hmem / n_mem)) / (1 - epsilon_mc)  # + S_Pt2_mem[i - 1]
    dif[f'dC_H2_mem_{n_mem} / dt'] = (J_H2_mem[- 1] - 0) / (Hmem / n_mem)


def dxdt_CCL(dif,sv, Jl_ccl_cgdl, Hcl, Sl_ccl, Jv_ccl_cgdl, S_sorp_ccl, Sv_ccl, J_O2_ccl_cgdl, J_O2_mem_ccl, Sp_ccl,
                                S_O2_ccl, prd0, prd_ccl, kcdis, r_m, drdt, J_Pt2_mem, Hmem, n_mem, epsilon_mc, epsilon_cl, **kwargs):

    M_Pt0 = 4 / 3 * np.pi * rho_Pt * np.trapezoid(y=prd0 * r_m ** 3, x=r_m)
    dMdisdt = 4 * np.pi * rho_Pt * np.trapezoid(y=prd_ccl * r_m ** 2 * drdt, x=r_m)
    dMcdisdt = 4 * np.pi * rho_Pt * np.trapezoid(y=prd_ccl * r_m ** 2 * kcdis, x=r_m)
    dif['dC_Pt2_ccl / dt'] = 0#-3.33 / M_Pt * (dMdisdt - dMcdisdt) / M_Pt0 - J_Pt2_mem[-1] / (Hmem / n_mem)/ (1 - epsilon_mc)
    dif['ds_ccl / dt'] = 1 / (rho_H2O(sv["Tccl"]) * epsilon_cl) * (- Jl_ccl_cgdl / Hcl + M_H2O * Sl_ccl)
    dif['dC_v_ccl / dt'] = 1 / (epsilon_cl * (1 - sv['s_ccl'])) * (- Jv_ccl_cgdl / Hcl - S_sorp_ccl + Sv_ccl + Sp_ccl)
    dif['dC_O2_ccl / dt'] = 1 / (epsilon_cl * (1 - sv['s_ccl'])) * ((J_O2_mem_ccl- J_O2_ccl_cgdl) / Hcl  + S_O2_ccl)


def dxdt_U(dif, i_fc, C_O2_ccl, eta_c, Tccl, Hcl, i0_c_ref, kappa_c, C_scl, f_drop, ECSA, **kwargs):
    i0_c = (ECSA * i0_c_ref * (C_O2_ccl / C_O2ref)** kappa_c) * np.exp(Eact / R * (1 / 353 - 1 / Tccl))
    dif['deta_c / dt'] = 1 / (C_scl * Hcl) * ((i_fc) - i0_c * np.exp(f_drop * alpha_c * F / (R * Tccl) * eta_c))


def dxdt_N2(dif,J_N2_in, J_N2_out, Lgc, **kwargs):
    dif['dC_N2 / dt'] = (J_N2_in - J_N2_out) / Lgc


def dxdt_Manifold(dif,Masm, Maem, Mcsm, Mcem, Tfc, Hgc, Wgc,
                                            Jv_a_in, Jv_a_out, Jv_c_in, Jv_c_out,
                                            Wasm_in, Wasm_out, Waem_in, Waem_out, Wcsm_in, Wcsm_out,
                                            Wcem_in, Wcem_out, Wv_asm_in, Wv_aem_out, Wv_csm_in,
                                            Wv_cem_out, **kwargs):
    """
    This function calculates the dynamic evolution of the pressure and humidity inside the manifolds.
    """
    # Pressure evolution inside the manifolds
    dif['dPasm / dt'] = (Wasm_in - n_cell * Wasm_out) / (Vsm * Masm) * R * Tfc
    dif['dPaem / dt'] = (n_cell * Waem_in - Waem_out) / (Vem * Maem) * R * Tfc
    dif['dPcsm / dt'] = (Wcsm_in - n_cell * Wcsm_out) / (Vsm * Mcsm) * R * Tfc
    dif['dPcem / dt'] = (n_cell * Wcem_in - Wcem_out) / (Vem * Mcem) * R * Tfc
    # Humidity evolution inside the manifolds
    dif['dPhi_asm / dt'] = (Wv_asm_in - Jv_a_in * Hgc * Wgc * n_cell) / Vsm * R * Tfc / Psat(Tfc)
    dif['dPhi_aem / dt'] = (Jv_a_out * Hgc * Wgc * n_cell - Wv_aem_out) / Vem * R * Tfc / Psat(Tfc)
    dif['dPhi_csm / dt'] = (Wv_csm_in - Jv_c_in * Hgc * Wgc * n_cell) / Vsm * R * Tfc / Psat(Tfc)
    dif['dPhi_cem / dt'] = (Jv_c_out * Hgc * Wgc * n_cell - Wv_cem_out) / Vem * R * Tfc / Psat(Tfc)

def dxdt_CP(dif, Wcp_des, Wa_inj_des, Wc_inj_des, Wcp, Wa_inj, Wc_inj, **kwargs):

    # Air compressor evolution
    dif['dWcp / dt'] = (Wcp_des - Wcp) / tau_cp  # Estimation at the first order.
    # Anode and cathode humidifiers evolution
    dif['dWa_inj / dt'] = (Wa_inj_des - Wa_inj) / tau_hum  # Estimation at the first order.
    dif['dWc_inj / dt'] = (Wc_inj_des - Wc_inj) / tau_hum  # Estimation at the first order.

def dxdt_TH(dif,Pagc, Pcgc, Abp_a, Abp_c, Tfc, Pa_des, Pc_des, **kwargs):
    
    # Calculation of the pressure derivative inside the gas channels
    dPagcdt = (dif['dC_v_agc / dt'] + dif['dC_H2_agc / dt']) * R * Tfc
    dPcgcdt = (dif['dC_v_cgc / dt'] + dif['dC_O2_cgc / dt'] + dif['dC_N2 / dt']) * R * Tfc

    # Throttle area evolution inside the anode auxiliaries
    dif['dAbp_a / dt'] = - 1e-6 * (Pa_des - Pagc) #+ 1e-7 * dPagcdt  # PD controller
    if Abp_a > A_T and dif['dAbp_a / dt'] > 0:  # The throttle area cannot be higher than the maximum value
        dif['dAbp_a / dt'] = 0
    elif Abp_a < 0 and dif['dAbp_a / dt'] < 0:  # The throttle area cannot be lower than 0
        dif['dAbp_a / dt'] = 0

    # Throttle area evolution inside the cathode auxiliaries
    dif['dAbp_c / dt'] = - Kp * (Pc_des - Pcgc) #+ Kd * dPcgcdt  # PD controller
    if Abp_c > A_T and dif['dAbp_c / dt'] > 0:  # The throttle area cannot be higher than the maximum value
        dif['dAbp_c / dt'] = 0
    elif Abp_c < 0 and dif['dAbp_c / dt'] < 0:  # The throttle area cannot be lower than 0
        dif['dAbp_c / dt'] = 0


def dxdt_PRD(dif, prd, theta, kox, kcdis, kdet, drdt, r_m):
    """
    Urchaga et al 2015 dr/dt = VmKrdpCpt,avg*exp(-R0/r) - VmKdisCpt,avg*exp(-R0/r)
    """
    dfdt = -np.gradient(prd * drdt, r_m) - kdet * prd
    dthetadt = (((kox - kcdis) / GAMMA_max) - (2 * theta / r_m) * drdt)
    for i in range(len(r_m)):
        dif[f"dS_N_ccl_{i}"+' / dt'] = 0 #dfdt[i]
        dif[f"dtheta_ccl_{i}"+' / dt'] = 0 # dthetadt[i]
