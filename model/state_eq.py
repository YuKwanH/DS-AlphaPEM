from model.kinetic_eq import *
from model.coefficients import Psat, rho_H2O
from scipy.integrate import trapezoid


def dxdt_AGC(dif,Jv_a_in, Jv_a_out, Lgc, Jv_agc_agdl, Hgc, J_H2_in, J_H2_out, J_H2_agc_agdl, **kwargs):
    dif['dC_v_agc / dt'] = (Jv_a_in - Jv_a_out) / Lgc - Jv_agc_agdl / Hgc
    dif['dC_H2_agc / dt'] = (J_H2_in - J_H2_out) / Lgc - J_H2_agc_agdl / Hgc
    

def dxdt_AGDL(dif, x, Hgdl, Hgc, Hcl,epsilon_gdl,rho_H2O, n_gdl, Jl_agdl_agdl,Jl_agdl_acl, Tfc,
                         Jv_agc_agdl, Jv_agdl_agdl, Jv_agdl_acl,Sl_agdl, Sv_agdl, 
                         J_H2_agc_agdl, J_H2_agdl_agdl, J_H2_agdl_acl, **kwargs):

    for i in range(n_gdl):
        if i == 0: #AGC/AGDL interface
            dif['dC_v_agdl_1 / dt']    = ((Jv_agc_agdl - Jv_agdl_agdl[0]) / (Hgdl / n_gdl + Hgc/2) + Sv_agdl[1])/ (epsilon_gdl * (1 - x['s_agdl_1']))
            dif['dC_H2_agdl_1 / dt'] = (J_H2_agc_agdl - J_H2_agdl_agdl[0]) / (Hgdl / n_gdl + Hgc/2) / (epsilon_gdl * (1 - x['s_agdl_1']))
            dif[f'ds_agdl_1 / dt']        = 0
        elif i == n_gdl-1: #AGDL/ACL interface
            dif[f'dC_v_agdl_{n_gdl} / dt']    = ((Jv_agdl_agdl[-1] - Jv_agdl_acl) / (Hgdl / n_gdl + Hcl/2) + Sv_agdl[-1]) / (epsilon_gdl * (1 - x[f's_agdl_{n_gdl}']))
            dif[f'dC_H2_agdl_{n_gdl} / dt'] = (J_H2_agdl_agdl[-1] - J_H2_agdl_acl) / (Hgdl / n_gdl + Hcl/2) / (epsilon_gdl * (1 - x[f's_agdl_{n_gdl}']))
            dif[f'ds_agdl_{n_gdl} / dt']    = ((Jl_agdl_agdl[-1]-Jl_agdl_acl) / (Hgdl / n_gdl + Hcl/2) + M_H2O * Sl_agdl[-1]) / (rho_H2O * epsilon_gdl)
        else:
            dif[f'dC_v_agdl_{i+1} / dt']   = ((Jv_agdl_agdl[i - 1] - Jv_agdl_agdl[i]) / (Hgdl / n_gdl) + Sv_agdl[i+1]) / (epsilon_gdl * (1 - x[f's_agdl_{i+1}']))
            dif[f'dC_H2_agdl_{i+1} / dt'] =  (J_H2_agdl_agdl[i - 1] - J_H2_agdl_agdl[i]) / (Hgdl / n_gdl) / (epsilon_gdl * (1 - x[f's_agdl_{i+1}']))
            dif[f'ds_agdl_{i+1} / dt']        = ((Jl_agdl_agdl[i-1] - Jl_agdl_agdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[i+1]) / (rho_H2O * epsilon_gdl)


def dxdt_ACL(dif,x, epsilon_cl,  Hcl, Hgdl, rho_H2O, n_gdl, Jv_agdl_acl, Jl_agdl_acl,J_H2_agdl_acl, J_H2_acl_mem, 
                        Sp_acl,  Sl_acl, S_sorp_acl, S_H2_acl, Tfc, **kwargs):
    dif['dC_v_acl / dt']   = (Jv_agdl_acl / (Hgdl / n_gdl + Hcl/2) - S_sorp_acl + Sp_acl - Sl_acl) / (epsilon_cl * (1 - x['s_acl']))
    dif['dC_H2_acl / dt'] = ((J_H2_agdl_acl) / (Hgdl / n_gdl + Hcl/2) + S_H2_acl) / (epsilon_cl * (1 - x['s_acl']))
    dif['ds_acl / dt']         =   (Jl_agdl_acl / (Hgdl / n_gdl + Hcl/2) + M_H2O * Sl_acl)/ (rho_H2O * epsilon_cl)


def dxdt_CGC(dif,J_O2_in, J_O2_out, Jv_c_out, Jv_c_in, Jv_cgdl_cgc, J_O2_cgdl_cgc, 
                        Hgc, Lgc, **kwargs):
    dif['dC_v_cgc / dt'] = (Jv_c_in - Jv_c_out) / Lgc + (Jv_cgdl_cgc) / Hgc
    dif['dC_O2_cgc / dt'] = (J_O2_in - J_O2_out) / Lgc + J_O2_cgdl_cgc / Hgc


def dxdt_CGDL(dif, x,  epsilon_gdl, n_gdl, rho_H2O, Hgdl, Hgc, Hcl, Tfc,
                         Jv_ccl_cgdl, Jv_cgdl_cgdl, Jv_cgdl_cgc, Jl_ccl_cgdl,
                         Sv_cgdl, Sl_cgdl, Jl_cgdl_cgdl,
                         J_O2_ccl_cgdl, J_O2_cgdl_cgdl, J_O2_cgdl_cgc,  **kwargs):
    for i in range(n_gdl):
        if i == 0:
            dif['dC_v_cgdl_1 / dt'] = ((Jv_ccl_cgdl - Jv_cgdl_cgdl[0]) / (Hgdl / n_gdl + Hcl/2) + Sv_cgdl[1]) / (epsilon_gdl * (1 - x['s_cgdl_1']))
            dif['dC_O2_cgdl_1 / dt'] = (J_O2_ccl_cgdl - J_O2_cgdl_cgdl[0]) / (Hgdl / n_gdl + Hcl/2)/ (epsilon_gdl * (1 - x['s_cgdl_1']))
            dif['ds_cgdl_1 / dt'] = (( Jl_ccl_cgdl- Jl_cgdl_cgdl[0]) / (Hgdl / n_gdl ) + M_H2O * Sl_cgdl[1]) / (rho_H2O * epsilon_gdl)
        elif i == n_gdl-1:
            dif[f'dC_v_cgdl_{n_gdl} / dt'] = ((Jv_cgdl_cgdl[n_gdl - 2]- Jv_cgdl_cgc) / (Hgdl / n_gdl + Hgc/2) + Sv_cgdl[-1]) / (epsilon_gdl * (1 - x[f's_cgdl_{n_gdl}']))
            dif[f'dC_O2_cgdl_{n_gdl} / dt'] = (J_O2_cgdl_cgdl[n_gdl - 2] - J_O2_cgdl_cgc) / (Hgdl / n_gdl + Hgc/2) / (epsilon_gdl * (1 - x[f's_cgdl_{n_gdl}']))
            dif[f'ds_cgdl_{n_gdl} / dt'] = 0
        else:
            dif[f'dC_v_cgdl_{i+1} / dt'] = ((Jv_cgdl_cgdl[i - 1] - Jv_cgdl_cgdl[i]) / (Hgdl / n_gdl) + Sv_cgdl[i+1]) / (epsilon_gdl * (1 - x[f's_cgdl_{i+1}']))
            dif[f'dC_O2_cgdl_{i+1} / dt'] = (J_O2_cgdl_cgdl[i - 1] - J_O2_cgdl_cgdl[i]) / (Hgdl / n_gdl) / (epsilon_gdl * (1 - x[f's_cgdl_{i+1}']))
            dif[f'ds_cgdl_{i+1} / dt'] = ((Jl_cgdl_cgdl[i - 1] - Jl_cgdl_cgdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_cgdl[i+1]) / (rho_H2O* epsilon_gdl)


def dxdt_CCL(dif,x, Jv_ccl_cgdl,  J_O2_ccl_cgdl, Jl_ccl_cgdl, S_sorp_ccl,  S_O2_ccl, Sl_ccl,
                         Hcl, Hgdl, n_gdl, epsilon_cl,  Tfc, rho_H2O, **kwargs):
    
    dif['ds_ccl / dt'] = (- Jl_ccl_cgdl /  (Hgdl / n_gdl + Hcl/2) + M_H2O * Sl_ccl) / (rho_H2O * epsilon_cl)
    dif['dC_v_ccl / dt'] = (- Jv_ccl_cgdl /  (Hgdl / n_gdl + Hcl/2) - S_sorp_ccl - Sl_ccl) /(epsilon_cl * (1 - x['s_ccl']))
    dif['dC_O2_ccl / dt'] = ((- J_O2_ccl_cgdl) /  (Hgdl / n_gdl + Hcl/2)  + S_O2_ccl) /(epsilon_cl * (1 - x['s_ccl']))


def dxdt_MEM(dif, x, epsilon_mc, Hcl, Hmem,n_mem, Ucell, Tfc,
                         J_lambda_mem_acl, J_lambda_mem_ccl, J_O2_mem, J_H2_mem, J_lambda_mem, J_O2_mem_ccl, J_H2_acl_mem,
                         Sp_ccl, S_sorp_acl, S_sorp_ccl, **kwargs):

    # MEM dynamics 
    dif['dlambda_acl / dt']         = M_eq / (rho_mem * epsilon_mc) * (-J_lambda_mem_acl /  (Hmem/n_mem + Hcl) + S_sorp_acl)
    dif['dlambda_ccl / dt']         = M_eq / (rho_mem * epsilon_mc) * (J_lambda_mem_ccl /  (Hmem/n_mem + Hcl) + S_sorp_ccl + Sp_ccl)
    dif['dlambda_mem_1 / dt'] = M_eq / rho_mem * (J_lambda_mem_acl - J_lambda_mem[0]) / (Hmem/n_mem + Hcl)
    for i in range(2, n_mem):
        dif[f'dlambda_mem_{i} / dt'] = M_eq / rho_mem * (J_lambda_mem[i-2] - J_lambda_mem[i-1]) / (Hmem/n_mem)
    dif[f'dlambda_mem_{n_mem} / dt'] = M_eq / rho_mem * (J_lambda_mem[-1] - J_lambda_mem_ccl) / (Hmem/n_mem + Hcl)
    # #Crossover fluxes of O2 and H2 in the membrane
    # dif[f'dC_H2_mem_{1} / dt'] = (J_H2_acl_mem - J_H2_mem[0]) / (Hmem / n_mem+Hcl)
    # dif[f'dC_O2_mem_{n_mem} / dt'] = (J_O2_mem[n_mem - 1] - J_O2_mem_ccl) / (Hmem / n_mem+ Hcl)
    # for i in np.arange(2, n_mem):
    #     dif[f'dC_O2_mem_{i} / dt'] = (J_O2_mem[i - 1] - J_O2_mem[i]) / (Hmem / n_mem)
    #     dif[f'dC_H2_mem_{i} / dt'] = (J_H2_mem[i - 1] - J_H2_mem[i]) / (Hmem / n_mem) # + S_Pt2_mem[i - 1]
    #     dif[f'dC_Pt2_mem_{i} / dt'] = 0 #((J_Pt2_mem[i-1] - J_Pt2_mem[i]) / (Hmem / n_mem)) / (1 - epsilon_mc)  # + S_Pt2_mem[i - 1]
    # dif[f'dC_H2_mem_{n_mem} / dt'] = (J_H2_mem[- 1] - 0) / (Hmem / n_mem)
    # Thickness degradation
    P_O2_ccl = x[f"C_O2_ccl"] * R * Tfc
    dif['ddelta_mem / dt'] = -20.8 / (0.82 * 1980e3) * flourideReleaseRate(MT=Hmem, U=Ucell, Tmem=Tfc, PO2_ca=P_O2_ccl, Hmem_init=1.2e-5)


def dxdt_U(dif, i_fc, C_O2_ccl, eta_c, Hcl, i0_c_ref, kappa_c, C_scl, f_drop, ECSA, Tfc, **kwargs):
    i0_c = (ECSA * i0_c_ref * (C_O2_ccl / C_O2ref)** kappa_c) * np.exp(Eact / R * (1 / 353 - 1 / Tfc))
    dif['deta_c / dt'] =  1 / (C_scl * Hcl) * ((i_fc) - i0_c * np.exp(f_drop * alpha_c * F / (R * Tfc) * eta_c))


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


def dxdt_TH(dif, Pagc, Pcgc, Abp_a, Abp_c, Tfc, Pa_des, Pc_des, **kwargs):
    # Calculation of the pressure derivative inside the gas channels
    dPagcdt = (dif['dC_v_agc / dt'] + dif['dC_H2_agc / dt']) * R * Tfc
    dPcgcdt = (dif['dC_v_cgc / dt'] + dif['dC_O2_cgc / dt'] + dif['dC_N2 / dt']) * R * Tfc
    # Throttle area evolution inside the anode auxiliaries
    dif['dAbp_a / dt'] = -Kp * (Pa_des - Pagc) + Kd * dPagcdt  # PD controller
    if Abp_a > A_T and dif['dAbp_a / dt'] > 0:  # The throttle area cannot be higher than the maximum value
        dif['dAbp_a / dt'] = 0
    elif Abp_a < 0 and dif['dAbp_a / dt'] < 0:  # The throttle area cannot be lower than 0
        dif['dAbp_a / dt'] = 0
    # Throttle area evolution inside the cathode auxiliaries
    dif['dAbp_c / dt'] = - Kp * (Pc_des - Pcgc) + Kd * dPcgcdt  # PD controller
    if Abp_c > A_T and dif['dAbp_c / dt'] > 0:  # The throttle area cannot be higher than the maximum value
        dif['dAbp_c / dt'] = 0
    elif Abp_c < 0 and dif['dAbp_c / dt'] < 0:  # The throttle area cannot be lower than 0
        dif['dAbp_c / dt'] = 0


def dxdt_PRD(dif, Hmem, n_mem, epsilon_mc, M_Pt0,
                       prd, theta_ccl, kdis, kox, kcdis, kdet, 
                       r_m, prd0, C_Pt2_ccl, J_Pt2_mem, **kwargs):
    """
    Urchaga et al 2015 dr/dt = VmKrdpCpt,avg*exp(-R0/r) - VmKdisCpt,avg*exp(-R0/r)
    """
    drdt = Vm_Pt * krdp * C_Pt2_ccl * np.exp(R0 / r_m) - Vm_Pt * (kdis + kox) * Cpt2_ref * np.exp(R0 / r_m)
    dMdisdt = 4 * np.pi * rho_Pt * trapezoid(y=prd * r_m ** 2 * drdt, x=r_m)
    dMcdisdt = 4 * np.pi * rho_Pt * trapezoid(y=prd * r_m ** 2 * kcdis, x=r_m)
    dfdt = -np.gradient(prd * drdt, r_m) - kdet * prd
    dthetadt = (((kox - kcdis) / GAMMA_max) - (2 * theta_ccl / r_m) * drdt)
    dif['dC_Pt2_ccl / dt'] = -3.33 / M_Pt * (dMdisdt - dMcdisdt) / M_Pt0 # - J_Pt2_mem[-1] / (Hmem / n_mem)/ (1 - epsilon_mc)
    for i in range(0, len(r_m)):
        dif[f"dS_N_ccl_{i+1}"+' / dt'] = dfdt[i]
        dif[f"dtheta_ccl_{i+1}"+' / dt'] = dthetadt[i]

        
