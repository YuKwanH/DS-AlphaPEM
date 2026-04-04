from model.Observer import observer
from scipy.optimize import root
import numpy as np



class observer_extension(observer):
    def __init__(self):
        super().__init__()
    def jac(self, x, eps):
        H = []
        F = []
        for i in range(len(x)):
            eps_array = np.zeros(len(x))
            eps_array[i] = eps
            H.append((self.h(x + eps_array) - self.h(x - eps_array)) / (2*eps))
            F.append((self.f(x + eps_array) - self.f(x - eps_array)) / (2*eps))
        return F, H

    def f(self,x):
        sol = root(lambda x:np.array(x) - np.array(self.state) - self.dt * np.array(self.dxdt(self.t, x)), x, tol=1e-6)
        return sol.x

    def h(self,x):
        return np.array([self.getUcell(x[3], x[2], x[1])])


    def H_analytic(self, x):

        slim = a_slim * (self.Pc_des / 1e5) + b_slim
        s_switch = a_switch * slim
        kR_mem = 0.5139 * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc))
        bR_mem = 0.326 * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc))
        kR_ccl = 0.5139 * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc)) * (epsilon_mc ** tau)
        bR_ccl = 0.326 * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc)) * (epsilon_mc ** tau)
        kdrop_s = 4 / (slim - s_switch)
        bdrop_s = 2 * (slim + s_switch) / (slim - s_switch)
        kEta_ccl = kdrop_s * R * self.Tfc / (alpha_c * F) * np.log(self.i_fc / i0_c_ref * (C_O2ref / self.C_O2) ** kappa_c)

        if x[1] > 1.0 and x[2] > 0.0:
            return np.array([0, self.i_fc * kR_mem * Hmem / (kR_mem * x[1] - bR_mem) ** 2,
                             self.i_fc * kR_ccl * Hcl / (kR_ccl * x[2] - bR_ccl) ** 2,
                             -2 * kEta_ccl * np.exp(2 * (kdrop_s * x[3] - bdrop_s)), 0, 0, 0, 0])
        elif x[1] > 0.0 and x[2] < 1.0:
            return np.array([0, self.i_fc * kR_mem * Hmem / (kR_mem * x[1] - bR_mem) ** 2, 0,
                             -2 * kEta_ccl * np.exp(2 * (kdrop_s * x[3] - bdrop_s)), 0, 0, 0, 0])
        elif x[1] < 1.0 and x[2] > 1.0:
            return np.array([0, 0, self.i_fc * kR_ccl * Hcl / (kR_ccl * x[2] - bR_ccl) ** 2,
                             -2 * kEta_ccl * np.exp(2 * (kdrop_s * x[3] - bdrop_s)), 0, 0, 0, 0])
        else:
            return np.array([0, 0, 0, -2 * kEta_ccl * np.exp(2 * (kdrop_s * x[3] - bdrop_s)), 0, 0, 0, 0])

    def F(self, variables):

        sv = {}
        for i_name in range(len(self.state_name)):
            sv[self.state_name[i_name]] = variables[i_name]

        # Massflow
        # Immediately calculation
        Psat = 101325 * 10 ** (-2.1794 + 0.02953 * (self.Tfc - 273.15) - 9.1837e-5 * (self.Tfc - 273.15) ** 2 +1.4454e-7 * (self.Tfc - 273.15) ** 3)
        self.C_H2 = (self.Pa_des - self.Phi_a_des * Psat) / (R * self.Tfc)
        self.C_O2 = yO2_ext * (self.Pc_des - self.Phi_c_des * Psat) / (R * self.Tfc)
        self.C_N2 = (1 - yO2_ext) * (self.Pc_des - self.Phi_c_des * Psat) / (R * self.Tfc)
        self.Pccl =  (sv['C_v_ccl']  + self.C_O2 + C_N2) * R * self.Tfc
        self.Pcgdl = (sv['C_v_cgdl'] + self.C_O2 + C_N2) * R * self.Tfc
        self.Pcgc =  (sv["C_v_cgc"]  + self.C_O2 + C_N2) * R * self.Tfc
        Phi_cgc = sv["C_v_cgc"] / C_v_sat(self.Tfc)
        y_cgc = self.C_O2 / (self.C_O2 + C_N2)
        Pcsm = self.Pc_des
        s_ccl_cgdl = sv["s_ccl"] / 2 + sv['s_cgdl'] / 2
        Pccl_cgdl = self.Pccl / 2 + self.Pcgdl / 2
        self.Pcgdl = (sv['C_v_cgdl'] + self.C_O2 + C_N2) * R * self.Tfc
        Pcgdl_cgc = self.Pcgdl / 2 + self.Pcgc / 2

        # Cathode inlet auxiliary
        Mcgc = Phi_cgc * Psat / self.Pcgc * M_H2O + y_cgc * (1 - Phi_cgc * Psat / self.Pcgc) * M_O2 + (1 - y_cgc) * (1 - Phi_cgc * Psat / self.Pcgc) * M_N2
        Jc_in = (1 + self.Phi_c_des * Psat / (Pcsm - self.Phi_c_des * Psat)) * 1 / yO2_ext * self.Sc * self.i_fc / (4 * F) * self.Aact / (Hgc * Wgc)
        Jc_out = Kem_in * (self.Pcgc - self.Pc_des) / (Hgc * Wgc * Mcgc)
        # Mass flow of vapor at CGC
        Jv_c_in = Phi_csm * Psat / Pcsm * Jc_in
        Jv_c_out = Phi_cgc * Psat / self.Pcgc * Jc_out
        Jv_cgdl_cgc = h_c(Pcgdl_cgc, self.Tfc, Wgc, Hgc) * (sv['C_v_cgdl'] - sv['C_v_cgc'])
        # Mass flow of vapor at CGDL
        epsilon_mean = epsilon_gdl / 2 + epsilon_cl / 2
        theta_c_mean = theta_c_gdl / 2 + theta_c_cl / 2
        Jv_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, self.Tfc, epsilon_c, epsilon_gdl) * \
                      (sv['C_v_cgdl'] - sv['C_v_ccl']) / (Hgdl + Hcl)
        Sl_cgdl = Svl(sv['s_cgdl'], sv['C_v_cgdl'], sv['C_v_cgdl'] + self.C_O2 + C_N2,epsilon_gdl, self.Tfc, gamma_cond, gamma_evap)
        Sv_cgdl = -Sl_cgdl
        # Mass flow at CCL
        J_lambda_mem_ccl = 2.5 / 22 * self.i_fc / F * sv["lambda_ccl"] - 2 * rho_mem / M_eq * D(sv["lambda_ccl"]) * (sv["lambda_ccl"]-sv["lambda_mem"]) / (Hmem + Hcl)
        Jl_ccl_cgdl = - 2 * sigma(self.Tfc) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l(self.Tfc) * abs(np.cos(theta_c_mean)) * \
                      (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * (s_ccl_cgdl ** e + 1e-7) * \
                      (1.417 - 4.24 * s_ccl_cgdl + 3.789 * s_ccl_cgdl ** 2) * (sv['s_cgdl'] - sv['s_ccl']) / (Hgdl  + Hcl)
        S_sorp_ccl = gamma_sorp(sv['C_v_ccl'], sv['s_ccl'], sv['lambda_ccl'], self.Tfc, Hcl, Kshape) * rho_mem / M_eq * \
                     (lambda_eq(sv['C_v_ccl'], sv['s_ccl'], self.Tfc, Kshape) - sv['lambda_ccl'])
        Sp_ccl = self.i_fc / (2 * F * Hcl)
        Sl_ccl = Svl(sv['s_ccl'], sv['C_v_ccl'], sv['C_v_ccl'] + self.C_O2 + C_N2, epsilon_cl, self.Tfc, gamma_cond, gamma_evap)
        Sv_ccl = - Sl_ccl
        # Mass flow at anode
        J_lambda_mem_acl=2.5 / 22 * self.i_fc / F * sv["lambda_acl"] - 2 * rho_mem / M_eq * D(sv["lambda_acl"]) * (sv["lambda_mem"] - sv["lambda_acl"]) / (Hmem + Hcl)
        Sp_acl = 0
        S_sorp_acl = gamma_sorp(0, 0, sv["lambda_acl"], self.Tfc, Hcl, Kshape) * rho_mem / M_eq * (lambda_eq(0, 0, self.Tfc, Kshape) - sv["lambda_acl"])


        self.gradient['dlambda_acl / dt'] = M_eq / (rho_mem * epsilon_mc) * (-J_lambda_mem_acl / Hcl + S_sorp_acl + Sp_acl)
        self.gradient['dlambda_mem / dt'] = M_eq / rho_mem * (J_lambda_mem_acl - J_lambda_mem_ccl) / Hmem
        self.gradient['dlambda_ccl / dt'] = M_eq / (rho_mem * epsilon_mc) * (J_lambda_mem_ccl / Hcl + S_sorp_ccl + Sp_ccl)
        self.gradient['ds_ccl / dt'] = 1 / (rho_H2O(self.Tfc) * epsilon_cl) * (- Jl_ccl_cgdl / Hcl + M_H2O * Sl_ccl)
        self.gradient['dC_v_ccl / dt'] = 1 / (epsilon_cl * (1 - sv['s_ccl'])) * (- Jv_ccl_cgdl / Hcl - S_sorp_ccl + Sv_ccl)
        self.gradient['ds_cgdl / dt'] = 1 / (rho_H2O(self.Tfc) * epsilon_gdl) * ((Jl_ccl_cgdl) / Hgdl + M_H2O * Sl_cgdl)
        self.gradient['dC_v_cgdl / dt'] = 1 / (epsilon_gdl * (1 - sv['s_cgdl'])) * ((Jv_ccl_cgdl - Jv_cgdl_cgc) /Hgdl + Sv_cgdl)
        self.gradient['dC_v_cgc / dt'] = (Jv_c_in - Jv_c_out) / Lgc + Jv_cgdl_cgc / Hgc

        self.Ucell = self.getUcell(sv['s_ccl'],sv['lambda_ccl'],sv['lambda_mem'])

        return list(self.gradient.values())