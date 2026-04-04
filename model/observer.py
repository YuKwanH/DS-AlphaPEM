from model.parameters import *
from modules.transitory_functions import *
from scipy.optimize import root
import warnings
warnings.filterwarnings("ignore")


class observer:

    def __init__(self):

        gradient = {}
        #"C_O2_ccl","C_O2_cgdl","C_O2_cgc",
        self.state_name = ["lambda_acl","lambda_mem","lambda_ccl", "s_ccl", "C_v_ccl", "s_cgdl", "C_v_cgdl", "C_v_cgc",
                           "C_O2_ccl","C_O2_cgdl","C_O2_cgc","C_N2","Wcp"]
        for name in self.state_name:
            gradient["d" + name + " / dt"] = 0
        self.gradient = gradient
        self.t = 0
        self.dt = 0

        self.Tfc = 328.15
        self.Ucell = 0
        self.Aact = 1.5e-3

        # Control
        self.i_fc = 0.8e4
        self.Pa_des = 1.35e5
        self.Pc_des = 1.0e5
        self.Phi_a_des = 0
        self.Phi_c_des = 1
        self.Phi_hum = 1
        self.Sc = 1.2
        self.Wcp_des = (1 / yO2_ext) * self.Sc * self.i_fc * Aact / (4 * F * Hgc * Wgc)

        self.state = []

        self.S = []
        self.Jv_cgc = []

    def dxdt(self, t, variables):

        self.dt = t - self.t
        self.t = t

        sv = {}
        for i_name in range(len(self.state_name)):
            sv[self.state_name[i_name]] = variables[i_name]

        # Massflow
        # Immediately calculation
        Psat = 101325 * 10 ** (-2.1794 + 0.02953 * (self.Tfc - 273.15) - 9.1837e-5 * (self.Tfc - 273.15) ** 2 +1.4454e-7 * (self.Tfc - 273.15) ** 3)

        self.C_H2 = (self.Pa_des - self.Phi_a_des * Psat) / (R * self.Tfc)
        self.Pcgc = (sv["C_v_cgc"] + sv["C_O2_cgc"] + sv["C_N2"]) * R * self.Tfc
        Phi_cgc = sv["C_v_cgc"] / C_v_sat(self.Tfc)
        yO2_cgc = sv["C_O2_cgc"] / (sv["C_O2_cgc"] + sv["C_N2"])

        self.Pccl =  (sv['C_v_ccl']  + sv["C_O2_ccl"] + sv["C_N2"]) * R * self.Tfc
        self.Pcgdl =  (sv['C_v_cgdl']  + sv["C_O2_cgdl"] + sv["C_N2"]) * R * self.Tfc
        Pccl_cgdl = self.Pccl / 2 + self.Pcgdl / 2
        s_ccl_cgdl = sv["s_ccl"] / 2 + sv['s_cgdl'] / 2
        Pcgdl_cgc = self.Pcgdl / 2 + self.Pcgc / 2

        # Mass flow of vapor at CGC
        Mcgc = (Phi_cgc * Psat / self.Pcgc * M_H2O +
               yO2_cgc * (1 - Phi_cgc * Psat / self.Pcgc) * M_O2 +
               (1 - yO2_cgc) * (1 - Phi_cgc * Psat / self.Pcgc) * M_N2)
        J_O2_in = yO2_ext * sv["Wcp"]
        J_N2_in = (1 - yO2_ext) * sv["Wcp"]
        Jv_c_in = (self.Phi_hum*Psat / (Pext - self.Phi_hum * Psat)) * sv["Wcp"]
        Jc_out = Kem_in * (self.Pcgc - 1e5) / (Hgc * Wgc * Mcgc) # Pressure difference between gas channel with ambient
        Jv_c_out = Phi_cgc * Psat / self.Pcgc * Jc_out
        J_O2_out = yO2_cgc * ((self.Pcgc - Phi_cgc * Psat) / self.Pcgc) * Jc_out
        J_N2_out = (1 - yO2_cgc)  * ((self.Pcgc - Phi_cgc * Psat) / self.Pcgc) * Jc_out
        Jv_cgdl_cgc = h_c(Pcgdl_cgc, self.Tfc, Wgc, Hgc) * (sv['C_v_cgdl'] - sv['C_v_cgc'])
        J_O2_cgdl_cgc = h_c(Pcgdl_cgc, self.Tfc, Wgc, Hgc) * (sv['C_O2_cgdl'] - sv["C_O2_cgc"])

        # Mass flow of vapor at CGDL
        epsilon_mean = epsilon_gdl / 2 + epsilon_cl / 2
        theta_c_mean = theta_c_gdl / 2 + theta_c_cl / 2
        Jv_ccl_cgdl = -  Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, self.Tfc, epsilon_c, epsilon_gdl) * (sv['C_v_cgdl'] - sv['C_v_ccl']) / (Hgdl + Hcl)
        J_O2_ccl_cgdl = -  Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, self.Tfc, epsilon_c, epsilon_gdl) * (sv['C_O2_cgdl'] - sv["C_O2_ccl"]) / (Hgdl + Hcl)
        Jl_ccl_cgdl = - 2 * sigma(self.Tfc) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l(self.Tfc) * abs(np.cos(theta_c_mean)) * \
                      (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * (s_ccl_cgdl ** e + 1e-7) * \
                      (1.417 - 4.24 * s_ccl_cgdl + 3.789 * s_ccl_cgdl ** 2) * (sv['s_cgdl'] - sv['s_ccl']) / (Hgdl + Hcl)
        Sl_cgdl = Svl(sv['s_cgdl'], sv['C_v_cgdl'], sv['C_v_cgdl'] + sv["C_O2_cgdl"] + sv["C_N2"], epsilon_gdl, self.Tfc, gamma_cond, gamma_evap)
        Sv_cgdl = -Sl_cgdl

        # Mass flow at CCL
        J_lambda_mem_ccl = 2.5 / 22 * self.i_fc / F * sv["lambda_ccl"] - 2 * rho_mem / M_eq * D(sv["lambda_ccl"]) * (sv["lambda_ccl"]-sv["lambda_mem"]) / (Hmem + Hcl)
        S_sorp_ccl = gamma_sorp(sv['C_v_ccl'], sv['s_ccl'], sv['lambda_ccl'], self.Tfc, Hcl, Kshape) * rho_mem / M_eq * \
                                (lambda_eq(sv['C_v_ccl'], sv['s_ccl'], self.Tfc, Kshape) - sv['lambda_ccl'])
        Sp_ccl = self.i_fc / (2 * F)
        Sl_ccl = Svl(sv['s_ccl'], sv['C_v_ccl'], sv['C_v_ccl'] + sv["C_O2_ccl"] + sv["C_N2"], epsilon_cl, self.Tfc, gamma_cond, gamma_evap)
        Sv_ccl = - Sl_ccl
        S_O2_ccl = - self.i_fc / (4 * F)

        # Mass flow at anode
        J_lambda_mem_acl = 2.5 / 22 * self.i_fc / F * sv["lambda_acl"] - 2 * rho_mem / M_eq * D(sv["lambda_acl"]) * (sv["lambda_mem"] - sv["lambda_acl"]) / (Hmem + Hcl)
        Sp_acl = 0
        S_sorp_acl = gamma_sorp(0, 0, sv["lambda_acl"], self.Tfc, Hcl, Kshape) * rho_mem / M_eq * (lambda_eq(0, 0, self.Tfc, Kshape) - sv["lambda_acl"])

        # State variables
        self.gradient['dlambda_acl / dt'] = M_eq / (rho_mem * epsilon_mc) * (-J_lambda_mem_acl / Hcl + S_sorp_acl + Sp_acl)
        self.gradient['dlambda_mem / dt'] = M_eq / rho_mem * (J_lambda_mem_acl - J_lambda_mem_ccl) / Hmem
        self.gradient['dlambda_ccl / dt'] = M_eq / (rho_mem * epsilon_mc) * (J_lambda_mem_ccl / Hcl + S_sorp_ccl + Sp_ccl)
        self.gradient['ds_ccl / dt'] = 1 / (rho_H2O(self.Tfc) * epsilon_cl) * (- Jl_ccl_cgdl / Hcl + M_H2O * Sl_ccl)
        self.gradient['dC_v_ccl / dt'] = 1 / (epsilon_cl * (1 - sv['s_ccl'])) * (- Jv_ccl_cgdl / Hcl - S_sorp_ccl + Sv_ccl)
        self.gradient['ds_cgdl / dt'] = 1 / (rho_H2O(self.Tfc) * epsilon_gdl) * ((Jl_ccl_cgdl) / Hgdl + M_H2O * Sl_cgdl)
        self.gradient['dC_v_cgdl / dt'] = 1 / (epsilon_gdl * (1 - sv['s_cgdl'])) * ((Jv_ccl_cgdl - Jv_cgdl_cgc) /Hgdl + Sv_cgdl)
        self.gradient['dC_v_cgc / dt'] = (Jv_c_in - Jv_c_out) / Lgc + Jv_cgdl_cgc / Hgc
        self.gradient['dC_O2_ccl / dt'] = 1 / (epsilon_cl * (1 - sv['s_ccl'])) * (-J_O2_ccl_cgdl / Hcl + S_O2_ccl)
        self.gradient['dC_O2_cgdl / dt'] = 1 / (epsilon_gdl * (1 - sv['s_cgdl'])) * (J_O2_ccl_cgdl - J_O2_cgdl_cgc) / Hgdl
        self.gradient['dC_O2_cgc / dt'] = (J_O2_in - J_O2_out) / Lgc + J_O2_cgdl_cgc / Hgc
        self.gradient['dC_N2 / dt'] = (J_N2_in - J_N2_out) / Lgc

        # Control variables
        self.gradient['dWcp / dt'] = (self.Wcp_des - sv["Wcp"]) / tau_cp
        # Observation
        self.Ucell = self.getUcell(sv['s_ccl'],sv['lambda_ccl'],sv['lambda_mem'],sv['C_O2_ccl'])

        self.S.append([S_sorp_ccl,Sp_ccl,Sl_ccl,J_lambda_mem_ccl])
        self.Jv_cgc.append([Jv_c_in/Lgc, Jv_c_out/Lgc,Jv_cgdl_cgc / Hgc])

        return list(self.gradient.values())

    def getUcell(self, s_ccl, lambda_ccl, lambda_mem, C_O2_ccl):

        Re = 5.70e-7  # ohm.m². It is the electron conduction resistance of the circuit.
        i0_c_ref = 2.49  # A.m-2.It is the reference exchange current density at the cathode.
        kappa_c = 1.61  # It is the overpotential correction exponent.

        # The equilibrium potential
        Ueq = (E0 - 8.5e-4 * (self.Tfc - 298.15) + R * self.Tfc / (2 * F) * (np.log(R * self.Tfc * self.C_H2 / Pref) + 0.5 * np.log(R * self.Tfc * C_O2_ccl / Pref)))

        # The proton resistance
        # The proton resistance at the membrane : Rmem
        if lambda_mem >= 1:
            Rmem = Hmem / ((0.5139 * lambda_mem - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc)))
        else:
            Rmem = Hmem / (0.1879 * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc)))
        #  The proton resistance at the cathode catalyst layer : Rccl
        if lambda_ccl >= 1:
            Rccl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_ccl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc)))
        else:
            Rccl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / self.Tfc)))

        #  The total proton resistance
        Rp = Rmem + Rccl  # its value is around [4-7]e-6 ohm.m².
        slim = a_slim * (2e5 / 1e5) + b_slim
        s_switch = a_switch * slim
        f_drop = 0.5 * (1.0 - np.tanh((4 * s_ccl - 2 * slim - 2 * s_switch) / (slim - s_switch)))
        eta_c = 1 / f_drop * R * self.Tfc / (alpha_c * F) * np.log(self.i_fc / i0_c_ref * (C_O2ref / C_O2_ccl) ** kappa_c)
        # The cell voltage
        return Ueq - self.i_fc * (Rp + Re) - eta_c

    def Y(self):
        return [self.Ucell]
