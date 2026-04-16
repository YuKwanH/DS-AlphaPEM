from configuration.initialize import *
from configuration.settings import *
from model.coefficients import *
from scipy.optimize import fsolve

class PEMFC_stat:
    def __init__(self, parameters, operation_inputs):
        self.parameters = parameters
        self.operation_inputs = operation_inputs
    
    def solve(self, i):

        Phi_a_des, Phi_c_des, Tfc = self.operation_inputs['Phi_a_des'], self.operation_inputs['Phi_c_des'], self.operation_inputs['Tfc']
        Pa_des, Pc_des = self.operation_inputs['Pa_des'], self.operation_inputs['Pc_des']
        Win_c, Wout_c = self.operation_inputs['Win_c'], self.operation_inputs['Wout_c']
        Win_a, Wout_a = self.operation_inputs['Win_a'], self.operation_inputs['Wout_a']
        Hcl, Hgdl, Hmem = self.parameters['Hcl'], self.parameters['Hgdl'], self.parameters['Hmem']
        Lgc, Wgc, Hgc = self.parameters['Lgc'], self.parameters['Wgc'], self.parameters['Hgc']
        epsilon_gdl, epsilon_cl, epsilon_c = self.parameters["epsilon_gdl"], self.parameters["epsilon_cl"], self.parameters["epsilon_c"]

        # ------------------------------  Boundary conditions ------------------------------ #
        Srxn = i / (2 * F) # mol/s/m^2
        # Water inlet
        Cv_in_a = Phi_a_des * C_v_sat(Tfc)
        Cv_in_c = Phi_c_des * C_v_sat(Tfc)
        #  Initial guess
        Jnet = 0
        Jw_ca = Srxn
        Jw_an = 0
        # The GDL vapor boundary conditions
        Cv_cgc = (Jw_ca * Lgc / Hgc + Cv_in_c * Win_c)/Wout_c
        Cv_agc = (Jw_an * Lgc / Hgc + Cv_in_a * Win_a)/Wout_a
        Cv_a_inter = Cv_agc + Jw_an/h_a(Pa_des, Tfc, Wgc, Hgc)
        Cv_acl = Cv_a_inter + Hgdl/Da_eff(0,epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) * Jw_an
        Cv_c_inter = Cv_cgc + Jw_ca/h_c(Pa_des, Tfc, Wgc, Hgc)
        Cv_ccl = Cv_c_inter - Hgdl/Dc_eff(0,epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) * -Jw_ca

        # determine the front position of the saturation in GDL
        if Cv_ccl > C_v_sat(Tfc) and Cv_cgc < C_v_sat(Tfc):
            s_front_cgdl = Hgdl - (C_v_sat(Tfc) - Cv_c_inter) * (Dc_eff(0,epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) / -Jw_ca)
        elif Cv_cgc > C_v_sat(Tfc) and Cv_ccl > C_v_sat(Tfc):
            s_front_cgdl = Hgdl
        else:
            s_front_cgdl = 0

        if Cv_acl > C_v_sat(Tfc) and Cv_agc < C_v_sat(Tfc):
            s_front_agdl = (C_v_sat(Tfc) - Cv_a_inter) * (Da_eff(0,epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) / Jw_an)
        elif Cv_agc > C_v_sat(Tfc) and Cv_acl > C_v_sat(Tfc):
            s_front_agdl = 0
        else:
            s_front_agdl = Hgdl

        # GDL profile
        s_cgdl = np.zeros(10)
        s_agdl = np.zeros(10)
        Cv_cgdl = np.zeros(10)
        Cv_agdl = np.zeros(10)
        i_node = 0
        for x in np.linspace(0, Hgdl, 10):
            # CGDL saturation profile
            rhs = (M_H2O * Jw_ca * (s_front_cgdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c,epsilon_gdl))**0.5)
            def s_cgdl_func(n):
                return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
            if x < s_front_cgdl:
                solution = fsolve(s_cgdl_func, x0=s_front_cgdl)
            else:
                solution = [0]
            s_cgdl[i_node] = solution[0]
            # AGDL saturation profile
            rhs = (M_H2O * Jw_an * (s_front_agdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c,epsilon_gdl))**0.5)
            def s_agdl_func(n):
                return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
            if x > s_front_agdl:
                solution = fsolve(s_agdl_func, x0=s_front_agdl)
            else:
                solution = [0]
            s_agdl[i_node] = solution[0]
            # Vapor concentration profile
            Cv_cgdl[i_node] = Cv_c_inter + (Hgdl-x)/Dc_eff(0,epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) * Jw_ca
            Cv_agdl[i_node] = Cv_a_inter + (x)/Da_eff(0,epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) * Jw_an
            i_node += 1

        Cv_agdl = np.minimum(Cv_agdl, C_v_sat(Tfc))
        Cv_cgdl = np.minimum(Cv_cgdl, C_v_sat(Tfc))

        # Water content
        if Cv_ccl > C_v_sat(Tfc):
            lambda_ccl = np.min([14 + 8*s_cgdl[0], 22])
        else:
            lambda_ccl = np.min([lambda_eq(Cv_ccl, 0, Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
        if Cv_acl > C_v_sat(Tfc):
            lambda_acl = np.min([14 + 8*s_agdl[0], 22])
        else:
            lambda_acl = np.min([lambda_eq(Cv_acl, 0, Tfc, 20) + Jw_an * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])

        Klambda = rho_mem/M_eq * Dw(lambda_ccl, 333.15) / (2.5/22 * i / F)
        lambda_mem = [lambda_ccl * np.exp(x/Klambda) for x in np.linspace( -2e-5, 0, 10)]

        Jmem = 2.5/22 * i / F * (lambda_ccl - lambda_acl* np.exp(parameters["Hmem"]/Klambda)) / (np.exp(parameters["Hmem"]/Klambda) - 1)
        n_iter = 0
        Ksearch = 0.01
        # ------------------------------ Solution ------------------------------ #
        while abs(Jnet - Jmem) > 1e-5:

            n_iter += 1
            if n_iter == 1000:
                Ksearch *= 0.5
            if n_iter ==3000:
                Ksearch *= 0.5
            if n_iter > 5000:
                break
            Jnet += Ksearch * (Jmem - Jnet)
            Jw_ca = i / (2 * F) - Jnet
            Jw_an = Jnet
            
            ### First solve the gdl vapor profile
            Cv_cgc = (Jw_ca * Lgc / Hgc + Cv_in_c * Win_c)/Wout_c
            Cv_agc = (Jw_an * Lgc / Hgc + Cv_in_a * Win_a)/Wout_a
            Cv_a_inter = Cv_agc + Jw_an/h_a(Pa_des, Tfc, Wgc, Hgc)
            Cv_acl = Cv_a_inter + Hgdl/Da_eff(np.mean(s_agdl),epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) * Jw_an
            Cv_c_inter = Cv_cgc + Jw_ca/h_c(Pc_des, Tfc, Wgc, Hgc)
            Cv_ccl = Cv_c_inter - Hgdl/Dc_eff(np.mean(s_cgdl),epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) * -Jw_ca

            # determine the front position of the saturation in GDL
            if Cv_ccl > C_v_sat(Tfc) and Cv_cgc < C_v_sat(Tfc):
                s_front_cgdl = Hgdl - (C_v_sat(Tfc) - Cv_c_inter) * (Dc_eff(np.mean(s_cgdl),epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) / -Jw_ca)
            elif Cv_cgc > C_v_sat(Tfc) and Cv_ccl > C_v_sat(Tfc):
                s_front_cgdl = Hgdl
            else:
                s_front_cgdl = 0

            if Cv_acl > C_v_sat(Tfc) and Cv_agc < C_v_sat(Tfc):
                s_front_agdl = (C_v_sat(Tfc) - Cv_a_inter) * (Da_eff(np.mean(s_agdl),epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) / Jw_an)
            elif Cv_agc > C_v_sat(Tfc) and Cv_acl > C_v_sat(Tfc):
                s_front_agdl = 0
            else:
                s_front_agdl = Hgdl

            # GDL profile
            s_cgdl = np.zeros(10)
            s_agdl = np.zeros(10)
            Cv_cgdl = np.zeros(10)
            Cv_agdl = np.zeros(10)
            i_node = 0
            for x in np.linspace(0, Hgdl, 10):
                # CGDL saturation profile
                rhs = (M_H2O * Jw_ca * (s_front_cgdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c,epsilon_gdl))**0.5)
                def s_cgdl_func(n):
                    return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                if x < s_front_cgdl:
                    solution = fsolve(s_cgdl_func, x0=s_front_cgdl)
                else:
                    solution = [0]
                s_cgdl[i_node] = solution[0]
                # AGDL saturation profile
                rhs = (M_H2O * Jw_an * (s_front_agdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c,epsilon_gdl))**0.5)
                def s_agdl_func(n):
                    return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                if x > s_front_agdl:
                    solution = fsolve(s_agdl_func, x0=s_front_agdl)
                else:
                    solution = [0]
                s_agdl[i_node] = solution[0]
                # Vapor concentration profile
                Cv_cgdl[i_node] = Cv_c_inter + (Hgdl-x)/Dc_eff(np.mean(s_cgdl),epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) * Jw_ca
                Cv_agdl[i_node] = Cv_a_inter + (x)/Da_eff(np.mean(s_agdl),epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) * Jw_an
                i_node += 1
            Cv_agdl = np.minimum(Cv_agdl, C_v_sat(Tfc))
            Cv_cgdl = np.minimum(Cv_cgdl, C_v_sat(Tfc))

            # Water content
            if Cv_ccl > C_v_sat(Tfc):
                lambda_ccl = np.min([14 + 8*s_cgdl[0], 22])
            else:
                lambda_ccl = np.min([lambda_eq(Cv_ccl, 0, Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
            if Cv_acl > C_v_sat(Tfc):
                lambda_acl = np.min([14 + 8*s_agdl[0], 22])
            else:
                lambda_acl = np.min([lambda_eq(Cv_acl, 0, Tfc, 20) + Jw_an * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])

            Klambda = rho_mem/M_eq * Dw(lambda_ccl, 333.15) / (2.5/22 * i / F)
            lambda_mem = [lambda_ccl * np.exp(x/Klambda) for x in np.linspace( -2e-5, 0, 10)]

            Jmem = 2.5/22 * i / F * (lambda_ccl - lambda_acl* np.exp(Hmem/Klambda)) / (np.exp(Hmem/Klambda) - 1)
        
        JH2 = -Srxn
        JO2 = -Srxn/2
        C_O2_cgdl = np.zeros(10)
        C_H2_agdl = np.zeros(10)
        C_H2_agc = Pa_des / (R * Tfc) - Cv_acl
        C_O2_cgc = (Pc_des / (R * Tfc) - Cv_ccl) * 0.21
        C_H2_inter = C_H2_agc + JH2/h_a(Pa_des, Tfc, Wgc, Hgc)
        C_O2_inter = C_O2_cgc + JO2/h_c(Pc_des, Tfc, Wgc, Hgc)
        i_node = 0
        for x in np.linspace(0, Hgdl, 10):
            C_O2_cgdl[i_node] = C_O2_inter + (Hgdl-x)/Dc_eff(s_cgdl[i_node],epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) * JO2
            C_H2_agdl[i_node] = C_H2_inter + (x)/Da_eff(s_agdl[i_node],epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) * JH2
            i_node += 1

        C_H2_acl = C_H2_agdl[-1] + Hcl/Da_eff(np.mean(s_agdl),epsilon_cl,Pa_des, Tfc,epsilon_c,epsilon_cl) * JH2
        C_O2_ccl = C_O2_cgdl[0] + Hcl/Dc_eff(np.mean(s_cgdl),epsilon_cl,Pc_des, Tfc,epsilon_c,epsilon_cl) * -JO2
        
        Ueq = (E0 - 8.5e-4 * (Tfc - 298.15) + R * Tfc / (2 * F) * (np.log(R * Tfc * C_H2_acl / Pref) + 0.5 * np.log(R * Tfc * C_O2_ccl / Pref)))
        f_drop = 0.5 * (1.0 - np.tanh((4 * s_cgdl[0] - 2 * slim - 2 * s_switch) / (slim - s_switch)))
        i0_c = i0_c_ref * np.exp(-Eact / R * (1 / Tfc - 1 / 353))
        eta_c = (1 / f_drop * R * Tfc / (alpha_c * F) * np.log((i) / i0_c * (C_O2ref / C_O2_ccl) ** kappa_c))
        Rmem = []

        for i_mem in range(10):
            # The proton resistance
            lambda_mem_i = lambda_mem[i_mem]
            if lambda_mem_i >= 1:
                Rmem += [(Hmem/parameters['n_mem']) / ((0.5139 * lambda_mem_i - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tfc)))]
            else:
                Rmem += [(Hmem/parameters['n_mem']) / (0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tfc)))]
        Rohm = sum(Rmem) + parameters["Re"]

        return {"Jnet": Jnet, "Jmem": Jmem,
                     "lambda_ccl": lambda_ccl, "lambda_acl": lambda_acl,"lambda_mem": lambda_mem,
                     "C_v_ccl": Cv_ccl, "C_v_acl": Cv_acl,"C_v_cgdl": Cv_cgdl, "C_v_agdl": Cv_agdl, 
                     "C_v_a_inter": Cv_a_inter, "C_v_c_inter": Cv_c_inter, "C_v_cgc": Cv_cgc, "C_v_agc": Cv_agc,
                     "s_front_cgdl": s_front_cgdl, "s_front_agdl": s_front_agdl,"s_cgdl": s_cgdl, "s_agdl": s_agdl,
                     "C_H2_acl": C_H2_acl, "C_O2_ccl": C_O2_ccl,
                     "C_H2_agc": C_H2_agc, "C_O2_cgc": C_O2_cgc,
                     "C_H2_inter": C_H2_inter, "C_O2_inter": C_O2_inter,
                     "Ueq": Ueq, "eta_c": eta_c, "Rohm": Rohm,
                     "Jw_ca":Jw_ca, "Jw_an": Jw_an, "JH2": JH2, "JO2": JO2,
                     "Jv_a_in": Win_a * Cv_in_a/Lgc, "Jv_a_out": Wout_a * Cv_agc/Lgc,
                     "Jv_c_in": Win_c * Cv_in_c/Lgc, "Jv_c_out": Wout_c * Cv_cgc/Lgc,}





