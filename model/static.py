from configuration.initialize import *
from configuration.settings import *
from model.coefficients import *
from scipy.optimize import fsolve

class PEMFC_1D:
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

        # Implement the solution of the PEMFC model here using the coefficients and parameters
        Srxn = i / (2 * F) # mol/s/m^2
        ### Water inlet
        Cv_in_a = Phi_a_des * Csat(Tfc)
        Cv_in_c = Phi_c_des * Csat(Tfc)

        # ------------------------------ Initial guess------------------------------ #
        Jnet = 0
        Jw_ca = Srxn
        Jw_an = 0
        ### First solve the gdl vapor profile
        Cv_cgc = (Jw_ca * Lgc / Hgc + Cv_in_c * Win_c)/Wout_c
        Cv_agc = (Jw_an * Lgc / Hgc + Cv_in_a * Win_a)/Wout_a
        Cv_a_inter = Cv_agc + Jw_an/h_conv(Pa_des, Tfc, Wgc, Hgc)
        Cv_acl = Cv_a_inter + Hgdl/Dc(Pa_des, Tfc) * Jw_an
        Cv_c_inter = Cv_cgc + Jw_ca/h_conv(Pc_des, Tfc, Wgc, Hgc)
        Cv_ccl = Cv_c_inter - Hgdl/Dc(Pc_des, Tfc) * -Jw_ca

        # determine the front position of the saturation in GDL
        if Cv_ccl > Csat(Tfc) and Cv_cgc < Csat(Tfc):
            s_front_cgdl = Hgdl - (Csat(Tfc) - Cv_c_inter) * (Dc(Pc_des, Tfc) / -Jw_ca)
        elif Cv_cgc > Csat(Tfc) and Cv_ccl > Csat(Tfc):
            s_front_cgdl = Hgdl
        else:
            s_front_cgdl = 0

        if Cv_acl > Csat(Tfc) and Cv_agc < Csat(Tfc):
            s_front_agdl = (Csat(Tfc) - Cv_a_inter) * (Dc(Pa_des, Tfc) / Jw_an)
        elif Cv_agc > Csat(Tfc) and Cv_acl > Csat(Tfc):
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
            rhs = (M_H2O * Jw_ca * (s_front_cgdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
            def s_cgdl_func(n):
                return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
            if x < s_front_cgdl:
                solution = fsolve(s_cgdl_func, x0=s_front_cgdl)
            else:
                solution = [0]
            s_cgdl[i_node] = solution[0]
            # AGDL saturation profile
            rhs = (M_H2O * Jw_an * (s_front_agdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
            def s_agdl_func(n):
                return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
            if x > s_front_agdl:
                solution = fsolve(s_agdl_func, x0=s_front_agdl)
            else:
                solution = [0]
            s_agdl[i_node] = solution[0]
            # Vapor concentration profile
            Cv_cgdl[i_node] = Cv_c_inter + (Hgdl-x)/Dc(Pc_des, Tfc) * Jw_ca
            Cv_agdl[i_node] = Cv_a_inter + (x)/Dc(Pa_des, Tfc) * Jw_an
            i_node += 1

        # Water content
        if Cv_ccl > Csat(Tfc):
            lambda_ccl = np.min([14 + 8*s_cgdl[0], 22])
        else:
            lambda_ccl = np.min([lambda_eq(Cv_ccl, 0, Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
        if Cv_acl > Csat(Tfc):
            lambda_acl = np.min([14 + 8*s_agdl[0], 22])
        else:
            lambda_acl = np.min([lambda_eq(Cv_acl, 0, Tfc, 20) + Jw_an * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])

        Klambda = rho_mem/M_eq * Dw(lambda_ccl, 333.15) / (2.5/22 * i / F)
        lambda_mem = [lambda_ccl * np.exp(x/Klambda) for x in np.linspace( -2e-5, 0, 10)]

        Jmem = 2.5/22 * i / F * (lambda_ccl - lambda_acl* np.exp(parameters["Hmem"]/Klambda)) / (np.exp(parameters["Hmem"]/Klambda) - 1)
        
        # ------------------------------ Solution ------------------------------ #
        while abs(Jnet - Jmem) > 1e-4:

            Jnet += 0.01 * (Jmem - Jnet)
            Jw_ca = i / (2 * F) - Jnet
            Jw_an = Jnet
            
            ### First solve the gdl vapor profile
            Cv_cgc = (Jw_ca * Lgc / Hgc + Cv_in_c * Win_c)/Wout_c
            Cv_agc = (Jw_an * Lgc / Hgc + Cv_in_a * Win_a)/Wout_a
            Cv_a_inter = Cv_agc + Jw_an/h_conv(Pa_des, Tfc, Wgc, Hgc)
            Cv_acl = Cv_a_inter + Hgdl/Dc(Pa_des, Tfc) * Jw_an
            Cv_c_inter = Cv_cgc + Jw_ca/h_conv(Pc_des, Tfc, parameters["Wgc"], parameters["Hgc"])
            Cv_ccl = Cv_c_inter - Hgdl/Dc(Pc_des, Tfc) * -Jw_ca

            # determine the front position of the saturation in GDL
            if Cv_ccl > Csat(Tfc) and Cv_cgc < Csat(Tfc):
                s_front_cgdl = Hgdl - (Csat(Tfc) - Cv_c_inter) * (Dc(Pc_des, Tfc) / -Jw_ca)
            elif Cv_cgc > Csat(Tfc) and Cv_ccl > Csat(Tfc):
                s_front_cgdl = Hgdl
            else:
                s_front_cgdl = 0

            if Cv_acl > Csat(Tfc) and Cv_agc < Csat(Tfc):
                s_front_agdl = (Csat(Tfc) - Cv_a_inter) * (Dc(Pa_des, Tfc) / Jw_an)
            elif Cv_agc > Csat(Tfc) and Cv_acl > Csat(Tfc):
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
                rhs = (M_H2O * Jw_ca * (s_front_cgdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
                def s_cgdl_func(n):
                    return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                if x < s_front_cgdl:
                    solution = fsolve(s_cgdl_func, x0=s_front_cgdl)
                else:
                    solution = [0]
                s_cgdl[i_node] = solution[0]
                # AGDL saturation profile
                rhs = (M_H2O * Jw_an * (s_front_agdl-x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
                def s_agdl_func(n):
                    return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                if x > s_front_agdl:
                    solution = fsolve(s_agdl_func, x0=s_front_agdl)
                else:
                    solution = [0]
                s_agdl[i_node] = solution[0]
                # Vapor concentration profile
                Cv_cgdl[i_node] = Cv_c_inter + (Hgdl-x)/Dc(Pc_des, Tfc) * Jw_ca
                Cv_agdl[i_node] = Cv_a_inter + (x)/Dc(Pa_des, Tfc) * Jw_an
                i_node += 1

            # Water content
            if Cv_ccl > Csat(Tfc):
                lambda_ccl = np.min([14 + 8*s_cgdl[0], 22])
            else:
                lambda_ccl = np.min([lambda_eq(Cv_ccl, 0, Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
            if Cv_acl > Csat(Tfc):
                lambda_acl = np.min([14 + 8*s_agdl[0], 22])
            else:
                lambda_acl = np.min([lambda_eq(Cv_acl, 0, Tfc, 20) + Jw_an * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])

            Klambda = rho_mem/M_eq * Dw(lambda_ccl, 333.15) / (2.5/22 * i / F)
            lambda_mem = [lambda_ccl * np.exp(x/Klambda) for x in np.linspace( -2e-5, 0, 10)]

            Jmem = 2.5/22 * i / F * (lambda_ccl - lambda_acl* np.exp(Hmem/Klambda)) / (np.exp(Hmem/Klambda) - 1)

        return {"Jnet": Jnet, "Jmem": Jmem,
                     "lambda_ccl": lambda_ccl, "lambda_acl": lambda_acl,
                     "Cv_ccl": Cv_ccl, "Cv_acl": Cv_acl,
                     "s_front_cgdl": s_front_cgdl, "s_front_agdl": s_front_agdl,
                     "Cv_cgdl": Cv_cgdl, "Cv_agdl": Cv_agdl, "lambda_mem": lambda_mem,
                     "s_cgdl": s_cgdl, "s_agdl": s_agdl}





