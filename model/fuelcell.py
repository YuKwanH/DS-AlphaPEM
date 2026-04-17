from configuration.initialize import *
from configuration.settings import *
from model.coefficients import *
from scipy.optimize import fsolve

class PEMFC_1D:
    
    def __init__(self, parameters, operating_inputs):
        self.parameters = parameters
        self.operating_inputs = operating_inputs
    
    def solve(self, i):

        Tfc = self.operating_inputs['Tfc']
        Win_c, Wout_c = self.operating_inputs['Win_c'], self.operating_inputs['Wout_c'] 
        Win_a, Wout_a = self.operating_inputs['Win_a'], self.operating_inputs['Wout_a']
        Phi_a_des, Phi_c_des = self.operating_inputs['Phi_a_des'], self.operating_inputs['Phi_c_des']
        Hcl, Hmem = self.parameters['Hcl'], self.parameters['Hmem']
    
        # ------------------------------ Initial guess------------------------------ #
        Jnet = 0 
        Jw_ca = i / (2 * F) 
        Jw_an = 0
        C_v_cgdl, C_v_ccl, C_v_cinter, s_cgdl, x_front_c = self.gdl_profile(Jw_ca, Win_c, Wout_c, Phi_c_des)
        C_v_agdl, C_v_acl, C_v_ainter, s_agdl, x_front_a = self.gdl_profile(Jw_an, Win_a, Wout_a, Phi_a_des)
        # Water content
        if Jw_ca > 0:
            if C_v_ccl > Csat(Tfc):
                lambda_ccl = np.min([14 + 8*s_cgdl[0], 22])
            else:
                lambda_ccl = np.min([lambda_eq(C_v_ccl, s_cgdl[0], Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
        else:
            lambda_ccl = lambda_eq(C_v_ccl, s_cgdl[0], Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) 

        Klambda = rho_mem/M_eq * Dw(lambda_ccl, Tfc) / (2.5/22 * i / F)
        lambda_acl = lambda_ccl * np.exp(-Hmem/Klambda)
        lambda_mem = [lambda_ccl * np.exp(-x/Klambda) for x in np.linspace(0, 2e-5, 10)]
        # ------------------------------ Solution iteration ------------------------------ #
        success = False
        while success == False:
            Jw_ca = i / (2 * F) - Jnet
            Jw_an = Jnet
            C_v_cgdl, C_v_ccl, C_v_cinter, s_cgdl, x_front_c = self.gdl_profile(Jw_ca, Win_c, Wout_c, Phi_c_des)
            C_v_agdl, C_v_acl, C_v_ainter, s_agdl, x_front_a = self.gdl_profile(Jw_an, Win_a, Wout_a, Phi_a_des)
            # Water content
            if C_v_ccl > Csat(Tfc):
                lambda_ccl = np.min([14 + 8*s_cgdl[0], 22])
            else:
                lambda_ccl = np.min([lambda_eq(C_v_ccl, s_cgdl[0], Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
            if C_v_acl > Csat(Tfc):
                lambda_acl = np.min([14 + 8*s_agdl[0], 22])
            else:
                lambda_acl = np.min([lambda_eq(C_v_acl, s_agdl[0], Tfc, 20) + Jw_an * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])

            Klambda = rho_mem/M_eq * Dw(lambda_ccl, Tfc) / (2.5/22 * i / F)
            lambda_mem = [(1-np.exp(-x/Klambda))/(1-np.exp(Hmem/Klambda)) *
                                        (lambda_acl - lambda_ccl) * np.exp(-Hmem/Klambda) +
                                        lambda_ccl * np.exp(-x/Klambda) for x in np.linspace(0, Hmem, 10)]

            Jmem = -2.5/22 * i / F * (lambda_ccl* np.exp(-Hmem/Klambda) - lambda_acl) / (np.exp(-Hmem/Klambda) - 1)

            if abs(Jnet - Jmem) <= 1e-4:
                success = True
            else:
                Jnet += 0.01 * (Jmem - Jnet)

        return {"Jnet": Jnet, "Jmem": Jmem,
                     "lambda_ccl": lambda_ccl, "lambda_acl": lambda_acl,
                     "C_v_ccl": C_v_ccl, "C_v_acl": C_v_acl,
                     "s_front_cgdl": x_front_c, "s_front_agdl": x_front_a,
                     "C_v_cgdl": C_v_cgdl, "C_v_agdl": C_v_agdl, "lambda_mem": lambda_mem,
                     "s_cgdl": s_cgdl, "s_agdl": s_agdl}

    def gdl_profile(self, Jw, Win, Wout, Phi_des):

        Pa_des, Pc_des = self.operating_inputs['Pa_des'], self.operating_inputs['Pc_des']
        Lgc, Wgc, Hgc, Hgdl = self.parameters['Lgc'], self.parameters['Wgc'], self.parameters['Hgc'], self.parameters['Hgdl']
        Tfc = self.operating_inputs['Tfc']
        mu_l = 3.56e-4 # Pa.s, viscosity of liquid water at 80C
        mu_g = 1.881e-5 # Pa.s, viscosity of water vapor at cathode
        
        ### Define the water flow at CCL
        Cv_in = Phi_des * Csat(Tfc)

        s = np.zeros(10)
        C_v_gdl = np.zeros(10)
        C_v_gc = (Jw * Lgc / Hgc + Cv_in * Win)/Wout
        
        # ------------------- Case 1: CL -> GC ------------------- #
        if Jw > 0:
            # -------- Boundary conditions -------- #
            C_v_inter = C_v_gc + Jw/h_conv(Pc_des, Tfc, Wgc, Hgc)
            C_v_cl = C_v_inter + Hgc/Dc(Pc_des, Tfc) * Jw
            # ------------------- Regime M ------------------- #
            if C_v_cl > Csat(Tfc) and C_v_gc < Csat(Tfc):
                x_front = (Csat(Tfc) - C_v_inter) * (Dc(Pa_des, Tfc) / Jw)
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = np.min([C_v_inter + (x)/Dc(Pc_des, Tfc) * Jw, Csat(Tfc)])
                    def s_gdl_func(n):
                        rhs = (M_H2O * Jw * (x - x_front)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
                        return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                    s[i_node] = np.max([fsolve(s_gdl_func, x0=0.1)[0], 0])
                    i_node += 1
            # ------------------- Regime L ------------------- #
            elif C_v_gc > Csat(Tfc) and C_v_cl > Csat(Tfc):
                mliquid = M_H2O * (Jw + (Win - Wout) * Hgc/Lgc)
                ans1 = (mliquid * Lgc * mu_l/ (Hgc * rho_H2O(Tfc) * mu_g)) ** (1/3)
                s_gdl_inter = ans1 / (ans1 + 1)
                x_front = Hgdl
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = Csat(Tfc)
                    def s_gdl_func(n):
                        rhs = (M_H2O * Jw * (x)) /(-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5) + \
                                0.35425 * s_gdl_inter ** 4 - 0.848 * s_gdl_inter **5 + 0.6315 * s_gdl_inter ** 6 
                        return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                    s[i_node] = np.max([fsolve(s_gdl_func, x0=s_gdl_inter)[0], 0])
                    i_node += 1
            # ------------------- Regime V ------------------- #
            else: 
                x_front = 0
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_inter + (x)/Dc(Pc_des, Tfc) * Jw
                    i_node += 1
        # ------------------- Case 2: GC -> CL ------------------- #
        elif Jw < 0: 
            # -------- Boundary conditions -------- #
            C_v_inter = C_v_gc + Jw/h_conv(Pc_des, Tfc, Wgc, Hgc)
            C_v_cl = C_v_inter + Hgdl/Dc(Pc_des, Tfc) * Jw
            mliquid = M_H2O * (Jw + (Win - Wout) * Hgc/Lgc)
            ans1 = (mliquid * Lgc * mu_l/ (Hgc * rho_H2O(Tfc) * mu_g)) ** (1/3)
            s_gdl_inter = ans1 / (ans1 + 1)
            rhs = (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
            x_front = (0.35425 *s_gdl_inter ** 4 - 0.848 *  s_gdl_inter**5 + 0.6315 *  s_gdl_inter ** 6 * rhs / (M_H2O * Jw))
            # ------------------- Regime V ------------------- #
            if  C_v_gc <= Csat(Tfc):
                x_front = 0
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_inter + (x)/Dc(Pc_des, Tfc) * Jw
                    i_node += 1
            else:
                mliquid = M_H2O * (Jw + (Win - Wout) * Hgc/Lgc)
                ans1 = (mliquid * Lgc * mu_l/ (Hgc * rho_H2O(Tfc) * mu_g)) ** (1/3)
                s_gdl_inter = ans1 / (ans1 + 1)
                rhs = (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5)
                xliquid =  (0.35425 *s_gdl_inter ** 4 - 0.848 *  s_gdl_inter**5 + 0.6315 *  s_gdl_inter ** 6 * rhs / (M_H2O * Jw))
            # ------------------- Regime M ------------------- #
                if xliquid < Hgdl: # Regime M
                    i_node = 0
                    for x in np.linspace(0, Hgdl, 10):
                        C_v_gdl[i_node] = np.min([Csat(Tfc) + (x - x_front)/Dc(Pc_des, Tfc) * Jw, Csat(Tfc)])
                        def s_gdl_func(n):
                            rhs = (M_H2O * Jw * (x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5) + \
                                    0.35425 * s_gdl_inter ** 4 - 0.848 * s_gdl_inter **5 + 0.6315 * s_gdl_inter ** 6 
                            return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                        s[i_node] = np.max([fsolve(s_gdl_func, x0=s_gdl_inter)[0], 0])
                        i_node += 1
            # ------------------- Regime L ------------------- #
                else: 
                    x_front = Hgdl
                    i_node = 0
                    for x in np.linspace(0, Hgdl, 10):
                        C_v_gdl[i_node] = Csat(Tfc)
                        def s_gdl_func(n):
                            rhs = (M_H2O * Jw * (x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c)/nu_l(Tfc)* np.cos(theta_c)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c))**0.5) + \
                                    0.35425 * s_gdl_inter ** 4 - 0.848 * s_gdl_inter **5 + 0.6315 * s_gdl_inter ** 6 
                            return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                        s[i_node] = fsolve(s_gdl_func, x0=s_gdl_inter)[0]
                        i_node += 1
        # ------------------- Case 3: No water flow ------------------- #
        else: 
            # -------- Boundary conditions -------- #
            C_v_inter = C_v_gc
            C_v_cl = C_v_inter 
            xliquid = 0
            # ------------------- Determine the regime in GDL ------------------- #
            if C_v_gc > Csat(Tfc): # Regime L
                x_front = Hgdl
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = Csat(Tfc)
                    def s_gdl_func(n):
                        rhs = 0
                        return 0.35425 * n ** 4 - 0.848 * n **5 + 0.6315 * n ** 6 - rhs
                    s[i_node] = np.max([fsolve(s_gdl_func, x0=0)[0], 0])
                    i_node += 1
            else: # Regime V
                x_front = 0
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_inter
                    i_node += 1

        return C_v_gdl, C_v_cl, C_v_inter, s, x_front




