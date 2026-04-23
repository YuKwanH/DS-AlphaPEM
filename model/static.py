from configuration.initialize import *
from configuration.settings import *
from model.coefficients import *
from scipy.optimize import fsolve, brentq

class PEMFC_stat:
    def __init__(self, parameters, operating_inputs):
        self.parameters = parameters
        self.operating_inputs = operating_inputs
    
    def solve(self, i):

        Tfc = self.operating_inputs['Tfc']
        Win_c, Wout_c = self.operating_inputs['Win_c'], self.operating_inputs['Wout_c'] 
        Win_a, Wout_a = self.operating_inputs['Win_a'], self.operating_inputs['Wout_a']
        Phi_a_des, Phi_c_des = self.operating_inputs['Phi_a_des'], self.operating_inputs['Phi_c_des']
        Hcl, Hmem, Wgc, Hgc, Lgc = self.parameters['Hcl'], self.parameters['Hmem'], self.parameters['Wgc'], self.parameters['Hgc'], self.parameters['Lgc']
        epsilon_c, epsilon_cl, epsilon_gdl = self.parameters['epsilon_c'], self.parameters['epsilon_cl'], self.parameters['epsilon_gdl']
        epsilon_mc, tau = self.parameters['epsilon_mc'], self.parameters['tau']
        # ------------------------------ Initial guess------------------------------ #
        Jnet = 0 
        Jw_ca = i / (2 * F) 
        Jw_an = 0
        C_v_cgdl, C_v_ccl, C_v_cinter, C_v_cgc, s_cgdl, x_front_c = self.gdl_profile(Jw_ca, Win_c, Wout_c, Phi_c_des, Pc_des)
        C_v_agdl, C_v_acl, C_v_ainter, C_v_agc, s_agdl, x_front_a = self.gdl_profile(Jw_an, Win_a, Wout_a, Phi_a_des, Pa_des)
        # Water content
        if Jw_ca > 0:
            if C_v_ccl > C_v_sat(Tfc):
                lambda_ccl = np.min([14 + 8*s_cgdl[0]/0.3, 22])
            else:
                lambda_ccl = np.min([lambda_eq(C_v_ccl, s_cgdl[0], Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
        else:
            lambda_ccl = lambda_eq(C_v_ccl, s_cgdl[0], Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) 

        Klambda = rho_mem/M_eq * Dw(lambda_ccl, Tfc) / (2.5/22 * i / F)
        lambda_acl = lambda_ccl * np.exp(-Hmem/Klambda)
        lambda_mem = [lambda_ccl * np.exp(-x/Klambda) for x in np.linspace(0, Hmem, 10)]
        # ------------------------------ Solution iteration ------------------------------ #
        n_iter = 0
        success = False
        while success == False:
            Jw_ca = i / (2 * F) - Jnet
            Jw_an = Jnet
            C_v_cgdl, C_v_ccl, C_v_cinter, C_v_cgc, s_cgdl, x_front_c = self.gdl_profile(Jw_ca, Win_c, Wout_c, Phi_c_des, Pc_des)
            C_v_agdl, C_v_acl, C_v_ainter, C_v_agc, s_agdl, x_front_a = self.gdl_profile(Jw_an, Win_a, Wout_a, Phi_a_des, Pa_des)
            # Water content
            if C_v_ccl > C_v_sat(Tfc):
                lambda_ccl = np.min([14 + 8*s_cgdl[0]/0.3, 22])
            else:
                lambda_ccl = np.min([lambda_eq(C_v_ccl, s_cgdl[0], Tfc, 20) + Jw_ca * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])
            if C_v_acl > C_v_sat(Tfc):
                lambda_acl = np.min([14 + 8*s_agdl[0]/0.3, 22])
            else:
                lambda_acl = np.min([lambda_eq(C_v_acl, s_agdl[0], Tfc, 20) + Jw_an * M_eq / (epsilon_cl * Hcl * 1.3 *rho_mem) , 14])

            Klambda = rho_mem/M_eq * Dw(lambda_ccl, Tfc) / (2.5/22 * i / F)
            lambda_mem = [(1-np.exp(-x/Klambda))/(1-np.exp(-Hmem/Klambda)) *
                                        (lambda_acl - lambda_ccl* np.exp(-Hmem/Klambda))  +
                                        lambda_ccl * np.exp(-x/Klambda) for x in np.linspace(0, Hmem, 10)]

            Jmem = -2.5/22 * i / F * (lambda_ccl* np.exp(-Hmem/Klambda) - lambda_acl) / (np.exp(-Hmem/Klambda) - 1)

            if abs(Jnet - Jmem) <= 1e-4:
                success = True
            else:
                Jnet += 0.01 * (Jmem - Jnet)
                n_iter += 1
            
            if n_iter > 1000:
                #print("Warning: Solution did not converge after 1000 iterations.")
                break

        JH2 = -i / (2 * F)
        JO2 = -i / (4 * F)
        C_O2_cgdl = np.zeros(10)
        C_H2_agdl = np.zeros(10)
        C_H2_agc = Pa_des / (R * Tfc) - C_v_agc
        C_O2_cgc = (Pc_des / (R * Tfc) - C_v_cgc) * 0.21
        C_H2_inter = C_H2_agc + JH2/h_a(Pa_des, Tfc, Wgc, Hgc)
        C_O2_inter = C_O2_cgc + JO2/h_c(Pc_des, Tfc, Wgc, Hgc)
        i_node = 0

        for x in np.linspace(0, Hgdl, 10):
            C_O2_cgdl[i_node] = C_O2_inter + (x)/Dc_eff(s_cgdl[i_node],epsilon_gdl,Pc_des, Tfc,epsilon_c,epsilon_gdl) * JO2
            C_H2_agdl[i_node] = C_H2_inter + (x)/Da_eff(s_agdl[i_node],epsilon_gdl,Pa_des, Tfc,epsilon_c,epsilon_gdl) * JH2
            i_node += 1

        C_H2_acl = C_H2_agdl[0] + Hcl/Da_eff(np.mean(s_agdl),epsilon_cl,Pa_des, Tfc,epsilon_c,epsilon_cl) * JH2
        C_O2_ccl = C_O2_cgdl[0] + Hcl/Dc_eff(np.mean(s_cgdl),epsilon_cl,Pc_des, Tfc,epsilon_c,epsilon_cl) * JO2
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

        if lambda_ccl >= 1:
            Rccl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_ccl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tfc)))
        else:
            Rccl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tfc)))
        # 
        if lambda_acl >= 1:
            Racl = Hcl / ((epsilon_mc ** tau) * (0.5139 * lambda_acl - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tfc)))
        else:
            Racl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tfc)))


        return {"Jnet": Jnet, "Jmem": Jmem,
                     "lambda_ccl": lambda_ccl, "lambda_acl": lambda_acl,"lambda_mem": lambda_mem,
                     "C_v_ccl": C_v_ccl, "C_v_acl": C_v_acl,"C_v_cgdl": C_v_cgdl, "C_v_agdl": C_v_agdl, 
                     "C_v_a_inter": C_v_ainter, "C_v_c_inter": C_v_cinter, "C_v_cgc": C_v_cgc, "C_v_agc": C_v_agc,
                     "s_front_cgdl": x_front_c, "s_front_agdl": x_front_a,"s_cgdl": s_cgdl, "s_agdl": s_agdl,
                     "C_H2_acl": C_H2_acl, "C_O2_ccl": C_O2_ccl,
                     "C_H2_agc": C_H2_agc, "C_O2_cgc": C_O2_cgc,
                     "C_H2_inter": C_H2_inter, "C_O2_inter": C_O2_inter,
                     "Ueq": Ueq, "eta_c": eta_c, "Rohm": Rohm, "Rccl": Rccl, "Racl": Racl,
                     "Jw_ca":Jw_ca, "Jw_an": Jw_an, "JH2": JH2, "JO2": JO2,
                     "Jv_a_in": Win_a * Phi_a_des * C_v_sat(Tfc)/Lgc, "Jv_a_out": Wout_a * C_v_agc/Lgc,
                     "Jv_c_in": Win_c * Phi_c_des * C_v_sat(Tfc)/Lgc, "Jv_c_out": Wout_c * C_v_cgc/Lgc,}

    def gdl_profile(self, Jw, Win, Wout, Phi_des, P_des):

        Lgc, Wgc, Hgc, Hgdl = self.parameters['Lgc'], self.parameters['Wgc'], self.parameters['Hgc'], self.parameters['Hgdl']
        epsilon_gdl, epsilon_c = self.parameters['epsilon_gdl'], self.parameters['epsilon_c']
        Tfc = self.operating_inputs['Tfc']
        mu_l = 3.56e-4 # Pa.s, viscosity of liquid water at 80C
        mu_g = 1.881e-5 # Pa.s, viscosity of water vapor at cathode
        
        ### Define the water flow at GC
        Cv_in = Phi_des * C_v_sat(Tfc)
        s = np.zeros(10)
        C_v_gdl = np.zeros(10)
        C_v_gc = (Jw * Lgc / Hgc + Cv_in * Win)/Wout
        # -------- Boundary conditions -------- #
        C_v_inter = C_v_gc + Jw/h_c(P_des, Tfc, Wgc, Hgc)
        C_v_cl = C_v_inter + epsilon_gdl*Hgdl/Dc_eff(np.mean(s),epsilon_gdl,P_des, Tfc,epsilon_c,epsilon_gdl) * Jw
        
        # ------------------- Case 1: CL -> GC ------------------- #
        if Jw > 0:
            # ------------------- Regime M ------------------- #
            if C_v_cl > C_v_sat(Tfc) and C_v_inter < C_v_sat(Tfc):
                x_front = (C_v_sat(Tfc) - C_v_inter) * (Dc_eff(np.mean(s),epsilon_gdl,P_des, Tfc,epsilon_c,epsilon_gdl) / Jw)
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = np.min([C_v_inter + (x)/Dc_eff(np.mean(s),epsilon_gdl,P_des, Tfc,epsilon_c,epsilon_gdl) * Jw, C_v_sat(Tfc)])
                    rhs = (M_H2O * Jw * (x - x_front)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5)
                    s[i_node] = self._solve_sat(rhs)
                    i_node += 1
            # ------------------- Regime L ------------------- #
            elif C_v_inter > C_v_sat(Tfc) and C_v_cl > C_v_sat(Tfc):
                mliquid = M_H2O * (Jw + (Win - Wout) * Hgc/Lgc)
                ans1 = (mliquid * Lgc * mu_l/ (Hgc * rho_H2O(Tfc) * mu_g)) ** (1/3)
                s_gdl_inter = ans1 / (ans1 + 1)
                x_front = Hgdl
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_sat(Tfc)
                    rhs = (M_H2O * Jw * (x)) /(-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5) + \
                                0.35425 * s_gdl_inter ** 4 - 0.848 * s_gdl_inter **5 + 0.6315 * s_gdl_inter ** 6
                    s[i_node] = self._solve_sat(rhs)
                    i_node += 1
            # ------------------- Regime V ------------------- #
            else: 
                x_front = 0
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_inter + (x)/Dc_eff(np.mean(s),epsilon_gdl,P_des, Tfc,epsilon_c,epsilon_gdl) * Jw
                    i_node += 1
        # ------------------- Case 2: GC -> CL ------------------- #
        elif Jw < 0: 
            # -------- Boundary conditions -------- #
            mliquid = M_H2O * (Jw + (Win - Wout) * Hgc/Lgc)
            ans1 = (mliquid * Lgc * mu_l/ (Hgc * rho_H2O(Tfc) * mu_g)) ** (1/3)
            s_gdl_inter = ans1 / (ans1 + 1)
            rhs = (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5)
            x_front = (0.35425 *s_gdl_inter ** 4 - 0.848 *  s_gdl_inter**5 + 0.6315 *  s_gdl_inter ** 6) * rhs / (M_H2O * Jw)
            # ------------------- Regime V ------------------- #
            if  C_v_inter <= C_v_sat(Tfc):
                x_front = 0
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_inter + (x)/Dc_eff(0,epsilon_gdl,P_des, Tfc,epsilon_c,epsilon_gdl) * Jw
                    i_node += 1
            else:
                mliquid = M_H2O * (Jw + (Win - Wout) * Hgc/Lgc)
                ans1 = (mliquid * Lgc * mu_l/ (Hgc * rho_H2O(Tfc) * mu_g)) ** (1/3)
                s_gdl_inter = ans1 / (ans1 + 1)
                rhs = (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5)
                xliquid =  (0.35425 *s_gdl_inter ** 4 - 0.848 *  s_gdl_inter**5 + 0.6315 *  s_gdl_inter ** 6) * rhs / (M_H2O * Jw)
            # ------------------- Regime M ------------------- #
                if xliquid < Hgdl: # Regime M
                    i_node = 0
                    for x in np.linspace(0, Hgdl, 10):
                        C_v_gdl[i_node] = np.min([C_v_sat(Tfc) + (x - x_front)/Dc_eff(0,epsilon_gdl,P_des, Tfc,epsilon_c,epsilon_gdl) * Jw, C_v_sat(Tfc)])
                        rhs = (M_H2O * Jw * (x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5) + \
                                0.35425 * s_gdl_inter ** 4 - 0.848 * s_gdl_inter **5 + 0.6315 * s_gdl_inter ** 6 
                        s[i_node] = self._solve_sat(rhs)
                        i_node += 1
            # ------------------- Regime L ------------------- #
                else: 
                    x_front = Hgdl
                    i_node = 0
                    for x in np.linspace(0, Hgdl, 10):
                        C_v_gdl[i_node] = C_v_sat(Tfc)
                        rhs = (M_H2O * Jw * (x)) / (-sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl)/nu_l(Tfc)* np.cos(theta_c_gdl)*(epsilon_gdl/K0(epsilon_gdl,epsilon_c, epsilon_gdl))**0.5) + \
                                0.35425 * s_gdl_inter ** 4 - 0.848 * s_gdl_inter **5 + 0.6315 * s_gdl_inter ** 6 
                        s[i_node] = self._solve_sat(rhs)
                        i_node += 1
        # ------------------- Case 3: No water flow ------------------- #
        else: 
            # -------- Boundary conditions -------- #
            C_v_inter = C_v_gc
            C_v_cl = C_v_inter 
            xliquid = 0
            # ------------------- Determine the regime in GDL ------------------- #
            if C_v_inter > C_v_sat(Tfc): # Regime L
                x_front = Hgdl
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_sat(Tfc)
                    s[i_node] = 0
                    i_node += 1
            else: # Regime V
                x_front = 0
                i_node = 0
                for x in np.linspace(0, Hgdl, 10):
                    C_v_gdl[i_node] = C_v_inter
                    i_node += 1


        return C_v_gdl, np.min([C_v_cl, C_v_sat(Tfc)]), np.min([C_v_inter, C_v_sat(Tfc)]), np.min([C_v_gc, C_v_sat(Tfc)]), s, x_front
    
    @staticmethod
    def _solve_sat(rhs):
        """Solve 0.35425*s^4 - 0.848*s^5 + 0.6315*s^6 = rhs for s in [0,1]."""
        if rhs <= 0:
            return 0.0
        f = lambda s: 0.35425 * s**4 - 0.848 * s**5 + 0.6315 * s**6 - rhs
        if f(1.0) < 0:
            return 1.0
        return brentq(f, 0, 1, xtol=1e-12)





