from model.coefficients import *
from model.inst_values import *
from modules.state_eq import *

class PEMFC_dyn:
    
    def __init__(self, parameters,   
                           operating_inputs,
                           initial_variable_values,
                           time_interval = None):

        # Initialize the operating inputs and parameters dictionaries.
        self.operating_inputs = operating_inputs
        self.parameters = parameters
        self.control_variables = {'Phi_a_des': self.operating_inputs['Phi_a_des'],
                                                'Phi_c_des': self.operating_inputs['Phi_c_des']}
        self.solver_variable_names = ['C_H2_agc', 'C_H2_agdl', 'C_H2_acl','C_H2_mem',
                                                            'C_O2_mem', 'C_O2_ccl', 'C_O2_cgdl', 'C_O2_cgc', 'C_N2',
                                                            'C_v_agc', 'C_v_agdl', 'C_v_acl', 'C_v_ccl', 'C_v_cgdl', 'C_v_cgc',
                                                            's_agdl', 's_acl', 's_ccl', 's_cgdl',
                                                            'lambda_acl', 'lambda_ccl', 'lambda_mem',
                                                            'eta_c', 'Pasm', 'Paem', 'Pcsm', 'Pcem', 'Phi_asm', 'Phi_aem', 'Phi_csm','Phi_cem',
                                                            'Wcp', 'Wa_inj', 'Wc_inj', 'Abp_a', 'Abp_c',
                                                            'C_Pt2_mem', 'C_Pt2_ccl', 'delta_mem', 'S_N_ccl', 'theta_ccl',
                                                            "Tagdl","Tacl","Tmem","Tccl", "Tcgdl"]
        self.micro_parameters = {"n_group_ptParticle": 10, "rmin": 1e-8, "rmax": 1e-6, "dr": 1e-6 / 10,
                                                "R0": 2e-9, "Vm_Pt": 1.45e-5, "krdp": 1e-6, "Cpt2_ref": 1e-3,
                                                "k1": 1e-6, "k1_ref": 1e-6, "k2": 1e-6, "k2_ref": 1e-6,"kdet_ref": 1e-6, "R0": 2e-9,
                                                "F": 96485, "rho_cc": 21.45e3, "Mcc": 195.08}
        self.micro_parameters["r_m"] = (np.linspace(self.micro_parameters["rmin"], self.micro_parameters["rmax"], self.micro_parameters["n_group_ptParticle"] + 1) + self.micro_parameters["dr"] / 2)[1:]
        # GDL nodes name discretization
        discretized_region = ['C_v_agdl', 'C_v_cgdl', 's_agdl', 's_cgdl', 'C_H2_agdl', 'C_O2_cgdl', "Tcgdl", "Tagdl"]
        for variable in discretized_region:
            index = self.solver_variable_names.index(variable)
            # Delete the previous points
            self.solver_variable_names.pop(index)
            # Increase the number of points
            self.solver_variable_names[index:index] = [f'{variable}_{i}' for i in range(1, self.parameters['n_gdl'] + 1)]
        # MEM nodes name discretization
        for name in ["C_O2_mem", "C_H2_mem", "C_Pt2_mem", "lambda_mem","Tmem"]:
            index = self.solver_variable_names.index(name)
            self.solver_variable_names.pop(index)
            self.solver_variable_names[index:index] = [f'{name}_{i}' for i in range(1, self.parameters['n_mem'] + 1)]
        index_prd = self.solver_variable_names.index('S_N_ccl')
        self.solver_variable_names.pop(index_prd)
        self.solver_variable_names[index_prd:index_prd] = [f'S_N_ccl_{i}' for i in range(1, self.micro_parameters["n_group_ptParticle"] + 1)]
        index_theta_ccl = self.solver_variable_names.index('theta_ccl')
        self.solver_variable_names.pop(index_theta_ccl)
        self.solver_variable_names[index_theta_ccl:index_theta_ccl] = [f'theta_ccl_{i}' for i in range(1, self.micro_parameters["n_group_ptParticle"] + 1)]
        #pd.DataFrame(self.solver_variable_names).to_csv('./var name.csv')
        self.all_variable_names = self.solver_variable_names + ['t', 'Ucell','S_sorp_acl', 'S_sorp_ccl',
                                                                                                            'J_lambda_mem_acl', 'J_lambda_mem_ccl',
                                                                                                            'Pagc', 'Pcgc', "Wasm_in", "Wrd"]
        self.variables = {key: [] for key in self.all_variable_names}
        self.dif_eq = {('d' + key + ' / dt'): 0 for key in self.solver_variable_names}
        # Simulation setup
        self.time_interval = (0, 1800)
        self.Imin_aux = 0
        # state recorders 
        self.t = 0
        self.t_hist = []
        self.dt = 0
        self.dt_hist = []   
        self.loadprofile = []
        self.ec_kinetics = {"Ueq": [], "Rmem": [], "Rccl": [], "Racl": [],
                                           "eta_act": [], "eta_conc": [], "i_fc": [], "fdrop": []}
        self.x = initial_variable_values
        self.x_previous = 0
        self.y = 0
        self.T_desire = self.operating_inputs["Tfc"]

        # Convective diffusion coefficients estimation
        self.Hcodi_a = lambda i: 0.1 + i * 1e-4
        self.Hcodi_c = lambda i: 0.1 + i * 1e-4
        self.Imin_aux = 10 #A

    def dxdt(self, t, x_sol):

        # Create state gradients dictionary
        dif = {('d' + key + ' / dt'): 0 for key in self.solver_variable_names}
        # Mapping macro-scale variables
        x = {}
        for index, key in enumerate(self.solver_variable_names):
            x[key] = x_sol[index]
        self.x = x
        iload = self.operating_inputs["current_density"](t)
        Iload = iload * self.parameters["Aact"]
        # Mapping constant parameters
        Hcl, Hgdl, Hgc, Hmem, Aact, Lgc = self.parameters["Hcl"], self.parameters["Hgdl"], self.parameters["Hgc"], self.parameters["Hmem"], self.parameters["Aact"], self.parameters["Lgc"]
        n_gdl, n_mem = self.parameters["n_gdl"], self.parameters["n_mem"]
        epsilon_gdl, epsilon_cl, epsilon_c, epsilon_mc = self.parameters["epsilon_gdl"], self.parameters["epsilon_cl"], self.parameters["epsilon_c"], self.parameters["epsilon_mc"]
        Wgc, Hgc = self.parameters["Wgc"], self.parameters["Hgc"]
        e, tau = self.parameters["e"], self.parameters["tau"]
        # Operating inputs
        Pa_des, Pc_des = self.operating_inputs['Pa_des'], self.operating_inputs["Pc_des"]
        Phi_a_des, Phi_c_des = self.operating_inputs['Phi_a_des'], self.operating_inputs["Phi_c_des"]
        Sa, Sc = self.operating_inputs['Sa'], self.operating_inputs["Sc"]
        Tfc = self.operating_inputs["Tfc"]

        # Pressures in the stack
        Pagc = (x["C_v_agc"] + x["C_H2_agc"]) * R * Tfc
        Pagdl = [(x[f'C_v_agdl_{i}'] + x[f'C_H2_agdl_{i}']) * R * x[f"Tagdl_{i}"] for i in range(1, n_gdl + 1)]
        Pacl = (x["C_v_acl"] + x["C_H2_acl"]) * R * x['Tacl']
        Pccl = (x["C_v_ccl"] + x["C_O2_ccl"] + x["C_N2"]) * R * x['Tccl']
        Pcgdl = [(x[f'C_v_cgdl_{i}'] + x[f'C_O2_cgdl_{i}'] + x["C_N2"]) * R * x[f"Tcgdl_{i}"] for i in range(1, n_gdl + 1)]
        Pcgc = (x["C_v_cgc"] + x["C_O2_cgc"] + x["C_N2"]) * R * Tfc
        #  Molar masses
        Phi_agc = x['C_v_agc'] / C_v_sat(Tfc)
        Phi_cgc = x['C_v_cgc'] / C_v_sat(Tfc)
        y_cgc = x['C_O2_cgc'] / (x['C_O2_cgc'] + x['C_N2'])
        y_cem = (x['Pcem'] - x['Phi_cem'] * Psat(Tfc) - x['C_N2'] * R * Tfc) / (x['Pcem'] - x['Phi_cem'] * Psat(Tfc))
        Magc = x['C_v_agc'] * R * Tfc / Pagc * M_H2O + \
                     x['C_H2_agc'] * R * Tfc / Pagc * M_H2
        Mcgc = Phi_cgc * Psat(Tfc) / Pcgc * M_H2O + \
                    y_cgc * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_O2 + \
                    (1 - y_cgc) * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_N2
        Maem = x['Phi_aem'] * Psat(Tfc) / x['Paem'] * M_H2O + \
                        (1 - x['Phi_aem'] * Psat(Tfc) / x['Paem']) * M_H2
        Masm = x['Phi_asm'] * Psat(Tfc) / x['Pasm'] * M_H2O + \
                    (1 - x['Phi_asm'] * Psat(Tfc) / x['Pasm']) * M_H2
        Mcem = x['Phi_cem'] * Psat(Tfc) / x['Pcem'] * M_H2O + \
                    y_cem * (1 - x['Phi_cem'] * Psat(Tfc) / x['Pcem']) * M_O2 + \
                    (1 - y_cem) * (1 - x['Phi_cem'] * Psat(Tfc) / x['Pcem']) * M_N2
        Mcsm = x['Phi_csm'] * Psat(Tfc) / x['Pcsm'] * M_H2O + \
                    yO2_ext * (1 - x['Phi_csm'] * Psat(Tfc) / x['Pcsm']) * M_O2 + \
                    (1 - yO2_ext) * (1 - x['Phi_csm'] * Psat(Tfc) / x['Pcsm']) * M_N2
        Mext = Phi_ext * Psat(Text) / Pext * M_H2O + \
                    yO2_ext * (1 - Phi_ext * Psat(Text) / Pext) * M_O2 + \
                    (1 - yO2_ext) * (1 - Phi_ext * Psat(Text) / Pext) * M_N2
        Pr_aem = (Pext / x['Paem'])
        Pr_cem = (Pext / x['Pcem'])
        # Mean values ...
        #       ... of the saturated liquid water variable
        s_agdl_agdl = [None] + [x[f's_agdl_{i}'] / 2 + x[f's_agdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
        s_agdl_acl = x[f's_agdl_{n_gdl}'] / 2 + x['s_acl'] / 2
        s_ccl_cgdl = x['s_ccl'] / 2 + x['s_cgdl_1'] / 2
        s_cgdl_cgdl = [None] + [x[f's_cgdl_{i}'] / 2 + x[f's_cgdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
        #       ... of the porosity and the contact angle
        epsilon_mean = epsilon_gdl / 2 + epsilon_cl / 2
        theta_c_mean = theta_c_gdl / 2 + theta_c_cl / 2
        #       ... of the dissolved water variable
        lambda_mem = [x[f'lambda_mem_{i}'] for i in range(1, n_mem + 1)]
        #       ... of the pressure
        Pagdl_agdl = [Pa_des] * n_gdl
        Pagdl_acl = Pagdl[-1] / 2 + Pacl / 2
        Pccl_cgdl = Pccl / 2 + Pcgdl[0] / 2
        Pcgdl_cgdl = [Pa_des] * n_gdl
        
        # Inlet and outlet flows (mol/s) or (kg/s)
        # Anode inlet
        if Iload < self.Imin_aux and self.Imin_aux > 0:
            Wrd = n_cell * M_H2 * Sa * (self.Imin_aux / Aact ) / (2 * F) * Aact  # kg.s-1
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
        
        
        # Resistance
        # The equilibrium potential
        Ueq = (E0 - 8.5e-4 * (x['Tccl'] - 298.15) + R * x['Tccl'] / (2 * F) * (np.log(R * x['Tccl'] * x['C_H2_acl'] / Pref) + 0.5 * np.log(R * x['Tccl'] * x['C_O2_ccl'] / Pref)))
        Rmem = []
        for i_mem in range(1, n_mem + 1):
            lambda_mem = x["lambda_mem_" + str(i_mem)]
            Tmem = x["Tmem_" + str(i_mem)]
            # The proton resistance
            # The proton resistance at the membrane: Rmem
            if lambda_mem >= 1:
                Rmem += [(Hmem/n_mem) / ((0.5139 * lambda_mem - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]
            else:
                Rmem += [(Hmem/n_mem) / (0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]

        #  The proton resistance at the cathode catalyst layer : Rccl
        if x['lambda_ccl'] >= 1:
            Rccl = Hcl / ((epsilon_mc ** tau) * (0.5139 * x['lambda_ccl'] - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / x['Tccl'])))
        else:
            Rccl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / x['Tccl'])))
        if x['lambda_acl'] >= 1:
            Racl = Hcl / ((epsilon_mc ** tau) * (0.5139 * x['lambda_acl'] - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / x['Tacl'])))
        else:
            Racl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / x['Tacl'])))

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
        Jv_agc_agdl = h_a(Pagc, Tfc, Wgc, Hgc) * (x['C_v_agc'] - x['C_v_agdl_1'])
        #   Cathode side
        Jv_cgdl_cgc = h_c(Pcgc, Tfc, Wgc, Hgc) * (x[f'C_v_cgdl_{n_gdl}'] - x['C_v_cgc'])
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
        if x['s_acl'] > 0 and x["C_v_acl"] > C_v_sat(x['Tacl']) and Jv_agdl_agdl[-1] < 0 and x['C_v_agdl_1'] < C_v_sat(Tfc):
            Jwater = -Jv_agdl_agdl[-1]
            s_front_agdl = (C_v_sat(Tfc) - x["C_v_agdl_1"]) * Da_eff(0, epsilon_c, Pc_des, Tfc, epsilon_c, epsilon_gdl) * epsilon_gdl **1.5 / (Jwater)
            if s_front_agdl > Hgdl and s_front_agdl < Hgdl*1.1:
                s_front_agdl = Hgdl
        elif x['C_v_agc'] > C_v_sat(Tfc) and Jv_agc_agdl < 0:
            s_front_agdl = 0
        else:
            s_front_agdl = Hgdl
        if x['s_ccl'] > 0 and x["C_v_ccl"] > C_v_sat(x['Tccl']) and Jv_cgdl_cgdl[0] > 0 and x['C_v_cgdl_10'] < C_v_sat(Tfc):
            Jwater = Jv_cgdl_cgdl[0]
            s_front_cgdl = Hgdl - (C_v_sat(Tfc) - x["C_v_cgdl_10"]) * Dc_eff(0, epsilon_c, Pc_des, Tfc, epsilon_c, epsilon_gdl) * epsilon_gdl **1.5 / (Jwater)
            if s_front_cgdl < 0 and s_front_cgdl > -Hgdl*0.1:
                s_front_cgdl = 0
        elif x['C_v_cgc'] > C_v_sat(Tfc) and Jv_cgdl_cgc > 0:
            s_front_cgdl = Hgdl
        else:
            s_front_cgdl = 0
        
        if s_front_agdl < 0 or s_front_cgdl < 0:
            raise ValueError("Negative saturation {}({}) front position. Check the inputs and the model assumptions.".format("anode" if s_front_agdl < 0 else "cathode", s_front_agdl if s_front_agdl < 0 else s_front_cgdl))
        if s_front_agdl > Hgdl or s_front_cgdl > Hgdl:
            print( x["C_v_cgdl_10"])
            raise ValueError("Saturation front position {} ({}) exceeds the GDL thickness. Check the inputs and the model assumptions.".format("anode" if s_front_agdl > Hgdl else "cathode", s_front_agdl if s_front_agdl > Hgdl else s_front_cgdl))
        

        #_____________________________________Liquid water flows (kg.m-2.s-1)__________________________________________
        if s_front_agdl == 0:
            Jl_agdl_agc = x["s_agdl_1"] **3 / (1 - x["s_agdl_1"]) * rho_H2O(x["Tagdl_1"]) *(1/1298)  *4.8 * 1e-5/3e-4
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
            Jl_cgdl_cgc = x["s_cgdl_{}".format(n_gdl)] **3 / (1 - x["s_cgdl_{}".format(n_gdl)]) * rho_H2O(x["Tcgdl_{}".format(n_gdl)]) *(1/1298)  *4.8 * 1e-5/3e-4
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

        # setpoint
        # Intermediate values
        Prd = x['Pasm']
        Pcp = x['Pcsm']
        # The desired air compressor flow rate Wcp_des (kg.s-1)
        Wcp_des = n_cell * Mext * Pext / (Pext - Phi_ext * Psat(Text)) * \
                          1 / yO2_ext * Sc * (iload) / (4 * F) * Aact
        Wa_inj_des = M_H2O * Phi_a_des * Psat(Tfc) / Prd * (Wrd / M_H2)
        Wc_v_des = M_H2O * Phi_c_des * Psat(Tfc) / Pcp * (x['Wcp'] / Mext)  # Desired vapor flow rate
        Wv_hum_in = M_H2O * Phi_ext * Psat(Text) / Pext * (x['Wcp'] / Mext) 
        Wc_inj_des = Wc_v_des - Wv_hum_in  # Desired humidifier flow rate

        # Auxiliary dynamics 
        dif['dPasm / dt'] = (Wasm_in - n_cell * Wasm_out) / (Vsm * Masm) * R * Tfc
        dif['dPaem / dt'] = (n_cell * Waem_in - Waem_out) / (Vem * Maem) * R * Tfc
        dif['dPcsm / dt'] = (Wcsm_in - n_cell * Wcsm_out) / (Vsm * Mcsm) * R * Tfc
        dif['dPcem / dt'] = (n_cell * Wcem_in - Wcem_out) / (Vem * Mcem) * R * Tfc
        ## Humidity evolution inside the manifolds
        dif['dPhi_asm / dt'] = (Wv_asm_in - Jv_a_in * Hgc * Wgc * n_cell) / Vsm * R * Tfc / Psat(Tfc)
        dif['dPhi_aem / dt'] = (Jv_a_out * Hgc * Wgc * n_cell - Wv_aem_out) / Vem * R * Tfc / Psat(Tfc)
        dif['dPhi_csm / dt'] = (Wv_csm_in - Jv_c_in * Hgc * Wgc * n_cell) / Vsm * R * Tfc / Psat(Tfc)
        dif['dPhi_cem / dt'] = (Jv_c_out * Hgc * Wgc * n_cell - Wv_cem_out) / Vem * R * Tfc / Psat(Tfc)
        dif['dWcp / dt'] = (Wcp_des - x['Wcp']) / tau_cp  # Estimation at the first order.
        ## Anode and cathode humidifiers evolution
        dif['dWa_inj / dt'] = (Wa_inj_des - x['Wa_inj']) / tau_hum  # Estimation at the first order.
        dif['dWc_inj / dt'] = (Wc_inj_des - x['Wc_inj']) / tau_hum  # Estimation at the first order.
        ## Throttle area evolution inside the anode auxiliaries
        dif['dAbp_a / dt'] = - 1e-6 * (Pa_des - Pagc) #+ 1e-7 * dPagcdt  # PD controller
        if x['Abp_a'] > A_T and dif['dAbp_a / dt'] > 0:  # The throttle area cannot be higher than the maximum value
            dif['dAbp_a / dt'] = 0
        elif x['Abp_a'] < 0 and dif['dAbp_a / dt'] < 0:  # The throttle area cannot be lower than 0
            dif['dAbp_a / dt'] = 0
        ## Throttle area evolution inside the cathode auxiliaries
        dif['dAbp_c / dt'] = - Kp * (Pc_des - Pcgc) #+ Kd * dPcgcdt  # PD controller
        if x['Abp_c'] > A_T and dif['dAbp_c / dt'] > 0:  # The throttle area cannot be higher than the maximum value
            dif['dAbp_c / dt'] = 0
        elif x['Abp_c'] < 0 and dif['dAbp_c / dt'] < 0:  # The throttle area cannot be lower than 0
            dif['dAbp_c / dt'] = 0
        
        # AGC dynamics
        dif['dC_v_agc / dt'] = (Jv_a_in - Jv_a_out) / Lgc - Jv_agc_agdl / Hgc
        dif['dC_H2_agc / dt'] = (J_H2_in - J_H2_out) / Lgc - J_H2_agc_agdl / Hgc

        # AGDL dynamics
        for i in range(n_gdl):
            if i == 0: #AGC/AGDL interface
                dif['dC_v_agdl_1 / dt']    = ((Jv_agc_agdl - Jv_agdl_agdl[0]) / (Hgdl / n_gdl + Hgc/2) + Sv_agdl[0])/ (epsilon_gdl * (1 - x['s_agdl_1']))
                dif['dC_H2_agdl_1 / dt'] = (J_H2_agc_agdl - J_H2_agdl_agdl[0]) / (Hgdl / n_gdl + Hgc/2) / (epsilon_gdl * (1 - x['s_agdl_1']))
                dif["dTagdl_1 / dt"]         = ((JT_agc_agdl - JT_agdl[0]) / (Hgdl/n_gdl + Hgc/2) + Sec_agdl)/(Cp_gdl * rho_gdl)
            elif i == n_gdl-1: #AGDL/ACL interface
                dif[f"dTagdl_{n_gdl} / dt"]         = ((JT_agdl[n_gdl-2] - JT_agdl_acl) / (Hgdl/n_gdl + Hcl) + Sec_agdl)/(Cp_gdl * rho_gdl)
                dif[f'dC_v_agdl_{n_gdl} / dt']    = ((Jv_agdl_agdl[n_gdl - 2] - Jv_agdl_acl) / (Hgdl / n_gdl + Hcl) + Sv_agdl[n_gdl-1]) / (epsilon_gdl * (1 - x[f's_agdl_{n_gdl}']))
                dif[f'dC_H2_agdl_{n_gdl} / dt'] = (J_H2_agdl_agdl[n_gdl - 2] - J_H2_agdl_acl) / (Hgdl / n_gdl + Hcl) / (epsilon_gdl * (1 - x[f's_agdl_{n_gdl}']))
            else:
                dif[f"dTagdl_{i+1} / dt"]        = ((JT_agdl[i-1] - JT_agdl[i]) / (Hgdl/n_gdl) + Sec_agdl)/(Cp_gdl * rho_gdl)
                dif[f'dC_v_agdl_{i+1} / dt']   = ((Jv_agdl_agdl[i - 1] - Jv_agdl_agdl[i]) / (Hgdl / n_gdl) + Sv_agdl[i-1]) / (epsilon_gdl * (1 - x[f's_agdl_{i+1}']))
                dif[f'dC_H2_agdl_{i+1} / dt'] =  (J_H2_agdl_agdl[i - 1] - J_H2_agdl_agdl[i]) / (Hgdl / n_gdl) / (epsilon_gdl * (1 - x[f's_agdl_{i+1}']))

            if s_front_agdl == Hgdl:
                dif[f'ds_agdl_{i+1} / dt']        = 0
            elif s_front_agdl == 0:
                if i == 0: #AGC/AGDL interface
                    dif[f'ds_agdl_{i+1} / dt']        = ((Jl_agdl_agc - Jl_agdl_agdl[0]) / (Hgdl / n_gdl + Hgc/2) + M_H2O * Sl_agdl[0]) / (rho_H2O(x[f"Tagdl_1"]) * epsilon_gdl)
                elif i == n_gdl-1: #AGDL/ACL interface
                    dif[f'ds_agdl_{n_gdl} / dt']    = ((Jl_agdl_agdl[n_gdl - 2] - Jl_agdl_acl) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[n_gdl-1]) / (rho_H2O(x[f"Tagdl_{n_gdl}"]) * epsilon_gdl)
                else:
                    dif[f'ds_agdl_{i+1} / dt']        = ((Jl_agdl_agdl[i - 1] - Jl_agdl_agdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[i-1]) / (rho_H2O(x[f"Tagdl_{i+1}"]) * epsilon_gdl)
            else:
                node_front = int((i+1) * s_front_agdl / Hgdl)
                if i+1 <= node_front:
                    dif[f'ds_agdl_{i+1} / dt']        = 0
                elif i == n_gdl-1: #AGDL/ACL interface
                    dif[f'ds_agdl_{n_gdl} / dt']    = ((Jl_agdl_agdl[n_gdl - 2] - Jl_agdl_acl) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[n_gdl-1]) / (rho_H2O(x[f"Tagdl_{n_gdl}"]) * epsilon_gdl)
                else:
                    dif[f'ds_agdl_{i+1} / dt']        = ((Jl_agdl_agdl[i - 1] - Jl_agdl_agdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_agdl[i-1]) / (rho_H2O(x[f"Tagdl_{i+1}"]) * epsilon_gdl)
                
        # ACL dynamics
        dif['ds_acl / dt']       = (Jl_agdl_acl /  (Hgdl / n_gdl) + M_H2O * Sl_acl) / (rho_H2O(x["Tacl"]) * epsilon_cl)
        dif['dC_v_acl / dt']   = (Jv_agdl_acl / (Hgdl / n_gdl + Hcl/2) - S_sorp_acl + Sv_acl + Sp_acl) / (epsilon_cl)
        dif['dC_H2_acl / dt'] = ((J_H2_agdl_acl - J_H2_acl_mem) / (Hgdl / n_gdl + Hcl/2) + S_H2_acl) / (epsilon_cl)
        dif["dTacl / dt"]        = ((JT_agdl_acl - JT_acl_mem) / (Hgdl / n_gdl + Hcl/2) + Sr_acl + Sre_acl + Sad_acl + Sec_acl)/(Cp_cl * rho_cl)

        # MEM dynamics 
        dif['dlambda_acl / dt']         = M_eq / (rho_mem * epsilon_mc) * (-J_lambda_mem_acl /  (Hmem/n_mem + Hcl/10) + S_sorp_acl)
        dif['dlambda_ccl / dt']         = M_eq / (rho_mem * epsilon_mc) * (J_lambda_mem_ccl /  (Hmem/n_mem + Hcl/5) + S_sorp_ccl + Sp_ccl)
        dif['dlambda_mem_1 / dt'] = M_eq / rho_mem * (J_lambda_mem_acl - J_lambda_mem[0]) / (Hmem/n_mem + Hcl/10)
        for i in range(2, n_mem):
            dif[f'dlambda_mem_{i} / dt'] = M_eq / rho_mem * (J_lambda_mem[i-2] - J_lambda_mem[i-1]) / (Hmem/n_mem)
        dif[f'dlambda_mem_{n_mem} / dt'] = M_eq / rho_mem * (J_lambda_mem[-1] - J_lambda_mem_ccl) / (Hmem/n_mem + Hcl/5)
        for i in range(n_mem):
                if i == 0:
                    dif[f"dTmem_{i+1} / dt"] = ((JT_acl_mem - JT_mem[0]) / (Hmem/n_mem + Hcl/3) + Sre_mem[i])/(Cp_mem * rho_mem)
                elif i == n_mem-1:
                    dif[f"dTmem_{i+1} / dt"] = ((JT_mem[n_mem-2] - JT_mem_ccl) / (Hmem/n_mem + Hcl/3) + Sre_mem[i])/(Cp_mem * rho_mem)
                else:
                    dif[f"dTmem_{i+1} / dt"] = ((JT_mem[i-1] - JT_mem[i]) / (Hmem/n_mem) + Sre_mem[i])/(Cp_mem * rho_mem)
        
        # CCL dynamics
        dif["dTccl / dt"] = ((JT_mem_ccl - JT_ccl_cgdl ) / Hcl + Sr_ccl + Sre_ccl + Sad_ccl + Sec_ccl)/(Cp_cl * rho_cl)
        dif['ds_ccl / dt'] = (- Jl_ccl_cgdl /  (Hgdl / n_gdl + Hcl/2) + M_H2O * Sl_ccl) / (rho_H2O(x["Tccl"]) * epsilon_cl)
        dif['dC_v_ccl / dt'] = (- Jv_ccl_cgdl /  (Hgdl / n_gdl + Hcl/2) - S_sorp_ccl + Sv_ccl) / (epsilon_cl * (1 - x['s_ccl']))
        dif['dC_O2_ccl / dt'] = ((J_O2_mem_ccl- J_O2_ccl_cgdl) /  (Hgdl / n_gdl + Hcl/2)  + S_O2_ccl) / (epsilon_cl * (1 - x['s_ccl']))

        # CGDL dynamics
        for i in range(n_gdl):
            if i == 0:
                dif['dC_v_cgdl_1 / dt'] = ((Jv_ccl_cgdl - Jv_cgdl_cgdl[0]) / (Hgdl / n_gdl + Hcl/2) + Sv_cgdl[0]) / (epsilon_gdl * (1 - x['s_cgdl_1']))
                dif['dC_O2_cgdl_1 / dt'] = (J_O2_ccl_cgdl - J_O2_cgdl_cgdl[0]) / (Hgdl / n_gdl + Hcl/2)/ (epsilon_gdl * (1 - x['s_cgdl_1']))
                dif["dTcgdl_1 / dt"] = ((JT_ccl_cgdl - JT_cgdl[0]) / (Hgdl/n_gdl + Hcl/2) + Sec_cgdl)/(Cp_gdl * rho_gdl)
            elif i == n_gdl-1:
                dif[f'dC_v_cgdl_{n_gdl} / dt'] = ((Jv_cgdl_cgdl[n_gdl - 2] - Jv_cgdl_cgc) / (Hgdl / n_gdl + Hgc/2) + Sv_cgdl[n_gdl-1]) / (epsilon_gdl * (1 - x[f's_cgdl_{n_gdl}']))
                dif[f'dC_O2_cgdl_{n_gdl} / dt'] = (J_O2_cgdl_cgdl[n_gdl - 2] - J_O2_cgdl_cgc) / (Hgdl / n_gdl + Hgc/2) / (epsilon_gdl * (1 - x[f's_cgdl_{n_gdl}']))
                dif[f"dTcgdl_{n_gdl} / dt"] = ((JT_cgdl[n_gdl-2] - JT_cgdl_cgc) / (Hgdl/n_gdl + Hgc/2) + Sec_cgdl)/(Cp_gdl * rho_gdl)
            else:
                dif[f'dC_v_cgdl_{i+1} / dt'] = ((Jv_cgdl_cgdl[i - 1] - Jv_cgdl_cgdl[i]) / (Hgdl / n_gdl) + Sv_cgdl[i-1]) / (epsilon_gdl * (1 - x[f's_cgdl_{i+1}']))
                dif[f'dC_O2_cgdl_{i+1} / dt'] = (J_O2_cgdl_cgdl[i - 1] - J_O2_cgdl_cgdl[i]) / (Hgdl / n_gdl) / (epsilon_gdl * (1 - x[f's_cgdl_{i+1}']))
                dif[f"dTcgdl_{i+1} / dt"] = ((JT_cgdl[i-1] - JT_cgdl[i]) / (Hgdl/n_gdl) + Sec_cgdl)/(Cp_gdl * rho_gdl)

            if s_front_cgdl == 0:
                dif[f'ds_cgdl_{i+1} / dt']        = 0
            elif s_front_cgdl == Hgdl: # CGC is fully humidified
                if i == 0: #CCL/CGDL interface
                    dif['ds_cgdl_1 / dt'] = ((Jl_ccl_cgdl - Jl_cgdl_cgdl[0]) / (Hgdl / n_gdl + Hcl/2) + M_H2O * Sl_cgdl[0]) / (rho_H2O(x["Tcgdl_1"]) * epsilon_gdl)
                elif i == n_gdl-1: #CGDL/CGC interface
                    dif[f'ds_cgdl_{n_gdl} / dt'] = ((Jl_cgdl_cgdl[n_gdl - 2] - Jl_cgdl_cgc) / (Hgdl / n_gdl + Hgc/2) + M_H2O * Sl_cgdl[n_gdl-1]) / (rho_H2O(x[f"Tcgdl_{n_gdl}"]) * epsilon_gdl)
                else:
                    dif[f'ds_cgdl_{i+1} / dt'] = ((Jl_cgdl_cgdl[i - 1] - Jl_cgdl_cgdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_cgdl[i-1]) / (rho_H2O(x[f"Tcgdl_{i+1}"]) * epsilon_gdl)
            else: 
                node_front = int((i+1) * s_front_cgdl / Hgdl)
                if i+1 >= node_front:
                    dif[f'ds_cgdl_{i+1} / dt']  = 0
                elif i == 0: #CCL/CGDL interface
                    dif['ds_cgdl_1 / dt'] = ((Jl_ccl_cgdl - Jl_cgdl_cgdl[0]) / (Hgdl / n_gdl + Hcl/2) + M_H2O * Sl_cgdl[0]) / (rho_H2O(x["Tcgdl_1"]) * epsilon_gdl)
                else:
                    dif[f'ds_cgdl_{i+1} / dt']    =((Jl_cgdl_cgdl[i - 1] - Jl_cgdl_cgdl[i]) / (Hgdl / n_gdl) + M_H2O * Sl_cgdl[i-1]) / (rho_H2O(x[f"Tcgdl_{i+1}"]) * epsilon_gdl)

        # CGC dynamics
        dif['dC_v_cgc / dt'] = (Jv_c_in - Jv_c_out) / Lgc + Jv_cgdl_cgc / Hgc
        dif['dC_O2_cgc / dt'] = (J_O2_in - J_O2_out) / Lgc + J_O2_cgdl_cgc / Hgc
        dif['dC_N2 / dt'] = (J_N2_in - J_N2_out) / Lgc

        # Mapping the gradients
        gradient = []
        for key in x:
            gradient.append(dif['d' + key + ' / dt'])

        return gradient


    def _recovery(self, sol):

        """Recover the values which have been calculated by the solver and add them into the variables' dictionary.
        However, the numerical resolution method does not, by design, recover all the internal states of the stack,
        even though they are calculated during this process. They therefore have to be recovered manually.
        """

        # Recovery of the time span
        self.variables['t'].extend(list(sol.t))

        # Recovery of the main variables dynamic evolution
        for index, key in enumerate(self.solver_variable_names):
            self.variables[key].extend(list(sol.y[index]))

        # Recovery of more variables

        for j in range(len(sol.t)):  # For each time...
            # ... recovery of i_fc.
            i_fc = self.operating_inputs["current_density"](self.variables['t'][j])
            self.loadprofile.append(i_fc * self.parameters["Aact"])
            last_solver_variables = {key: self.variables[key][j] for key in self.solver_variable_names}
            flows_recovery = self.calculate_flows(t = self.variables['t'][j], sv = last_solver_variables)
            for key in ['S_sorp_acl', 'S_sorp_ccl', 'J_lambda_mem_acl', 'J_lambda_mem_ccl', 'Pagc', 'Pcgc','Wrd', 'Wasm_in']:
                self.variables[key].append(flows_recovery[key])
            prd_ccl = [last_solver_variables[f"S_N_ccl_{i + 1}"] for i in range(self.micro_parameters["n_group_ptParticle"])]
            theta_ccl =  [last_solver_variables[f"theta_ccl_{i + 1}"] for i in range(self.micro_parameters["n_group_ptParticle"])]
            #  recovery of Ucell.
            Rmem_t, Rccl_t, Racl_t = Rproton(last_solver_variables, self.parameters)
            Ueq_t = Ueq(last_solver_variables)
            f_drop_t = fdrop(last_solver_variables, self.operating_inputs, self.parameters)
            if f_drop_t == 1:
                self.ec_kinetics["eta_act"].append(self.variables["eta_c"][j])
                self.ec_kinetics["eta_conc"].append(0)
            else:
                eta_conc_t = self.variables["eta_c"][j] * (1 - f_drop_t)/f_drop_t
                eta_act_t = self.variables["eta_c"][j] - eta_conc_t
                self.ec_kinetics["eta_act"].append(eta_act_t)
                self.ec_kinetics["eta_conc"].append(eta_conc_t)
            self.ec_kinetics["i_fc"].append(i_fc)
            self.ec_kinetics["fdrop"].append(f_drop_t)
            self.ec_kinetics["Ueq"].append(Ueq_t)
            self.ec_kinetics["Rmem"].append(Rmem_t)
            self.ec_kinetics["Rccl"].append(Rccl_t)
            self.ec_kinetics["Racl"].append(Racl_t)
            self.variables["Ucell"].append(Ucell(self.variables, self.operating_inputs, self.parameters))

    def calculate_flows(self,t, sv):

        # Mapping macro-scale variables
        x = sv
        self.x = x
        iload = self.operating_inputs["current_density"](t)
        Iload = iload * self.parameters["Aact"]
        # Mapping constant parameters
        Hcl, Hgdl, Hgc, Hmem, Aact, Lgc = self.parameters["Hcl"], self.parameters["Hgdl"], self.parameters["Hgc"], self.parameters["Hmem"], self.parameters["Aact"], self.parameters["Lgc"]
        n_gdl, n_mem = self.parameters["n_gdl"], self.parameters["n_mem"]
        epsilon_gdl, epsilon_cl, epsilon_c, epsilon_mc = self.parameters["epsilon_gdl"], self.parameters["epsilon_cl"], self.parameters["epsilon_c"], self.parameters["epsilon_mc"]
        Wgc, Hgc = self.parameters["Wgc"], self.parameters["Hgc"]
        e, tau = self.parameters["e"], self.parameters["tau"]
        # Operating inputs
        Pa_des, Pc_des = self.operating_inputs['Pa_des'], self.operating_inputs["Pc_des"]
        Phi_a_des, Phi_c_des = self.operating_inputs['Phi_a_des'], self.operating_inputs["Phi_c_des"]
        Sa, Sc = self.operating_inputs['Sa'], self.operating_inputs["Sc"]
        Tfc = self.operating_inputs["Tfc"]
        k_purge = 2.5
        # Pressures in the stack
        Pagc = (x["C_v_agc"] + x["C_H2_agc"]) * R * Tfc
        Pagdl = [(x[f'C_v_agdl_{i}'] + x[f'C_H2_agdl_{i}']) * R * x[f"Tagdl_{i}"] for i in range(1, n_gdl + 1)]
        Pacl = (x["C_v_acl"] + x["C_H2_acl"]) * R * x['Tacl']
        Pccl = (x["C_v_ccl"] + x["C_O2_ccl"] + x["C_N2"]) * R * x['Tccl']
        Pcgdl = [(x[f'C_v_cgdl_{i}'] + x[f'C_O2_cgdl_{i}'] + x["C_N2"]) * R * x[f"Tcgdl_{i}"] for i in range(1, n_gdl + 1)]
        Pcgc = (x["C_v_cgc"] + x["C_O2_cgc"] + x["C_N2"]) * R * Tfc
        #       Molar masses
        Phi_agc = x['C_v_agc'] / C_v_sat(Tfc)
        Phi_cgc = x['C_v_cgc'] / C_v_sat(Tfc)
        y_cgc = x['C_O2_cgc'] / (x['C_O2_cgc'] + x['C_N2'])
        y_cem = (x['Pcem'] - x['Phi_cem'] * Psat(Tfc) - x['C_N2'] * R * Tfc) / (x['Pcem'] - x['Phi_cem'] * Psat(Tfc))
        Magc = x['C_v_agc'] * R * Tfc / Pagc * M_H2O + \
                        x['C_H2_agc'] * R * Tfc / Pagc * M_H2
        Mcgc = Phi_cgc * Psat(Tfc) / Pcgc * M_H2O + \
                    y_cgc * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_O2 + \
                    (1 - y_cgc) * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * M_N2
        Maem = x['Phi_aem'] * Psat(Tfc) / x['Paem'] * M_H2O + \
                        (1 - x['Phi_aem'] * Psat(Tfc) / x['Paem']) * M_H2
        Masm = x['Phi_asm'] * Psat(Tfc) / x['Pasm'] * M_H2O + \
                    (1 - x['Phi_asm'] * Psat(Tfc) / x['Pasm']) * M_H2
        Mcem = x['Phi_cem'] * Psat(Tfc) / x['Pcem'] * M_H2O + \
                    y_cem * (1 - x['Phi_cem'] * Psat(Tfc) / x['Pcem']) * M_O2 + \
                    (1 - y_cem) * (1 - x['Phi_cem'] * Psat(Tfc) / x['Pcem']) * M_N2
        Mcsm = x['Phi_csm'] * Psat(Tfc) / x['Pcsm'] * M_H2O + \
                    yO2_ext * (1 - x['Phi_csm'] * Psat(Tfc) / x['Pcsm']) * M_O2 + \
                    (1 - yO2_ext) * (1 - x['Phi_csm'] * Psat(Tfc) / x['Pcsm']) * M_N2
        Mext = Phi_ext * Psat(Text) / Pext * M_H2O + \
                    yO2_ext * (1 - Phi_ext * Psat(Text) / Pext) * M_O2 + \
                    (1 - yO2_ext) * (1 - Phi_ext * Psat(Text) / Pext) * M_N2
        Pr_aem = (Pext / x['Paem'])
        Pr_cem = (Pext / x['Pcem'])
        # Mean values ...
        #       ... of the saturated liquid water variable
        s_agdl_agdl = [None] + [x[f's_agdl_{i}'] / 2 + x[f's_agdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
        s_agdl_acl = x[f's_agdl_{n_gdl}'] / 2 + x['s_acl'] / 2
        s_ccl_cgdl = x['s_ccl'] / 2 + x['s_cgdl_1'] / 2
        s_cgdl_cgdl = [None] + [x[f's_cgdl_{i}'] / 2 + x[f's_cgdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
        #       ... of the porosity and the contact angle
        epsilon_mean = epsilon_gdl / 2 + epsilon_cl / 2
        theta_c_mean = theta_c_gdl / 2 + theta_c_cl / 2
        #       ... of the dissolved water variable
        lambda_mem = [x[f'lambda_mem_{i}'] for i in range(1, n_mem + 1)]
        #       ... of the pressure
        Pagdl_agdl = [Pa_des] * n_gdl
        Pagdl_acl = Pagdl[-1] / 2 + Pacl / 2
        Pccl_cgdl = Pccl / 2 + Pcgdl[0] / 2
        Pcgdl_cgdl = [Pa_des] * n_gdl
        
        # Inlet and outlet flows (mol/s) or (kg/s)
        # Anode inlet
        if Iload < self.Imin_aux and self.Imin_aux > 0:
            Wrd = n_cell * M_H2 * Sa * (self.Imin_aux / Aact ) / (2 * F) * Aact  # kg.s-1
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
        
        # Resistance
        # The equilibrium potential
        Ueq = (E0 - 8.5e-4 * (x['Tccl'] - 298.15) + R * x['Tccl'] / (2 * F) * (np.log(R * x['Tccl'] * x['C_H2_acl'] / Pref) + 0.5 * np.log(R * x['Tccl'] * x['C_O2_ccl'] / Pref)))
        Rmem = []
        for i_mem in range(1, n_mem + 1):
            lambda_mem = x["lambda_mem_" + str(i_mem)]
            Tmem = x["Tmem_" + str(i_mem)]
            # The proton resistance
            # The proton resistance at the membrane: Rmem
            if lambda_mem >= 1:
                Rmem += [(Hmem/n_mem) / ((0.5139 * lambda_mem - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]
            else:
                Rmem += [(Hmem/n_mem) / (0.1879 * np.exp(1268 * (1 / 303.15 - 1 / Tmem)))]

        #  The proton resistance at the cathode catalyst layer : Rccl
        if x['lambda_ccl'] >= 1:
            Rccl = Hcl / ((epsilon_mc ** tau) * (0.5139 * x['lambda_ccl'] - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / x['Tccl'])))
        else:
            Rccl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / x['Tccl'])))
        if x['lambda_acl'] >= 1:
            Racl = Hcl / ((epsilon_mc ** tau) * (0.5139 * x['lambda_acl'] - 0.326) * np.exp(1268 * (1 / 303.15 - 1 / x['Tacl'])))
        else:
            Racl = Hcl / ((epsilon_mc ** tau) * 0.1879 * np.exp(1268 * (1 / 303.15 - 1 / x['Tacl'])))

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

        #_____________________________________Liquid water flows (kg.m-2.s-1)__________________________________________
        Jl_agdl_agc = x["s_agdl_1"] **3 / (1 - x["s_agdl_1"]) * rho_H2O(x["Tagdl_1"]) *(1/1298)  *4.8 * 1e-5/3e-4
        # Anode side
        Jl_agdl_agdl = [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            Jl_agdl_agdl[i-1] = - sigma(x[f"Tagdl_{i}"]) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l(x[f"Tagdl_{i}"]) * abs(np.cos(theta_c_gdl)) * \
                                            (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * (s_agdl_agdl[i] ** e) * (1.417 - 4.24 * s_agdl_agdl[i] + 3.789 * s_agdl_agdl[i] ** 2) * \
                                            (x[f's_agdl_{i + 1}'] - x[f's_agdl_{i}']) / (Hgdl / n_gdl)
        Jl_agdl_acl = - 2 * sigma((x[f"Tagdl_{n_gdl}"] + x['Tacl']) / 2) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l((x[f"Tagdl_{n_gdl}"] + x['Tacl']) / 2) * abs(np.cos(theta_c_mean)) * \
                                (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * (s_agdl_acl ** e) * (1.417 - 4.24 * s_agdl_acl + 3.789 * s_agdl_acl ** 2) * \
                                (x["s_acl"] - x[f's_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl/2)
        # Cathode side
        Jl_cgdl_cgc = x["s_cgdl_{}".format(n_gdl)] **3 / (1 - x["s_cgdl_{}".format(n_gdl)]) * rho_H2O(x["Tcgdl_{}".format(n_gdl)]) *(1/1298)  *4.8 * 1e-5/3e-4
        Jl_cgdl_cgdl = [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            Jl_cgdl_cgdl[i-1] = - sigma(x[f"Tcgdl_{i}"]) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l(x[f"Tcgdl_{i}"]) * abs(np.cos(theta_c_gdl)) * \
                                            (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * (s_cgdl_cgdl[i] ** e) * (1.417 - 4.24 * s_cgdl_cgdl[i] + 3.789 * s_cgdl_cgdl[i] ** 2) * \
                                            (x[f's_cgdl_{i + 1}'] - x[f's_cgdl_{i}']) / (Hgdl / n_gdl)
        Jl_ccl_cgdl = - 2 * sigma((x["Tcgdl_1"] + x['Tccl']) / 2) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l((x["Tcgdl_1"] + x['Tccl']) / 2) * abs(np.cos(theta_c_mean)) * \
                                (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 *(s_ccl_cgdl ** e) * (1.417 - 4.24 * s_ccl_cgdl + 3.789 * s_ccl_cgdl ** 2) * \
                                (x['s_cgdl_1'] - x['s_ccl']) / (Hgdl / n_gdl + Hcl/2)

        # _____________________________________________Vapor flows (mol.m-2.s-1)____________________________________________
        # Convective vapor flows
        #   Anode side
        Jv_agc_agdl = h_a(Pagc, Tfc, Wgc, Hgc) * (x['C_v_agc'] - x['C_v_agdl_1']) * self.Hcodi_a(iload)
        #   Cathode side
        Jv_cgdl_cgc = h_c(Pcgc, Tfc, Wgc, Hgc) * (x[f'C_v_cgdl_{n_gdl}'] - x['C_v_cgc'])  * self.Hcodi_c(iload)
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

        # setpoint
        # Intermediate values
        Prd = x['Pasm']
        Pcp = x['Pcsm']
        # The desired air compressor flow rate Wcp_des (kg.s-1)
        Wcp_des = n_cell * Mext * Pext / (Pext - Phi_ext * Psat(Text)) * \
                            1 / yO2_ext * Sc * (iload) / (4 * F) * Aact
        Wa_inj_des = M_H2O * Phi_a_des * Psat(Tfc) / Prd * (Wrd / M_H2)
        Wc_v_des = M_H2O * Phi_c_des * Psat(Tfc) / Pcp * (x['Wcp'] / Mext)  # Desired vapor flow rate
        Wv_hum_in = M_H2O * Phi_ext * Psat(Text) / Pext * (x['Wcp'] / Mext) 
        Wc_inj_des = Wc_v_des - Wv_hum_in  # Desired humidifier flow rate

        return {'Jv_a_in': Jv_a_in, 'Jv_a_out': Jv_a_out, 'Jv_c_in': Jv_c_in, 'Jv_c_out': Jv_c_out, 'J_H2_in': J_H2_in,
                     'J_H2_out': J_H2_out, 'J_O2_in': J_O2_in, 'J_O2_out': J_O2_out, 'J_N2_in': J_N2_in, 'J_N2_out': J_N2_out,
                     'Jv_agc_agdl': Jv_agc_agdl, 'Jv_agdl_agdl': Jv_agdl_agdl, 'Jv_agdl_acl': Jv_agdl_acl,
                     'Jv_cgdl_cgc': Jv_cgdl_cgc, 'Jv_cgdl_cgdl': Jv_cgdl_cgdl, 'Jv_ccl_cgdl': Jv_ccl_cgdl, 
                     'Jl_agdl_acl': Jl_agdl_acl, 'Jl_agdl_agdl': Jl_agdl_agdl, 'Jl_agdl_agc': Jl_agdl_agc,
                     'Jl_ccl_cgdl': Jl_ccl_cgdl, 'Jl_cgdl_cgdl': Jl_cgdl_cgdl, 'Jl_cgdl_cgc': Jl_cgdl_cgc,
                     'J_lambda_mem_acl': J_lambda_mem_acl, 'J_lambda_mem_ccl': J_lambda_mem_ccl, 'J_lambda_mem': J_lambda_mem,
                     'J_H2_agc_agdl': J_H2_agc_agdl, 'J_H2_agdl_agdl': J_H2_agdl_agdl, 'J_H2_agdl_acl': J_H2_agdl_acl,
                     'J_O2_ccl_cgdl': J_O2_ccl_cgdl, 'J_O2_cgdl_cgdl': J_O2_cgdl_cgdl, 'J_O2_cgdl_cgc': J_O2_cgdl_cgc,
                     'S_sorp_acl': S_sorp_acl, 'S_sorp_ccl': S_sorp_ccl, 'Sp_acl': Sp_acl, 'Sp_ccl': Sp_ccl,
                     'S_H2_acl': S_H2_acl, 'S_O2_ccl': S_O2_ccl, 
                     'Sv_cgdl': Sv_cgdl, 'Sv_agdl': Sv_agdl, 'Sv_acl': Sv_acl, 'Sv_ccl': Sv_ccl,
                     'Sl_agdl': Sl_agdl, 'Sl_acl': Sl_acl, 'Sl_ccl': Sl_ccl, 'Sl_cgdl': Sl_cgdl,
                     'Pagc': Pagc, 'Pcgc': Pcgc, 'Wasm_in': Wasm_in, 'Wasm_out': Wasm_out, 'Waem_in': Waem_in,
                     'Waem_out': Waem_out, 'Wcsm_in': Wcsm_in, 'Wcsm_out': Wcsm_out, 'Wcem_in': Wcem_in, 'Wcem_out': Wcem_out,
                     'Wrd': Wrd, 'Ware': Ware, 'Wv_asm_in': Wv_asm_in, 'Wv_aem_out': Wv_aem_out, 'Wv_csm_in': Wv_csm_in,
                     'Wv_cem_out': Wv_cem_out}