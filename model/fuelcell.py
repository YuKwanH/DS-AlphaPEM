import pandas as pd
import scipy

from model.coefficients import *
from model.states import *
from dynamic.gradients import *
from dynamic.control import control_operating_conditions

import warnings
warnings.filterwarnings("ignore")

class PEMFC_1D:
    
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
        self.all_variable_names = self.solver_variable_names + ['t', 'Ucell','ecsa', 'S_sorp_acl', 'S_sorp_ccl',
                                                                                                            'J_lambda_mem_acl', 'J_lambda_mem_ccl',
                                                                                                            'Pagc', 'Pcgc', 'Phi_a_des', 'Phi_c_des',"Wasm_in"]
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
        self.elec_variables = {"Ueq": [], "Rp": [], "eta_act": [], "eta_conc": [], "i_fc": [], "fdrop": []}
        self.x = initial_variable_values
        self.x_previous = 0
        self.y = 0
        self.T_desire = self.operating_inputs["Tfc"]

        # Convective diffusion coefficients estimation
        self.Hcodi = lambda i: 0.1 + i * 1e-4

    def dxdt(self, t, x_solver):

        self.dt = t - self.t
        self.t = t
        # Create state gradients dictionary
        self.dif_eq = {('d' + key + ' / dt'): 0 for key in self.solver_variable_names}
        # Mapping macro-scale variables
        self.x = x_solver
        x = {}
        for index, key in enumerate(self.solver_variable_names):
            x[key] = x_solver[index]
        Hcl, Hgdl, Hgc, Hmem = self.parameters["Hcl"], self.parameters["Hgdl"], self.parameters["Hgc"], self.parameters["Hmem"]
        Re = self.parameters["Re"]
        n_gdl, n_mem = self.parameters["n_gdl"], self.parameters["n_mem"]
        # Mapping micro-scale variables
        prd0_ccl = initPRD(resolution=self.micro_parameters["n_group_ptParticle"])
        prd_ccl = prd0_ccl.copy()
        theta_ccl = np.zeros(self.micro_parameters["n_group_ptParticle"])
        r_m = self.micro_parameters["r_m"]
        for i in range(self.micro_parameters["n_group_ptParticle"]):
            prd_ccl[i] = x[f"S_N_ccl_{i + 1}"]
            theta_ccl[i] = x[f"theta_ccl_{i + 1}"]
        # Temperature variables
        self.Tcgc = self.operating_inputs["Tfc"]
        self.Tagc = self.operating_inputs["Tfc"]
        self.C_Pt2_CCL = x["C_Pt2_ccl"]
        self.delta_mem = x["delta_mem"]
        self.parameters["Hmem"] = x["delta_mem"]
        # Calculate the state of fuel cell system
        # electrical potential
        i_fc = self.operating_inputs['current_density'](t)
        I_fc = i_fc * self.parameters["Aact"]
        imin_aux = self.Imin_aux / self.parameters["Aact"]
        eta_c_intermediate_values = calculate_eta_c_intermediate_values(x, self.operating_inputs, self.parameters)
        Ueq, Rmem, Rccl, Racl = calculate_cell_voltage_intermediate( x,self.parameters)
        Rohm = sum(Rmem) + Rccl + Racl + Re
        self.Ucell = Ueq - x["eta_c"] - (i_fc) * Rohm
        # intermediate mass and pressure calculation
        instant_state = dif_eq_int_values(t, x, i_fc, self.control_variables, self.operating_inputs, self.parameters, self.Imin_aux)
        Pagc = instant_state["Pagc"]
        Pcgc = instant_state["Pcgc"]
        Masm = instant_state["Masm"]
        Maem = instant_state["Maem"]
        Mcsm = instant_state["Mcsm"]
        Mcem = instant_state["Mcem"]
        Wcp_des = instant_state["Wcp_des"]
        Wa_inj_des = instant_state["Wa_inj_des"]
        Wc_inj_des = instant_state["Wc_inj_des"]
        # imtermediate mass flow calculation
        massflow = self.updateMassFlow(t=t, sv=x, iload=i_fc, I_aux_min=self.Imin_aux,
                                                                    operating_inputs=self.operating_inputs, 
                                                                    parameters=self.parameters)
        # Modification for the convective diffusion coefficients
        if I_fc < 10 and self.Imin_aux > 0:
            massflow["Jv_agc_agdl"] = massflow["Jv_agc_agdl"] * self.Hcodi(imin_aux)
            massflow["Jv_cgdl_cgc"] = massflow["Jv_cgdl_cgc"] * self.Hcodi(imin_aux)
        else:
            massflow["Jv_agc_agdl"] = massflow["Jv_agc_agdl"] * self.Hcodi(i_fc)
            massflow["Jv_cgdl_cgc"] = massflow["Jv_cgdl_cgc"] * self.Hcodi(i_fc)
        # CCL kinetic coefficients calculation
        self.C_H_CCL = Cproton_CCL(lambda_w=x["lambda_ccl"])
        kdis = PtDissolution(self.Ucell, self.operating_inputs["Tfc"], x["C_Pt2_ccl"], theta_ccl)
        kox = PtOxidation(self.Ucell, self.operating_inputs["Tfc"], self.C_H_CCL, theta_ccl)
        kcdis = PtOxideDissolution(theta_ccl, self.C_H_CCL)
        kdet = PtDetachment(self.Ucell, self.operating_inputs["Tfc"], r_m)
        drdt = Vm_Pt * krdp * self.C_Pt2_CCL * np.exp(R0 / r_m) - Vm_Pt * (kdis + kox) * Cpt2_ref * np.exp(R0 / r_m)
        # Update health indicator
        self.ECSA_norm = getECSA(prd_ccl, r_m) / getECSA(prd0_ccl, r_m)
        # State gradients calculation
        gradient_prdCCL(dif=self.dif_eq, prd=prd_ccl, theta=theta_ccl,
                                        kox=kox, kcdis=kcdis, kdet=kdet, drdt=drdt, r_m=r_m)
        gradient_compressor(dif=self.dif_eq, Wcp_des=Wcp_des, Wa_inj_des=Wa_inj_des, Wc_inj_des=Wc_inj_des,
                                                **self.parameters, **x)
        gradient_AGC(dif=self.dif_eq, **self.parameters, **massflow)
        gradient_CGC(dif=self.dif_eq, **self.parameters, **massflow)
        gradient_N2(dif=self.dif_eq, **self.parameters, **massflow)
        gradient_AGDL(dif=self.dif_eq, sv=x,  **self.parameters, **massflow)
        gradient_CGDL(dif=self.dif_eq, sv=x, **self.parameters, **massflow)
        gradient_ACL(dif=self.dif_eq, sv=x, **self.parameters, **massflow)
        gradient_CCL(dif=self.dif_eq, sv=x,  kcdis=kcdis, r_m=r_m, drdt=drdt,
                                 prd0= prd0_ccl, prd_ccl=prd_ccl,**self.parameters,**massflow)
        gradient_mem(dif=self.dif_eq, Ucell=self.Ucell, Tfc=self.operating_inputs["Tfc"], C_O2_ccl=x["C_O2_ccl"],
                                **self.parameters, **massflow)
        gradient_Manifold(dif=self.dif_eq, Masm=Masm, Maem=Maem, Mcsm=Mcsm, Mcem=Mcem,
                                       **self.parameters, **massflow, **x, **self.operating_inputs)
        gradient_throttleArea(dif=self.dif_eq, Pagc=Pagc, Pcgc=Pcgc,
                                            **x, **self.parameters, **self.operating_inputs)
        gradient_Vfc(dif=self.dif_eq, i_fc=i_fc, ECSA = 1, **self.parameters,
                              **x, **self.operating_inputs, **eta_c_intermediate_values)
        
        # Temperature dynamic
        k_GDL = 6.5 # [W/(m*K)] thermal conductivity of GDL
        k_CL = 0.27 # [W/(m*K)] thermal conductivity of CL
        k_PEM = 21 # [W/(m*K)] thermal conductivity of PEM
        Cp_cl = 7.7e2 # J/(kg*K) specific heat capacity of CL
        Cp_gdl = 8.4e2 # J/(kg*K) specific heat capacity of GDL
        Cp_mem = 1.1e3 # J/(kg*K) specific heat capacity of PEM
        deltaS_OOR = -163.3 #  J/(mol*K)
        deltaS_HOR = 0.104 # J/(mol*K)
        Tfc = self.operating_inputs["Tfc"]
        JT_ccl_cgdl = (k_CL+k_GDL)/2 * (x["Tccl"] - x["Tcgdl_1"]) / ((Hcl+Hgdl)/2)
        JT_agdl_acl = (k_CL+k_GDL)/2 * (x[f"Tagdl_{n_gdl}"] - x["Tacl"]) / ((Hgdl+Hcl)/2)
        JT_mem_ccl = (k_CL+k_PEM)/2 * (x[f"Tmem_{n_mem}"] - x["Tccl"]) / ((Hmem+Hcl)/2)
        JT_acl_mem = (k_CL+k_PEM)/2 * (x["Tacl"] - x["Tmem_1"]) / ((Hmem+Hcl)/2)
        JT_agc_agdl =  k_GDL * (Tfc - x["Tagdl_1"]) / (Hgdl/n_gdl + Hgc)
        JT_cgdl_cgc = k_GDL * (x[f"Tcgdl_{n_gdl}"] - Tfc) / (Hgdl/n_gdl + Hgc)
        JT_cgdl = np.zeros(n_gdl-1)
        JT_agdl = np.zeros(n_gdl-1)
        JT_mem = np.zeros(n_mem-1)
        Sr_acl = - deltaS_HOR * i_fc/(2*F)* x["Tacl"]
        Sr_ccl = x["eta_c"] * i_fc - deltaS_OOR * i_fc/(4*F) * x["Tccl"]
        Sad_acl =  0#massflow["S_sorp_acl"] * 42e3
        Sad_ccl =  0#massflow["S_sorp_ccl"] * 42e3
        Sec_acl = 0#massflow["Sv_acl"] * 42e3
        Sec_ccl = 0#massflow["Sv_ccl"] * 42e3
        Sre_acl = 0#Racl * i_fc**2
        Sre_ccl = 0#Rccl * i_fc**2
        Sre_mem =  np.array(Rmem) * i_fc**2 

        for i in range(1, n_gdl):
            JT_agdl[i-1] = k_GDL * (x[f"Tagdl_{i}"] - x[f"Tagdl_{i+1}"]) / (Hgdl/n_gdl)
        for i in range(1, n_gdl):
            JT_cgdl[i-1] = k_GDL * (x[f"Tcgdl_{i}"] - x[f"Tcgdl_{i+1}"]) / (Hgdl/n_gdl)
        for i in range(1, n_mem):
            JT_mem[i-1] = k_PEM * (x[f"Tmem_{i}"] - x[f"Tmem_{i+1}"]) / (Hmem/n_mem)

        self.dif_eq["dTccl / dt"] = ((JT_mem_ccl - JT_ccl_cgdl ) / Hcl + Sr_ccl + Sre_ccl + Sad_ccl + Sec_ccl)/(Cp_cl * rho_cl)
        self.dif_eq["dTacl / dt"] = ((JT_agdl_acl - JT_acl_mem) / Hcl + Sr_acl + Sre_acl + Sad_acl + Sec_acl)/(Cp_cl * rho_cl)
        for i in range(n_gdl):
                Sec_agdl = 0#-massflow[f"Sv_agdl"][i+1] * 42e3
                Sec_cgdl = 0#-massflow[f"Sv_cgdl"][i+1] * 42e3
                if i == 0:
                    self.dif_eq[f"dTagdl_{i+1} / dt"] = ((JT_agc_agdl - JT_agdl[0]) / (Hgdl/n_gdl + Hgc) + Sec_agdl)/(Cp_gdl * rho_gdl)
                    self.dif_eq[f"dTcgdl_{i+1} / dt"] = ((JT_ccl_cgdl - JT_cgdl[0]) / (Hgdl/n_gdl + Hcl) + Sec_cgdl)/(Cp_gdl * rho_gdl)
                elif i == n_gdl-1:
                    self.dif_eq[f"dTagdl_{i+1} / dt"] = ((JT_agdl[n_gdl-2] - JT_agdl_acl) / (Hgdl/n_gdl + Hcl) + Sec_agdl)/(Cp_gdl * rho_gdl)
                    self.dif_eq[f"dTcgdl_{i+1} / dt"] = ((JT_cgdl[n_gdl-2] - JT_cgdl_cgc) / (Hgdl/n_gdl + Hgc) + Sec_cgdl)/(Cp_gdl * rho_gdl)
                else:
                    self.dif_eq[f"dTagdl_{i+1} / dt"] = ((JT_agdl[i-1] - JT_agdl[i]) / (Hgdl/n_gdl) + Sec_agdl)/(Cp_gdl * rho_gdl)
                    self.dif_eq[f"dTcgdl_{i+1} / dt"] = ((JT_cgdl[i-1] - JT_cgdl[i]) / (Hgdl/n_gdl) + Sec_cgdl)/(Cp_gdl * rho_gdl)
        
        for i in range(n_mem):
                if i == 0:
                    self.dif_eq[f"dTmem_{i+1} / dt"] = ((JT_acl_mem - JT_mem[0]) / (Hmem/n_mem + Hcl) + Sre_mem[i])/(Cp_mem * rho_mem)
                elif i == n_mem-1:
                    self.dif_eq[f"dTmem_{i+1} / dt"] = ((JT_mem[n_mem-2] - JT_mem_ccl) / (Hmem/n_mem + Hcl) + Sre_mem[i])/(Cp_mem * rho_mem)
                else:
                    self.dif_eq[f"dTmem_{i+1} / dt"] = ((JT_mem[i-1] - JT_mem[i]) / (Hmem/n_mem) + Sre_mem[i])/(Cp_mem * rho_mem)
        
        # Mapping the gradients
        gradient = []
        for key in x:
            gradient.append(self.dif_eq['d' + key + ' / dt'])

        return gradient


    def updateMassFlow(self, t, sv, iload, operating_inputs, parameters, I_aux_min=0):

        # ___________________________________________________Preliminaries__________________________________________________
        # Extraction of the variables
        C_v_agc, C_v_acl, C_v_ccl, C_v_cgc = sv['C_v_agc'], sv['C_v_acl'], sv['C_v_ccl'], sv['C_v_cgc']
        s_acl, s_ccl = sv['s_acl'], sv['s_ccl']
        lambda_acl, lambda_ccl = sv['lambda_acl'], sv['lambda_ccl']
        C_H2_agc, C_H2_acl, C_O2_ccl, C_O2_cgc = sv['C_H2_agc'], sv['C_H2_acl'], sv['C_O2_ccl'], sv['C_O2_cgc']
        C_N2 = sv['C_N2']
        Pasm, Paem = sv['Pasm'], sv['Paem']
        Pcsm, Pcem = sv['Pcsm'], sv['Pcem']
        Abp_a, Abp_c = sv['Abp_a'], sv['Abp_c']
        Wcp = sv['Wcp']

        # Extraction of the operating inputs and parameters
        Tfc = self.operating_inputs['Tfc']
        Hgdl, Hmem, Hcl = self.parameters['Hgdl'], self.parameters['Hmem'], self.parameters['Hcl']
        Hgc, Wgc = self.parameters['Hgc'], self.parameters['Wgc']
        epsilon_gdl, epsilon_c, epsilon_cl = self.parameters['epsilon_gdl'], self.parameters['epsilon_c'], self.parameters['epsilon_cl']
        e, kappa_co = self.parameters['e'], self.parameters['kappa_co']
        n_gdl, n_mem = self.parameters['n_gdl'], self.parameters['n_mem']
        Tacl, Tccl = sv['Tacl'], sv['Tccl']
        Sa, Sc = operating_inputs['Sa'], operating_inputs['Sc']
        Aact = parameters['Aact']
        if iload * Aact < I_aux_min and I_aux_min > 0:
            iload = I_aux_min / Aact
            Iload = I_aux_min
        else:
            iload = iload
            Iload = iload * Aact

        # Intermediate values
        # intermediate mass and pressure calculation
        instant_state = dif_eq_int_values(t, sv, iload, self.control_variables, self.operating_inputs, self.parameters, self.Imin_aux)
        Pagc, Pcgc = instant_state["Pagc"], instant_state["Pcgc"]
        Pr_aem, Pr_cem = instant_state["Pr_aem"], instant_state["Pr_cem"]
        Mext, Masm, Maem, Mcsm, Mcem, Magc, Mcgc = instant_state["Mext"], instant_state["Masm"], instant_state["Maem"], instant_state["Mcsm"], instant_state["Mcem"], instant_state["Magc"], instant_state["Mcgc"]
        Phi_asm, Phi_aem, Phi_csm, Phi_cem, Phi_agc, Phi_cgc = instant_state["Phi_asm"], instant_state["Phi_aem"], instant_state["Phi_csm"], instant_state["Phi_cem"], instant_state["Phi_agc"], instant_state["Phi_cgc"]
        y_cgc = instant_state["y_cgc"]

        # Pressures in the stack
        Pagc = (C_v_agc + C_H2_agc) * R * Tfc
        Pagdl = [(sv[f'C_v_agdl_{i}'] + sv[f'C_H2_agdl_{i}']) * R * sv[f"Tagdl_{i}"] for i in range(1, n_gdl + 1)]
        Pacl = (C_v_acl + C_H2_acl) * R * sv['Tacl']
        Pccl = (C_v_ccl + C_O2_ccl + C_N2) * R * sv['Tccl']
        Pcgdl = [(sv[f'C_v_cgdl_{i}'] + sv[f'C_O2_cgdl_{i}'] + C_N2) * R * sv[f"Tcgdl_{i}"] for i in range(1, n_gdl + 1)]
        Pcgc = (C_v_cgc + C_O2_cgc + C_N2) * R * Tfc

        # Mean values ...
        #       ... of the saturated liquid water variable
        s_agdl_agdl = [None] + [sv[f's_agdl_{i}'] / 2 + sv[f's_agdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
        s_agdl_acl = sv[f's_agdl_{n_gdl}'] / 2 + s_acl / 2
        s_ccl_cgdl = s_ccl / 2 + sv['s_cgdl_1'] / 2
        s_cgdl_cgdl = [None] + [sv[f's_cgdl_{i}'] / 2 + sv[f's_cgdl_{i + 1}'] / 2 for i in range(1, n_gdl)]
        #       ... of the porosity and the contact angle
        epsilon_mean = epsilon_gdl / 2 + epsilon_cl / 2
        theta_c_mean = theta_c_gdl / 2 + theta_c_cl / 2
        #       ... of the dissolved water variable
        lambda_mem = [sv[f'lambda_mem_{i}'] for i in range(1, n_mem + 1)]
        #       ... of the pressure
        Pagc_agdl = Pagc / 2 + Pagdl[0] / 2
        Pagdl_agdl = [None] + [Pagdl[i] / 2 + Pagdl[i + 1] / 2 for i in range(0, n_gdl - 1)]
        Pagdl_acl = Pagdl[-1] / 2 + Pacl / 2
        Pccl_cgdl = Pccl / 2 + Pcgdl[0] / 2
        Pcgdl_cgdl = [None] + [Pcgdl[i] / 2 + Pcgdl[i + 1] / 2 for i in range(0, n_gdl - 1)]
        Pcgdl_cgc = Pcgdl[n_gdl - 1] / 2 + Pcgc / 2
        
        # Inlet and outlet flows (mol/s) or (kg/s)
        if Iload < I_aux_min and I_aux_min > 0:
            # Anode inlet
            Wrd = n_cell * M_H2 * Sa * (I_aux_min / Aact ) / (2 * F) * Aact  # kg.s-1
        else:
            Wrd = n_cell * M_H2 * Sa * (iload) / (2 * F) * Aact  # kg.s-1
        Wasm_in = Wrd + Wa_inj  # kg.s-1
        Wasm_out =  (Pasm - Pagc)  * Ksm_out  # kg.s-1
        Ja_in = Wasm_out / (Hgc * Wgc * Masm)  # mol.m-2.s-1
        # Anode outlet
        Waem_in = Kem_in * (Pagc - Paem)  # kg.s-1
        Ware = 0  # kg.s-1
        Waem_out = C_D * Abp_a * Paem / np.sqrt(R * Tfc) * Pr_aem ** (1 / gamma_H2) * \
                             np.sqrt(Magc * 2 * gamma_H2 / (gamma_H2 - 1) * (1 - Pr_aem ** ((gamma_H2 - 1) / gamma_H2)))
        # kg.s-1
        Ja_out = Waem_in / (Hgc * Wgc * Magc)  # mol.m-2.s-1
        # Cathode inlet         
        Wcsm_in = Wcp + Wc_inj  # kg.s-1
        Wcsm_out = (Pcsm - Pcgc) * Ksm_out  # kg.s-1
        Jc_in = Wcsm_out / (Hgc * Wgc * Mcsm)  # mol.m-2.s-1
        # Cathode outlet
        Wcem_in = Kem_in * (Pcgc - Pcem)  # kg.s-1
        Wcem_out = C_D * Abp_c * Pcem / np.sqrt(R * Tfc) * Pr_cem ** (1 / gamma) * \
                             np.sqrt(Mcgc * 2 * gamma / (gamma - 1) * (1 - Pr_cem ** ((gamma - 1) / gamma)))  # kg.s-1
        Jc_out = Wcem_in / (Hgc * Wgc * Mcgc)  # mol.m-2.s-1

        # Back pressure valve area
        if Abp_a > A_T:
            Abp_a = A_T
        elif Abp_a < 0:
            Abp_a = 0
        if Abp_c > A_T:
            Abp_c = A_T
        elif Abp_c < 0:
            Abp_c = 0

        #________________________________________Dissolved water flows (mol.m-2.s-1)_______________________________________
        # Anode side
        J_lambda_mem_acl = 2.5 / 22 * iload / F * lambda_acl - 2 * rho_mem / M_eq * D(lambda_acl, Tacl) * (
                                                lambda_mem[0] - lambda_acl) / (Hmem/ n_mem + Hcl)
        # Cathode side
        J_lambda_mem_ccl = 2.5 / 22 * iload / F * lambda_ccl - 2 * rho_mem / M_eq * D(lambda_ccl, Tccl) * (
                                                lambda_ccl - lambda_mem[-1]) / (Hmem/ n_mem + Hcl)
        # Membrane internal
        J_lambda_mem = [0] * (n_mem-1)
        for i in range(n_mem-1):
            J_lambda_mem[i] = 2.5 / 22 * iload / F * lambda_mem[i] - 2 * rho_mem / M_eq * D(lambda_mem[i], sv[f"Tmem_{i+1}"]) * (
                                                lambda_mem[i+1] - lambda_mem[i]) / (Hmem / n_mem)

        #_________________________________________Liquid water flows (kg.m-2.s-1)__________________________________________
        # Anode side
        Jl_agdl_agdl = [None] + [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            Jl_agdl_agdl[i] = - sigma(sv[f"Tagdl_{i}"]) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l(sv[f"Tagdl_{i}"]) * abs(np.cos(theta_c_gdl)) * \
                                         (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * \
                                         (s_agdl_agdl[i] ** e + 1e-7) * (1.417 - 4.24 * s_agdl_agdl[i] + 3.789 * s_agdl_agdl[i] ** 2) * \
                                         (sv[f's_agdl_{i + 1}'] - sv[f's_agdl_{i}']) / (Hgdl / n_gdl)
        Jl_agdl_acl = - 2 * sigma((sv[f"Tagdl_{n_gdl}"] + Tacl) / 2) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l((sv[f"Tagdl_{n_gdl}"] + Tacl) / 2) * abs(np.cos(theta_c_mean)) * \
                                (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * \
                                (s_agdl_acl ** e + 1e-7) * (1.417 - 4.24 * s_agdl_acl + 3.789 * s_agdl_acl ** 2) * \
                                (s_acl - sv[f's_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl)
        # Cathode side
        Jl_cgdl_cgdl = [None] + [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            Jl_cgdl_cgdl[i] = - sigma(sv[f"Tcgdl_{i}"]) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l(sv[f"Tcgdl_{i}"]) * abs(np.cos(theta_c_gdl)) * \
                                         (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * \
                                         (s_cgdl_cgdl[i] ** e + 1e-7) * (1.417 - 4.24 * s_cgdl_cgdl[i] + 3.789 * s_cgdl_cgdl[i] ** 2) * \
                                         (sv[f's_cgdl_{i + 1}'] - sv[f's_cgdl_{i}']) / (Hgdl / n_gdl)
        Jl_ccl_cgdl = - 2 * sigma((sv["Tcgdl_1"] + Tccl) / 2) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l((sv["Tcgdl_1"] + Tccl) / 2) * abs(np.cos(theta_c_mean)) * \
                                (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * \
                                (s_ccl_cgdl ** e + 1e-7) * (1.417 - 4.24 * s_ccl_cgdl + 3.789 * s_ccl_cgdl ** 2) * \
                                (sv['s_cgdl_1'] - s_ccl) / (Hgdl / n_gdl + Hcl)

        # _____________________________________________Vapor flows (mol.m-2.s-1)____________________________________________
        # Convective vapor flows
        #   Anode side
        Jv_agc_agdl = h_a(Pagc_agdl, Tfc, Wgc, Hgc) * (C_v_agc - sv['C_v_agdl_1'])
        #   Cathode side
        Jv_cgdl_cgc = h_c(Pcgdl_cgc, Tfc, Wgc, Hgc) * (sv[f'C_v_cgdl_{n_gdl}'] - C_v_cgc)

        # Conductive vapor flows
        #   Anode side
        Jv_a_in = Phi_asm * Psat(Tfc) / Pasm * Ja_in
        Jv_a_out = Phi_agc * Psat(Tfc) / Pagc * Ja_out
        Jv_agdl_agdl = [None] + [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            Jv_agdl_agdl[i] = - Da_eff(s_agdl_agdl[i], epsilon_gdl, Pagdl_agdl[i], sv[f"Tagdl_{i}"], epsilon_c, epsilon_gdl) * \
                                           (sv[f'C_v_agdl_{i + 1}'] - sv[f'C_v_agdl_{i}']) / (Hgdl / n_gdl)
        Jv_agdl_acl = - 2 * Da_eff(s_agdl_acl, epsilon_mean, Pagdl_acl, (sv[f"Tagdl_{n_gdl}"] + Tacl) / 2, epsilon_c, epsilon_gdl) * \
                                 (C_v_acl - sv[f'C_v_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl)
        Wv_asm_in = Wa_inj / M_H2O
        Wv_aem_out = Phi_aem * Psat(Tfc) / Paem * (Waem_out / Maem)

        #   Cathode side
        Jv_c_in = Phi_csm * Psat(Tfc) / Pcsm * Jc_in
        Jv_c_out = Phi_cgc * Psat(Tfc) / Pcgc * Jc_out
        Jv_cgdl_cgdl = [None] + [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            Jv_cgdl_cgdl[i] = - Dc_eff(s_cgdl_cgdl[i], epsilon_gdl, Pcgdl_cgdl[i], sv[f"Tcgdl_{i}"], epsilon_c, epsilon_gdl) * \
                                           (sv[f'C_v_cgdl_{i + 1}'] - sv[f'C_v_cgdl_{i}']) / (Hgdl / n_gdl)
        Jv_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, (sv[f"Tcgdl_1"] + Tccl) / 2, epsilon_c, epsilon_gdl) * \
                                (sv['C_v_cgdl_1'] - C_v_ccl) / (Hgdl / n_gdl + Hcl)
        Wv_csm_in = Phi_ext * Psat(Text) / Pext * (Wcp / Mext) + Wc_inj / M_H2O
        Wv_cem_out = Phi_cem * Psat(Tfc) / Pcem * (Wcem_out / Mcem)

        # __________________________________________H2 and O2 flows (mol.m-2.s-1)___________________________________________
        # Hydrogen and oxygen consumption
        # Anode side
        S_H2_acl = - iload / (2 * F * Hcl) 
        # Cathode side
        S_O2_ccl = - iload / (4 * F * Hcl) 

        # Conductive-convective H2 and O2 flows
        #   Anode side
        J_H2_in = (1 - Phi_asm * Psat(Tfc) / Pasm) * Ja_in
        J_H2_out = (1 - Phi_agc * Psat(Tfc) / Pagc) * Ja_out
        J_H2_agc_agdl = h_a(Pagc_agdl, Tfc, Wgc, Hgc) * (C_H2_agc - sv['C_H2_agdl_1'])
        #   Cathode side
        J_O2_in = yO2_ext * (1 - Phi_csm * Psat(Tfc) / Pcsm) * Jc_in
        J_O2_out = y_cgc * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * Jc_out
        J_N2_in = (1 - yO2_ext) * (1 - Phi_csm * Psat(Tfc) / Pcsm) * Jc_in
        J_N2_out = (1 - y_cgc) * (1 - Phi_cgc * Psat(Tfc) / Pcgc) * Jc_out
        J_O2_cgdl_cgc = h_c(Pcgdl_cgc, Tfc, Wgc, Hgc) * (sv[f'C_O2_cgdl_{n_gdl}'] - C_O2_cgc)

        # Conductive H2 and O2 flows
        # Anode side
        J_H2_agdl_agdl = [None] + [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            J_H2_agdl_agdl[i] = - Da_eff(s_agdl_agdl[i], epsilon_gdl, Pagdl_agdl[i], sv[f"Tagdl_{i}"], epsilon_c, epsilon_gdl) * \
                                                (sv[f'C_H2_agdl_{i + 1}'] - sv[f'C_H2_agdl_{i}']) / (Hgdl / n_gdl)
        J_H2_agdl_acl = - 2 * Da_eff(s_agdl_acl, epsilon_mean, Pagdl_acl, (sv[f"Tagdl_{n_gdl}"] + Tacl) / 2, epsilon_c, epsilon_gdl) * \
                                      (C_H2_acl - sv[f'C_H2_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl)

        #   Cathode side
        J_O2_cgdl_cgdl = [None] + [0] * (n_gdl - 1)
        for i in range(1, n_gdl):
            J_O2_cgdl_cgdl[i] = - Dc_eff(s_cgdl_cgdl[i], epsilon_gdl, Pcgdl_cgdl[i], sv[f"Tcgdl_{i}"], epsilon_c, epsilon_gdl) * \
                                                (sv[f'C_O2_cgdl_{i + 1}'] - sv[f'C_O2_cgdl_{i}']) / (Hgdl / n_gdl)
        J_O2_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, (sv[f"Tcgdl_1"] + Tccl) / 2, epsilon_c, epsilon_gdl) * \
                                     (sv['C_O2_cgdl_1'] - C_O2_ccl) / (Hgdl / n_gdl + Hcl)

        # __________________________________________Water generated (mol.m-3.s-1)___________________________________________
        # Water produced in the membrane at the CL through the chemical reaction and crossover
        #   Anode side
        Sp_acl = 0
        #   Cathode side
        Sp_ccl = iload / (2 * F * Hcl)

        # Water sorption in the CL:
        #   Anode side
        S_sorp_acl = gamma_sorp(C_v_acl, s_acl, lambda_acl, Tacl, Hcl, Kshape) * rho_mem / M_eq * \
                                (lambda_eq(C_v_acl, s_acl, Tacl, Kshape) - lambda_acl)
        #   Cathode side
        S_sorp_ccl = gamma_sorp(C_v_ccl, s_ccl, lambda_ccl, Tccl, Hcl, Kshape) * rho_mem / M_eq * \
                                (lambda_eq(C_v_ccl, s_ccl, Tccl, Kshape) - lambda_ccl)

        # Liquid water generated through vapor condensation or degenerated through evaporation
        #   Anode side
        Sl_agdl = [None] + [Svl(sv[f's_agdl_{i}'], sv[f'C_v_agdl_{i}'], sv[f'C_v_agdl_{i}'] + sv[f'C_H2_agdl_{i}'],
                          epsilon_gdl, sv[f"Tagdl_{i}"], gamma_cond, gamma_evap) for i in range(1, n_gdl + 1)]
        Sl_acl = Svl(s_acl, C_v_acl, C_v_acl + C_H2_acl, epsilon_cl, Tacl, gamma_cond, gamma_evap)
        #   Cathode side
        Sl_cgdl = [None] + [Svl(sv[f's_cgdl_{i}'], sv[f'C_v_cgdl_{i}'], sv[f'C_v_cgdl_{i}'] + sv[f'C_O2_cgdl_{i}'] + C_N2,
                         epsilon_gdl, sv[f"Tcgdl_{i}"], gamma_cond, gamma_evap) for i in range(1, n_gdl + 1)]
        Sl_ccl = Svl(s_ccl, C_v_ccl, C_v_ccl + C_O2_ccl + C_N2, epsilon_cl, Tccl, gamma_cond, gamma_evap)

        # Vapor generated through liquid water evaporation or degenerated through condensation
        #   Anode side
        Sv_agdl = [None] + [-S for S in Sl_agdl[1:]]
        Sv_acl = - Sl_acl
        #   Cathode side
        Sv_cgdl = [None] + [-S for S in Sl_cgdl[1:]]
        Sv_ccl = - Sl_ccl

        # Platinum dissolution
        R_H20 = 1.8e-5
        R_iono = 5.56e-4
        J_O2_mem = np.zeros(n_mem).tolist()
        J_H2_mem = np.zeros(n_mem).tolist()
        J_Pt2_mem = np.zeros(n_mem).tolist()
        S_Pt2_mem = np.zeros(n_mem).tolist()
        J_O2_mem_ccl = 0
        J_H2_acl_mem = 0
        epsilon_cm = 1 - self.parameters["epsilon_mc"]

        for i_node in np.arange(1, n_mem):

            x_H2O = (sv[f"lambda_mem_{i_node+1}"] * R_H20) / (R_iono + sv[f"lambda_mem_{i_node+1}"] * R_H20)
            k_H2_mem = (0.29 + 2.2 * x_H2O) * 1e-15 * np.exp(2.1e4 / R * (1 / 303 - 1 / sv[f"Tmem_{i_node+1}"]))
            k_O2_mem = (0.11 + 1.9 * x_H2O) * 1e-15 * np.exp(2.2e4 / R * (1 / 303 - 1 / sv[f"Tmem_{i_node+1}"]))
            D_H2 = 2.584 * np.exp(170 / sv[f"Tmem_{i_node+1}"]) * k_O2_mem
            D_O2 = 1348 * np.exp(-666 / sv[f"Tmem_{i_node+1}"]) * k_H2_mem
            D_Pt2 = 1e-13 * (sv[f"lambda_mem_{i_node+1}"] * R_H20) / (R_H20 + sv[f"lambda_mem_{i_node+1}"] * R_iono)  # cm2/s
            k_Pt2 = 4 * np.pi * D_H2 * N_A * 0.15e-9
            if i_node == 1:
                #J_Pt2_mem[0] = -D_Pt2 * (sv["C_Pt2_mem_1"] - 0) * epsilon_cm **1.5 / (Hmem / n_mem + Hcl)
                J_O2_mem[0] = -D_O2 * (sv["C_O2_mem_1"] - 0) / (Hmem / n_mem)
                J_H2_mem[0] = -D_H2 * (sv[f"C_H2_mem_1"] - 0) / (Hmem / n_mem)-D_H2 * (sv[f"C_H2_mem_{i_node + 1}"] - sv[f"C_H2_mem_{i_node}"]) / (Hmem / n_mem)
                J_H2_acl_mem = -D_H2 * (sv[f"C_H2_mem_1"] - C_H2_acl) / (Hmem / n_mem + Hcl)
            elif i_node == n_mem - 1:
                J_H2_mem[i_node] = -D_H2 * (0 - sv[f"C_H2_mem_{n_mem}"]) / (Hmem / n_mem)
                J_O2_mem_ccl = -D_O2 * (C_O2_ccl - sv[f"C_O2_mem_{n_mem}"]) / (Hmem / n_mem + Hcl)
                J_O2_mem[i_node] = -D_O2 * (sv[f"C_O2_mem_{i_node + 1}"] - sv[f"C_O2_mem_{i_node}"]) / (Hmem / n_mem)
                #J_Pt2_mem[i_node] = -D_Pt2 * (sv["C_Pt2_ccl"] - sv[f"C_Pt2_mem_{n_mem}"]) * epsilon_cm ** 1.5 / (Hmem / n_mem + Hcl)
            else:
                #S_Pt2_mem[i_node] = 0 # -k_Pt2 * sv[f"C_Pt2_mem_{i_node + 1}"] * sv[f"C_H2_mem_{i_node + 1}"]
                #J_Pt2_mem[i_node] = -D_Pt2 * (sv[f"C_Pt2_mem_{i_node + 1}"] - sv[f"C_Pt2_mem_{i_node}"]) / (Hmem / n_mem)
                J_O2_mem[i_node] = -D_O2 * (sv[f"C_O2_mem_{i_node + 1}"] - sv[f"C_O2_mem_{i_node}"]) / (Hmem / n_mem)
                J_H2_mem[i_node] = -D_H2 * (sv[f"C_H2_mem_{i_node + 1}"] - sv[f"C_H2_mem_{i_node}"]) / (Hmem / n_mem)

        return {'Jv_a_in': Jv_a_in, 'Jv_a_out': Jv_a_out, 'Jv_c_in': Jv_c_in, 'Jv_c_out': Jv_c_out, 'J_H2_in': J_H2_in,
                      'J_H2_out': J_H2_out, 'J_O2_in': J_O2_in, 'J_O2_out': J_O2_out, 'J_N2_in': J_N2_in,
                      'J_N2_out': J_N2_out,
                      'J_H2_mem': J_H2_mem, 'J_O2_mem':J_O2_mem, "J_O2_mem_ccl":J_O2_mem_ccl, "J_H2_acl_mem":J_H2_acl_mem,
                      'J_Pt2_mem':J_Pt2_mem,"S_Pt2_mem":S_Pt2_mem,
                      'Jv_agc_agdl': Jv_agc_agdl, 'Jv_agdl_agdl': Jv_agdl_agdl, 'Jv_agdl_acl': Jv_agdl_acl,
                      'Jv_ccl_cgdl': Jv_ccl_cgdl, 'Jv_cgdl_cgdl': Jv_cgdl_cgdl, 'Jv_cgdl_cgc': Jv_cgdl_cgc,
                      'Jl_agdl_acl': Jl_agdl_acl, 'Jl_ccl_cgdl': Jl_ccl_cgdl, 'Jl_cgdl_cgdl': Jl_cgdl_cgdl, 'Jl_agdl_agdl': Jl_agdl_agdl,
                      'J_lambda_mem_acl': J_lambda_mem_acl, 'J_lambda_mem_ccl': J_lambda_mem_ccl, "J_lambda_mem": J_lambda_mem,
                      'J_H2_agc_agdl': J_H2_agc_agdl, 'J_H2_agdl_agdl': J_H2_agdl_agdl, 'J_H2_agdl_acl': J_H2_agdl_acl,
                      'J_O2_ccl_cgdl': J_O2_ccl_cgdl, 'J_O2_cgdl_cgdl': J_O2_cgdl_cgdl, 'J_O2_cgdl_cgc': J_O2_cgdl_cgc,
                      'S_sorp_acl': S_sorp_acl, 'S_sorp_ccl': S_sorp_ccl,  'Sp_acl': Sp_acl, 'Sp_ccl': Sp_ccl,
                      'S_H2_acl': S_H2_acl, 'S_O2_ccl': S_O2_ccl, 'Sv_agdl': Sv_agdl, 'Sv_acl': Sv_acl, 'Sv_ccl': Sv_ccl,
                      'Sv_cgdl': Sv_cgdl, 'Sl_agdl': Sl_agdl, 'Sl_acl': Sl_acl, 'Sl_ccl': Sl_ccl, 'Sl_cgdl': Sl_cgdl,
                      'Pagc': Pagc, 'Pcgc': Pcgc, 
                      'Wasm_in': Wasm_in, 'Wasm_out': Wasm_out, 'Waem_in': Waem_in, 'Waem_out': Waem_out, 
                      'Wcsm_in': Wcsm_in, 'Wcsm_out': Wcsm_out, 'Wcem_in': Wcem_in, 'Wcem_out': Wcem_out,
                      'Ware': Ware, 'Wv_asm_in': Wv_asm_in, 'Wv_aem_out': Wv_aem_out, 'Wv_csm_in': Wv_csm_in,
                      'Wv_cem_out': Wv_cem_out}


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

        self.variables['ecsa'].extend(list(sol.y[index]))

        # Recovery of more variables
        #   The control variables should be reinitialized. To be reviewed.

        self.control_variables['Phi_a_des'] = self.operating_inputs['Phi_a_des']
        self.control_variables['Phi_c_des'] = self.operating_inputs['Phi_c_des']

        for j in range(len(sol.t)):  # For each time...
            # ... recovery of i_fc.
            i_fc = self.operating_inputs["current_density"](self.variables['t'][j])
            self.loadprofile.append(i_fc * self.parameters["Aact"])
            last_solver_variables = {key: self.variables[key][j] for key in self.solver_variable_names}
            flows_recovery = self.updateMassFlow(self.variables['t'][j], last_solver_variables, i_fc,
                                                                                   self.operating_inputs, self.parameters, self.Imin_aux)
            for key in ['S_sorp_acl', 'S_sorp_ccl', 'J_lambda_mem_acl', 'J_lambda_mem_ccl', 'Pagc', 'Pcgc',"Wasm_in"]:
                self.variables[key].append(flows_recovery[key])
            prd_ccl = [last_solver_variables[f"S_N_ccl_{i + 1}"] for i in range(self.micro_parameters["n_group_ptParticle"])]
            theta_ccl =  [last_solver_variables[f"theta_ccl_{i + 1}"] for i in range(self.micro_parameters["n_group_ptParticle"])]
            self.variables["ecsa"][j] = getECSA(prd_ccl, self.micro_parameters["r_m"])
            #  recovery of Ucell.
            Ueq_t, Rmem_t, Rccl_t, Racl_t = calculate_cell_voltage_intermediate(last_solver_variables, self.parameters)
            Rp_t = sum(Rmem_t)  + Rccl_t
            eta_c_component = calculate_eta_c_intermediate_values(last_solver_variables, self.operating_inputs, self.parameters)
            f_drop_t = eta_c_component["f_drop"]
            if f_drop_t == 1:
                self.elec_variables["eta_act"].append(self.variables["eta_c"][j])
                self.elec_variables["eta_conc"].append(0)
            else:
                eta_conc_t = self.variables["eta_c"][j] * (1 - f_drop_t)/f_drop_t
                eta_act_t = self.variables["eta_c"][j] - eta_conc_t
                self.elec_variables["eta_act"].append(eta_act_t)
                self.elec_variables["eta_conc"].append(eta_conc_t)
            self.elec_variables["i_fc"].append(i_fc)
            self.elec_variables["fdrop"].append(f_drop_t)
            self.elec_variables["Ueq"].append(Ueq_t)
            self.elec_variables["Rp"].append(Rp_t)

        self.variables["Ucell"].extend(calculate_cell_voltage(self.ECSA_norm, self.variables, self.operating_inputs,self.parameters))
        

def calculate_flows(t, sv, control_variables, i_fc, I_aux_min, operating_inputs, parameters):
    """This function calculates the flows inside the fuel cell system.

    Parameters
    ----------
    t : float
        Time (s).
    sv : dict
        Variables calculated by the solver. They correspond to the fuel cell internal states.
        sv is a contraction of solver_variables for enhanced readability.
    control_variables : dict
        Variables controlled by the user.
    i_fc : float
        Fuel cell current density at time t (A.m-2).
    operating_inputs : dict
        Operating inputs of the fuel cell.
    parameters : dict
        Parameters of the fuel cell model.

    Returns
    -------
    dict
        Flows inside the fuel cell system.
    """

    # ___________________________________________________Preliminaries__________________________________________________

    # Extraction of the variables
    C_v_agc, C_v_acl, C_v_ccl, C_v_cgc = sv['C_v_agc'], sv['C_v_acl'], sv['C_v_ccl'], sv['C_v_cgc']
    s_acl, s_ccl = sv['s_acl'], sv['s_ccl']
    lambda_acl, lambda_ccl = sv['lambda_acl'], sv['lambda_ccl']
    C_H2_agc, C_H2_acl, C_O2_ccl, C_O2_cgc = sv['C_H2_agc'], sv['C_H2_acl'], sv['C_O2_ccl'], sv['C_O2_cgc']
    C_N2 = sv['C_N2']
    Tccl, Tcgdl = sv['Tccl'], sv['Tcgdl']

    # Extraction of the operating inputs and parameters
    Tfc, Pc_des = operating_inputs['Tfc'], operating_inputs['Pc_des']
    Hgdl, Hmem, Hcl = parameters['Hgdl'], parameters['Hmem'], parameters['Hcl']
    Hgc, Wgc = parameters['Hgc'], parameters['Wgc']
    epsilon_gdl, epsilon_c, epsilon_cl = parameters['epsilon_gdl'], parameters['epsilon_c'], parameters['epsilon_cl']   
    e, kappa_co, n_gdl = parameters['e'], parameters['kappa_co'], parameters['n_gdl']
    a_slim, b_slim = parameters['a_slim'], parameters['b_slim']

    # Intermediate values
    (Pagc, Pcgc, s_agdl_agdl, s_agdl_acl, s_ccl_cgdl, s_cgdl_cgdl, epsilon_mean, theta_c_mean, lambda_acl_mem,
     lambda_mem_ccl, Pagc_agdl, Pagdl_agdl, Pagdl_acl, Pccl_cgdl, Pcgdl_cgdl, Pcgdl_cgc, nu_l) \
        = flows_int_values(sv, operating_inputs, parameters)

    # Inlet and outlet flows
    Jv_a_in, Jv_a_out, Jv_c_in, Jv_c_out, J_H2_in, J_H2_out, J_O2_in, J_O2_out, J_N2_in, J_N2_out, \
        Wasm_in, Wasm_out, Waem_in, Waem_out, Wcsm_in, Wcsm_out, Wcem_in, Wcem_out, Ware, \
        Wv_asm_in, Wv_aem_out, Wv_csm_in, Wv_cem_out \
        = auxiliaries(t, sv, control_variables, i_fc, I_aux_min, operating_inputs, parameters)

    # ________________________________________Dissolved water flows (mol.m-2.s-1)_______________________________________

    # Anode side
    J_lambda_mem_acl = 2.5 / 22 * i_fc / F * lambda_acl_mem - 2 * rho_mem / M_eq * \
               D(lambda_acl_mem, Tccl) * (lambda_mem - lambda_acl) / (Hmem + Hcl)
    # Cathode side
    J_lambda_mem_ccl = 2.5 / 22 * i_fc / F * lambda_mem_ccl - 2 * rho_mem / M_eq * \
               D(lambda_mem_ccl, Tccl) * (lambda_ccl - lambda_mem) / (Hmem + Hcl)

    # _________________________________________Liquid water flows (kg.m-2.s-1)__________________________________________

    # Anode side
    Jl_agdl_agdl = [None] + [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jl_agdl_agdl[i] = - sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l * abs(np.cos(theta_c_gdl)) * \
                          (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * \
                          (s_agdl_agdl[i] ** e + 1e-7) * (1.417 - 4.24 * s_agdl_agdl[i] + 3.789 * s_agdl_agdl[i] ** 2) * \
                          (sv[f's_agdl_{i + 1}'] - sv[f's_agdl_{i}']) / (Hgdl / n_gdl)
    Jl_agdl_acl = - 2 * sigma(Tfc) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l * abs(np.cos(theta_c_mean)) * \
                  (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * \
                  (s_agdl_acl ** e + 1e-7) * (1.417 - 4.24 * s_agdl_acl + 3.789 * s_agdl_acl ** 2) * \
                  (s_acl - sv[f's_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl)

    # Cathode side
    Jl_cgdl_cgdl = [None] + [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jl_cgdl_cgdl[i] = - sigma(Tfc) * K0(epsilon_gdl, epsilon_c, epsilon_gdl) / nu_l * abs(np.cos(theta_c_gdl)) * \
                          (epsilon_gdl / K0(epsilon_gdl, epsilon_c, epsilon_gdl)) ** 0.5 * \
                          (s_cgdl_cgdl[i] ** e + 1e-7) * (1.417 - 4.24 * s_cgdl_cgdl[i] + 3.789 * s_cgdl_cgdl[i] ** 2) * \
                          (sv[f's_cgdl_{i + 1}'] - sv[f's_cgdl_{i}']) / (Hgdl / n_gdl)
    Jl_ccl_cgdl = - 2 * sigma(Tfc) * K0(epsilon_mean, epsilon_c, epsilon_gdl) / nu_l * abs(np.cos(theta_c_mean)) * \
                  (epsilon_mean / K0(epsilon_mean, epsilon_c, epsilon_gdl)) ** 0.5 * \
                  (s_ccl_cgdl ** e + 1e-7) * (1.417 - 4.24 * s_ccl_cgdl + 3.789 * s_ccl_cgdl ** 2) * \
                  (sv['s_cgdl_1'] - s_ccl) / (Hgdl / n_gdl + Hcl)

    # _____________________________________________Vapor flows (mol.m-2.s-1)____________________________________________

    # Convective vapor flows
    #   Anode side
    Jv_agc_agdl = h_a(Pagc_agdl, Tfc, Wgc, Hgc) * (C_v_agc - sv['C_v_agdl_1'])
    #   Cathode side
    Jv_cgdl_cgc = h_c(Pcgdl_cgc, Tfc, Wgc, Hgc) * (sv[f'C_v_cgdl_{n_gdl}'] - C_v_cgc)

    # Conductive vapor flows
    #   Anode side
    Jv_agdl_agdl = [None] + [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jv_agdl_agdl[i] = - Da_eff(s_agdl_agdl[i], epsilon_gdl, Pagdl_agdl[i], Tfc, epsilon_c, epsilon_gdl) * \
                          (sv[f'C_v_agdl_{i + 1}'] - sv[f'C_v_agdl_{i}']) / (Hgdl / n_gdl)
    Jv_agdl_acl = - 2 * Da_eff(s_agdl_acl, epsilon_mean, Pagdl_acl, Tfc, epsilon_c, epsilon_gdl) * \
                  (C_v_acl - sv[f'C_v_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl)

    #   Cathode side
    Jv_cgdl_cgdl = [None] + [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        Jv_cgdl_cgdl[i] = - Dc_eff(s_cgdl_cgdl[i], epsilon_gdl, Pcgdl_cgdl[i], Tfc, epsilon_c, epsilon_gdl) * \
                          (sv[f'C_v_cgdl_{i + 1}'] - sv[f'C_v_cgdl_{i}']) / (Hgdl / n_gdl)
    Jv_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, Tfc, epsilon_c, epsilon_gdl) * \
                  (sv['C_v_cgdl_1'] - C_v_ccl) / (Hgdl / n_gdl + Hcl)

    # __________________________________________H2 and O2 flows (mol.m-2.s-1)___________________________________________

    # Hydrogen and oxygen consumption
    #   Anode side
    S_H2_acl = - i_fc / (2 * F * Hcl) - \
               R * Tfc / (Hmem * Hcl) * (k_H2(lambda_mem, Tfc, kappa_co) * C_H2_acl +
                                         2 * k_O2(lambda_mem, Tfc, kappa_co) * C_O2_ccl)
    #   Cathode side
    S_O2_ccl = - i_fc / (4 * F * Hcl) - \
               R * Tfc / (Hmem * Hcl) * (k_O2(lambda_mem, Tfc, kappa_co) * C_O2_ccl +
                                         1 / 2 * k_H2(lambda_mem, Tfc, kappa_co) * C_H2_acl)

    # Conductive-convective H2 and O2 flows
    #   Anode side
    J_H2_agc_agdl = h_a(Pagc_agdl, Tfc, Wgc, Hgc) * (C_H2_agc - sv['C_H2_agdl_1'])
    #   Cathode side
    J_O2_cgdl_cgc = h_c(Pcgdl_cgc, Tfc, Wgc, Hgc) * (sv[f'C_O2_cgdl_{n_gdl}'] - C_O2_cgc)

    # Conductive H2 and O2 flows
    #   Anode side
    J_H2_agdl_agdl = [None] + [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        J_H2_agdl_agdl[i] = - Da_eff(s_agdl_agdl[i], epsilon_gdl, Pagdl_agdl[i], Tfc, epsilon_c, epsilon_gdl) * \
                            (sv[f'C_H2_agdl_{i + 1}'] - sv[f'C_H2_agdl_{i}']) / (Hgdl / n_gdl)
    J_H2_agdl_acl = - 2 * Da_eff(s_agdl_acl, epsilon_mean, Pagdl_acl, Tfc, epsilon_c, epsilon_gdl) * \
                    (C_H2_acl - sv[f'C_H2_agdl_{n_gdl}']) / (Hgdl / n_gdl + Hcl)

    #   Cathode side
    J_O2_cgdl_cgdl = [None] + [0] * (n_gdl - 1)
    for i in range(1, n_gdl):
        J_O2_cgdl_cgdl[i] = - Dc_eff(s_cgdl_cgdl[i], epsilon_gdl, Pcgdl_cgdl[i], Tfc, epsilon_c, epsilon_gdl) * \
                            (sv[f'C_O2_cgdl_{i + 1}'] - sv[f'C_O2_cgdl_{i}']) / (Hgdl / n_gdl)
    J_O2_ccl_cgdl = - 2 * Dc_eff(s_ccl_cgdl, epsilon_mean, Pccl_cgdl, Tfc, epsilon_c, epsilon_gdl) * \
                    (sv['C_O2_cgdl_1'] - C_O2_ccl) / (Hgdl / n_gdl + Hcl)

    # __________________________________________Water generated (mol.m-3.s-1)___________________________________________

    # Water produced in the membrane at the CL through the chemical reaction and crossover
    #   Anode side
    Sp_acl = 2 * k_O2(lambda_mem, Tfc, kappa_co) * R * Tfc / (Hmem * Hcl) * C_O2_ccl
    #   Cathode side
    Sp_ccl = i_fc / (2 * F * Hcl) + k_H2(lambda_mem, Tfc, kappa_co) * R * Tfc / (Hmem * Hcl) * C_H2_acl

    # Water sorption in the CL:
    #   Anode side
    S_sorp_acl = gamma_sorp(C_v_acl, s_acl, lambda_acl, Tfc, Hcl, Kshape) * rho_mem / M_eq * \
                 (lambda_eq(C_v_acl, s_acl, Tfc, Kshape) - lambda_acl)
    #   Cathode side
    S_sorp_ccl = gamma_sorp(C_v_ccl, s_ccl, lambda_ccl, Tfc, Hcl, Kshape) * rho_mem / M_eq * \
                 (lambda_eq(C_v_ccl, s_ccl, Tfc, Kshape) - lambda_ccl)

    # Liquid water generated through vapor condensation or degenerated through evaporation
    #   Anode side
    Sl_agdl = [None] + [Svl(sv[f's_agdl_{i}'], sv[f'C_v_agdl_{i}'], sv[f'C_v_agdl_{i}'] + sv[f'C_H2_agdl_{i}'],
                            epsilon_gdl, Tfc, gamma_cond, gamma_evap) for i in range(1, n_gdl + 1)]
    Sl_acl = Svl(s_acl, C_v_acl, C_v_acl + C_H2_acl, epsilon_cl, Tfc, gamma_cond, gamma_evap)
    #   Cathode side
    Sl_cgdl = [None] + [Svl(sv[f's_cgdl_{i}'], sv[f'C_v_cgdl_{i}'], sv[f'C_v_cgdl_{i}'] + sv[f'C_O2_cgdl_{i}'] + C_N2,
                            epsilon_gdl, Tfc, gamma_cond, gamma_evap) for i in range(1, n_gdl + 1)]
    Sl_ccl = Svl(s_ccl, C_v_ccl, C_v_ccl + C_O2_ccl + C_N2, epsilon_cl, Tfc, gamma_cond, gamma_evap)

    # Vapor generated through liquid water evaporation or degenerated through condensation
    #   Anode side
    Sv_agdl = [None] + [-x for x in Sl_agdl[1:]]
    Sv_acl = - Sl_acl
    #   Cathode side
    Sv_cgdl = [None] + [-x for x in Sl_cgdl[1:]]
    Sv_ccl = - Sl_ccl

    return {'Jv_a_in': Jv_a_in, 'Jv_a_out': Jv_a_out, 'Jv_c_in': Jv_c_in, 'Jv_c_out': Jv_c_out, 'J_H2_in': J_H2_in,
                'J_H2_out': J_H2_out, 'J_O2_in': J_O2_in, 'J_O2_out': J_O2_out, 'J_N2_in': J_N2_in, 'J_N2_out': J_N2_out,
                'Jv_agc_agdl': Jv_agc_agdl, 'Jv_agdl_agdl': Jv_agdl_agdl, 'Jv_agdl_acl': Jv_agdl_acl,
                'S_sorp_acl': S_sorp_acl, 'S_sorp_ccl': S_sorp_ccl, 'Jv_ccl_cgdl': Jv_ccl_cgdl,
                'Jv_cgdl_cgdl': Jv_cgdl_cgdl, 'Jv_cgdl_cgc': Jv_cgdl_cgc, 'Jl_agdl_agdl': Jl_agdl_agdl,
                'Jl_agdl_acl': Jl_agdl_acl, 'J_lambda_mem_acl': J_lambda_mem_acl, 'J_lambda_mem_ccl': J_lambda_mem_ccl,
                'Jl_ccl_cgdl': Jl_ccl_cgdl, 'Jl_cgdl_cgdl': Jl_cgdl_cgdl, 'Sp_acl': Sp_acl, 'Sp_ccl': Sp_ccl,
                'J_H2_agc_agdl': J_H2_agc_agdl, 'J_H2_agdl_agdl': J_H2_agdl_agdl, 'J_H2_agdl_acl': J_H2_agdl_acl,
                'J_O2_ccl_cgdl': J_O2_ccl_cgdl, 'J_O2_cgdl_cgdl': J_O2_cgdl_cgdl, 'J_O2_cgdl_cgc': J_O2_cgdl_cgc,
                'S_H2_acl': S_H2_acl, 'S_O2_ccl': S_O2_ccl, 'Sv_agdl': Sv_agdl, 'Sv_acl': Sv_acl, 'Sv_ccl': Sv_ccl,
                'Sv_cgdl': Sv_cgdl, 'Sl_agdl': Sl_agdl, 'Sl_acl': Sl_acl, 'Sl_ccl': Sl_ccl, 'Sl_cgdl': Sl_cgdl,
                'Pagc': Pagc, 'Pcgc': Pcgc, 'Wasm_in': Wasm_in, 'Wasm_out': Wasm_out, 'Waem_in': Waem_in,
                'Waem_out': Waem_out, 'Wcsm_in': Wcsm_in, 'Wcsm_out': Wcsm_out, 'Wcem_in': Wcem_in, 'Wcem_out': Wcem_out,
                'Ware': Ware, 'Wv_asm_in': Wv_asm_in, 'Wv_aem_out': Wv_aem_out, 'Wv_csm_in': Wv_csm_in,
                'Wv_cem_out': Wv_cem_out}