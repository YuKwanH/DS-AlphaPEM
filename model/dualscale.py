from model.coefficients import *
from model.inst_values import *
from model.state_eq import *
from model.kinetic_eq import *

class PEMFC:

        def __init__(self, param, operating_inputs, 
                             variable_names, flux_names):

                self.parameters = param
                self.variable_names = variable_names
                self.flux_names = flux_names
                self.operating_inputs = operating_inputs

                # GDL nodes name discretization
                discretized_region = ['C_v_agdl', 'C_v_cgdl', 's_agdl', 's_cgdl', 'C_H2_agdl', 'C_O2_cgdl', "Tcgdl", "Tagdl"]
                for variable in discretized_region:
                        index = self.variable_names.index(variable)
                # Delete the previous points
                        self.variable_names.pop(index)
                        # Increase the number of points
                        self.variable_names[index:index] = [f'{variable}_{i}' for i in range(1, self.parameters['n_gdl'] + 1)]
                # MEM nodes name discretization
                for name in ["C_O2_mem", "C_H2_mem", "C_Pt2_mem", "lambda_mem","Tmem"]:
                        index = self.variable_names.index(name)
                        self.variable_names.pop(index)
                        self.variable_names[index:index] = [f'{name}_{i}' for i in range(1, self.parameters['n_mem'] + 1)]
                # Micro-scale variables
                dr = 1e-6 / self.parameters['n_group_pt']
                self.parameters['r_m'] = (np.linspace(1e-8, 1e-6, self.parameters['n_group_pt'] + 1) + dr / 2)[1:]
                self.parameters['prd0'] = initPRD(resolution=self.parameters['n_group_pt'])
                index_prd = self.variable_names.index('S_N_ccl')
                self.variable_names.pop(index_prd)
                self.variable_names[index_prd:index_prd] = [f'S_N_ccl_{i}' for i in range(1, len(self.parameters['r_m']) + 1)]
                index_theta_ccl = self.variable_names.index('theta_ccl')
                self.variable_names.pop(index_theta_ccl)
                self.variable_names[index_theta_ccl:index_theta_ccl] = [f'theta_ccl_{i}' for i in range(1, len(self.parameters['r_m']) + 1)]

                # Initialization of the variables dictionary to store the results of the simulation
                self.variables = {key: [] for key in self.variable_names}
                self.variables['t'] = []
                self.fluxes = {key: [] for key in self.flux_names}
                self.fluxes['t'] = []
                self.echem_traj = {key: [] for key in ['Ucell','eta_act', 'eta_conc', 'i_fc', 'fdrop', 'Ueq', 'Rmem', 'Rccl', 'Racl', 'S_N', 'PRD']}
                self.echem_traj['t'] = []

        def dxdt(self, t, x):
                
                # Create state gradients dictionary
                dif = {('d' + key + ' / dt'): 0 for key in self.variable_names}
                # Mapping macro-scale variables
                states = {}
                for name in self.variable_names:
                        states[name] = x[self.variable_names.index(name)]

                inst_states = dif_eq_int_values(t=t, x = states, operating_inputs= self.operating_inputs, parameters=self.parameters)
                massflow = calculate_flows(t, states, self.operating_inputs, self.parameters, **inst_states)

                dxdt_AGC(dif, **inst_states, **self.parameters, **massflow)
                dxdt_CGC(dif,  **inst_states, **self.parameters, **massflow)
                dxdt_AGDL(dif, states, **inst_states, **self.parameters, **massflow)
                dxdt_CGDL(dif, states, **inst_states, **self.parameters, **massflow)
                dxdt_ACL(dif, states, **inst_states, **self.parameters, **massflow)
                dxdt_CCL(dif,  states, **inst_states, **self.parameters, **massflow)
                dxdt_MEM(dif, **inst_states, **self.parameters, **massflow)
                dxdt_CP(dif, **states, **inst_states, **self.parameters, **massflow)
                dxdt_Manifold(dif, **inst_states, **self.parameters, **massflow, **self.operating_inputs)
                dxdt_TH(dif, **states, **inst_states, **self.parameters, **massflow,  **self.operating_inputs)
                dxdt_U(dif, **states, **inst_states, **self.parameters, **massflow)
                dxdt_N2(dif, **inst_states, **self.parameters, **massflow)
                dxdt_PRD(dif = dif, **states, **inst_states, **self.parameters, **massflow)

                return list(dif.values())
                
        def _recovery(self, sol):

                # Recovery of the time span
                self.variables['t'].extend(list(sol.t))
                self.echem_traj['t'].extend(list(sol.t))

                # Recovery of the main variables dynamic evolution
                for index, key in enumerate(self.variable_names):
                        self.variables[key].extend(list(sol.y[index]))
                
                for j in range(len(sol.t)):  # For each time...
                        t = sol.t[j]
                        # ... recovery of i_fc.
                        i_fc = self.operating_inputs["current_density"](self.variables['t'][j])
                        last_solver_variables = {key: self.variables[key][j] for key in self.variable_names}
                        inst_states_t = dif_eq_int_values(t=t, x = last_solver_variables, operating_inputs = self.operating_inputs, parameters = self.parameters)
                        flows_recovery = calculate_flows(t=t, x = last_solver_variables, operating_inputs = self.operating_inputs, parameters = self.parameters, **inst_states_t)
                        for index, key in enumerate(self.flux_names):
                                self.fluxes[key].append(flows_recovery[key])
                        #  recovery of Ucell.
                        Rmem_t, Rccl_t, Racl_t = Rproton(last_solver_variables, self.parameters)
                        Ueq_t = Ueq(last_solver_variables)
                        f_drop_t = fdrop(last_solver_variables, self.operating_inputs, self.parameters)
                        if f_drop_t == 1:
                                self.echem_traj["eta_act"].append(self.variables["eta_c"][j])
                                self.echem_traj["eta_conc"].append(0)
                        else:
                                eta_conc_t = self.variables["eta_c"][j] * (1 - f_drop_t)/f_drop_t
                                eta_act_t = self.variables["eta_c"][j] - eta_conc_t
                                self.echem_traj["eta_act"].append(eta_act_t)
                                self.echem_traj["eta_conc"].append(eta_conc_t)
                        self.echem_traj["i_fc"].append(i_fc)
                        self.echem_traj["fdrop"].append(f_drop_t)
                        self.echem_traj["Ueq"].append(Ueq_t)
                        self.echem_traj["Rmem"].append(Rmem_t)
                        self.echem_traj["Rccl"].append(Rccl_t)
                        self.echem_traj["Racl"].append(Racl_t)
                        self.echem_traj["Ucell"].append(Ucell(t=t, variables=last_solver_variables, operating_inputs=self.operating_inputs, parameters=self.parameters))
                        PRD_t = [last_solver_variables[f'S_N_ccl_{i}'] for i in range(1, self.parameters['n_group_pt'] + 1)] 
                        ECSA_t = getECSA(PRD_t, self.parameters['r_m']) / getECSA(self.parameters['prd0'], self.parameters['r_m'])
                        self.echem_traj["S_N"].append(ECSA_t)
                        self.echem_traj["PRD"].append(PRD_t)


        def _flush(self):
                for key in self.variables.keys():
                        self.variables[key] = []

