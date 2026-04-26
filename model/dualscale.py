from model.coefficients import *
from model.inst_values import *
from model.state_eq import *
from model.kinetic_eq import *

class PEMFC:

        def __init__(self, param, variable_names, operating_inputs):

                self.parameters = param
                self.variable_names = variable_names
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

                dr = 1e-6 / self.parameters['n_group_pt']
                self.parameters['r_m'] = (np.linspace(1e-8, 1e-6, self.parameters['n_group_pt'] + 1) + dr / 2)[1:]
                self.parameters['prd0'] = initPRD(resolution=self.parameters['n_group_pt'])

                index_prd = self.variable_names.index('S_N_ccl')
                self.variable_names.pop(index_prd)
                self.variable_names[index_prd:index_prd] = [f'S_N_ccl_{i}' for i in range(1, len(self.parameters['r_m']) + 1)]

                index_theta_ccl = self.variable_names.index('theta_ccl')
                self.variable_names.pop(index_theta_ccl)
                self.variable_names[index_theta_ccl:index_theta_ccl] = [f'theta_ccl_{i}' for i in range(1, len(self.parameters['r_m']) + 1)]
                

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
                
                
