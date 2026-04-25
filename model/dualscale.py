import pandas as pd
import scipy

from config.initialize import *
from model.coefficients import *
from model.inst_values import *
from modules.state_eq import *

class PEMFC:

        def __init__(self, parameters):
                self.parameters = parameters

        def dxdt(self, t, x, u):
                
                # Create state gradients dictionary
                dif = {('d' + key + ' / dt'): 0 for key in solver_variable_names}
                # Mapping macro-scale variables
                x = {}
                for key in solver_variable_names:
                        x[key] = x[solver_variable_names.index(key)]
                massflow = calculate_flows(self, t, x, u, self.parameters)
                inst_states = dif_eq_int_values(t=t, x = x, control_variables= u,operating_inputs= u, parameters=self.parameters)
                dxdt_AGC(dif, **inst_states, **self.parameters, **massflow)
                dxdt_CGC(dif,  **inst_states, **self.parameters, **massflow)
                dxdt_AGDL(dif, **inst_states, **self.parameters, **massflow)
                dxdt_CGDL(dif, **inst_states, **self.parameters, **massflow)
                dxdt_ACL(dif, **inst_states, **self.parameters, **massflow)
                dxdt_CCL(dif, **inst_states, **self.parameters, **massflow)
                dxdt_MEM(dif, **inst_states, **self.parameters, **massflow)
                dxdt_CP(dif, **inst_states, **self.parameters, **massflow)
                dxdt_Manifold(dif, **inst_states, **self.parameters, **massflow)
                dxdt_TH(dif, **inst_states, **self.parameters, **massflow)
                dxdt_U(dif, **inst_states, **self.parameters, **massflow)
                dxdt_N2(dif, **inst_states, **self.parameters, **massflow)
                dxdt_PRD(dif, **inst_states, **self.parameters, **massflow)

                return list(dif.values())
                
                
