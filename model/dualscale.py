import pandas as pd
import scipy

from config.initialize import Tfc
from model.coefficients import *
from model.states import *
from modules.state_eq import *

class PEMFC:

        def __init__(self, parameters):
                self.parameters = parameters

        def dxdt(self, t, x, u):
                
                # Create state gradients dictionary
                dif = {('d' + key + ' / dt'): 0 for key in solver_variable_names}
                # Mapping macro-scale variables
                x = {}

                inst_states = dif_eq_int_values(t=t, x = x, iload = iload, control_variables= u,
                                                                         operating_inputs= u,
                                                                         parameters=self.parameters)
                
                
