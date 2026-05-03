import matplotlib.pyplot as plt
from config.initialize import *
from config.settings import *
from model.dualscale import PEMFC
from scipy.integrate import solve_ivp
from modules.display import display


if __name__ == "__main__":

    model = PEMFC(param=parameters, operating_inputs=operating_inputs,
                               variable_names=solver_variable_names, flux_names=solver_flux_names)
    solution_init = init_x(operating_inputs, parameters)
    