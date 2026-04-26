import matplotlib.pyplot as plt
from config.initialize import *
from config.settings import *
from model.dualscale import PEMFC
from scipy.integrate import solve_ivp
from modules.display import display

if __name__ == "__main__":
    model = PEMFC(param=parameters, variable_names=solver_variable_names, operating_inputs=operating_inputs)
    solution_init = init_x(operating_inputs, parameters)
    solution = solve_ivp(fun=model.dxdt, t_span=(0, 10), y0=solution_init, method='BDF')
    print("Simulation done")
    display(solution, model)
    print("Display done")
    
