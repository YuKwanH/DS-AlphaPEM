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
    jac_sparsity = model.compute_jac_sparsity(solution_init)
    solution = solve_ivp(fun=model.dxdt, t_span=(0, 100), y0=solution_init,
                                      method='BDF', jac_sparsity=jac_sparsity)
    print("Simulation done")
    display(solution, model)
    print("Display done")
    
