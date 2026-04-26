import matplotlib.pyplot as plt
from config.initialize import *
from config.settings import *
from model.dualscale import PEMFC
from scipy.integrate import solve_ivp
from modules.display import display

if __name__ == "__main__":
    model = PEMFC(param=parameters, variable_names=solver_variable_names, operating_inputs=operating_inputs)
    solution_init = init_x(operating_inputs, parameters)

    # For ~7x extra speedup with a slight (within-tolerance) trajectory shift,
    # uncomment the next two lines and pass jac_sparsity=jac_sparsity to solve_ivp.
    # jac_sparsity = model.compute_jac_sparsity(solution_init)
    solution = solve_ivp(fun=model.dxdt, t_span=(0, 10), y0=solution_init, method='BDF')
    print("Simulation done")
    display(solution, model)
    print("Display done")
    
