import numpy as np

# Stack parameters
parameters = {'Aact': 30e-4, 'Hmem': 2.5e-5, 'Hgc': 8e-4, 'Wgc': 4e-4, 'Lgc': 1.287,
                        'Hcl': 1e-5, "Hgdl": 2e-4} #m
# Physical constants
F = 96485.3329 # Faraday's constant (C/mol)
M_H2O = 18.01528e-3  # kg/mol
# Material properties
epsilon_gdl = 0.6
epsilon_cl = 0.3
epsilon_c = 0.399
theta_c = 120 * np.pi / 180  
rho_mem = 1980  # kg.m-3. It is the density of the dry membrane.
rho_cl = 3.87e2 # kg.m-3. It is the density of the catalyst layer.
rho_gdl = 2e3  # kg.m-3. It is the density of the gas diffusion layer.
M_eq = 2.1  # kg.mol-1. It is the equivalent molar mass of ionomer.