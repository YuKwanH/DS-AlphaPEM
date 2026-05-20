from scipy.sparse import csr_matrix
from model.coefficients import *
from model.inst_values import *
from model.state_eq import *
from model.kinetic_eq import *
from modules.nan_tracker import trace_nan


class PEMFC:

        def __init__(self, param, operating_inputs, 
                             variable_names, flux_names):

                self.parameters = dict(param)
                self.variable_names = list(variable_names)
                self.flux_names = list(flux_names)
                self.operating_inputs = operating_inputs

                # GDL nodes name discretization
                discretized_region = ['C_v_agdl', 'C_v_cgdl', 's_agdl', 's_cgdl', 'C_H2_agdl', 'C_O2_cgdl']
                for variable in discretized_region:
                        index = self.variable_names.index(variable)
                        # Delete the previous points
                        self.variable_names.pop(index)
                        # Increase the number of points
                        self.variable_names[index:index] = [f'{variable}_{i}' for i in range(1, self.parameters['n_gdl'] + 1)]
                # MEM nodes name discretization
                for name in ["C_O2_mem", "C_H2_mem", "C_Pt2_mem", "lambda_mem"]:
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
                # This calculation is neccessary for computation efficiency
                self.inst_constant = {"rho_H2O":rho_H2O(self.operating_inputs["Tfc"]),
                                                     "M_Pt0" : 4 / 3 * np.pi * rho_Pt * trapezoid(y=self.parameters['prd0'] * self.parameters['r_m'] ** 3, x=self.parameters['r_m']),
                                                     "ECSA0": getECSA(self.parameters["prd0"], radius= self.parameters['r_m'] )}


                self._idx = {n: i for i, n in enumerate(self.variable_names)}
                self._dif_keys = tuple('d' + n + ' / dt' for n in self.variable_names)
                self._n_states = len(self.variable_names)

                self.t = 0

        def dxdt(self, t, x):

                self.t = t
                dif = dict.fromkeys(self._dif_keys, 0.0)
                states = {n: x[i] for n, i in self._idx.items()}
                inst_states = dif_eq_int_values(t=t, x=states, operating_inputs=self.operating_inputs, parameters=self.parameters)
                all_inst_states = {**self.parameters, **inst_states, **calculate_flows(t, states, self.operating_inputs, self.parameters, **inst_states)}
                all_inst_values = {**all_inst_states, **self.operating_inputs, **self.inst_constant}

                dxdt_AGC(dif, **all_inst_values)
                dxdt_CGC(dif, **all_inst_values)
                dxdt_AGDL(dif, states, **all_inst_values)
                dxdt_CGDL(dif, states, **all_inst_values)
                dxdt_ACL(dif, states, **all_inst_values)
                dxdt_CCL(dif, states, **all_inst_values)
                dxdt_MEM(dif, states, **all_inst_values)
                dxdt_CP(dif, **states, **all_inst_values)
                dxdt_N2(dif, **all_inst_values)
                dxdt_PRD(dif=dif, **states, **all_inst_values)

                return np.fromiter(dif.values(), dtype=float, count=self._n_states)

        def jac_sparsity(self, y0, t=0.0, n_probe=10, rel_step=1e-6, seed=0, extra_states=None):
                """Estimate the sparsity pattern of the Jacobian d(dxdt)/dx.

                The pattern is detected by finite-difference probing: state j is
                perturbed and every dxdt entry that changes is marked as a
                (possibly) non-zero Jacobian entry. To capture branch-dependent
                couplings (saturation-front regimes, lambda >= 1 switches, ...),
                the pattern is the UNION over several probe states: y0, n_probe
                randomly perturbed variants of y0, and any caller-supplied
                extra_states (e.g. snapshots from a previous trajectory).

                A non-finite difference is treated as 'coupled' so the pattern is
                conservatively over-inclusive: missing an entry would corrupt the
                solver's grouped finite-difference Jacobian, while an extra entry
                only costs a little speed.

                Returns a scipy.sparse.csr_matrix of bool, suitable for the
                `jac_sparsity` argument of solve_ivp (method 'BDF' or 'Radau').
                """
                y0 = np.asarray(y0, dtype=float)
                N = self._n_states
                rng = np.random.default_rng(seed)
                S = np.zeros((N, N), dtype=bool)

                probe_states = [y0]
                for _ in range(n_probe):
                        probe_states.append(y0 * (1.0 + 0.03 * rng.standard_normal(N)))
                if extra_states is not None:
                        probe_states.extend(np.asarray(s, dtype=float) for s in extra_states)

                for y in probe_states:
                        f0 = self.dxdt(t, y)
                        if not np.all(np.isfinite(f0)):
                                continue  # unusable probe state, skip it
                        for j in range(N):
                                h = rel_step * max(abs(y[j]), 1e-3)
                                yp = y.copy()
                                yp[j] += h
                                df = self.dxdt(t, yp) - f0
                                # entry changed, or went non-finite -> mark as coupled
                                S[:, j] |= ~(df == 0.0)
                np.fill_diagonal(S, True)  # every state appears in its own equation
                self._jac_sparsity = csr_matrix(S)
                return self._jac_sparsity

        def solve(self, t_span, y0, method='BDF', max_step=None, verbose=True,
                  sparsity=True, **solve_ivp_kwargs):
                # Hand BDF/Radau the Jacobian sparsity pattern so it builds the
                # finite-difference Jacobian with column grouping (a handful of
                # dxdt calls) instead of one column per state (~170 calls).
                if (sparsity and method in ('BDF', 'Radau')
                            and 'jac' not in solve_ivp_kwargs
                            and 'jac_sparsity' not in solve_ivp_kwargs):
                        if getattr(self, '_jac_sparsity', None) is None:
                                self.jac_sparsity(y0)
                        solve_ivp_kwargs['jac_sparsity'] = self._jac_sparsity
                return trace_nan(self, t_span=t_span, y0=y0, method=method,
                                 max_step=max_step, verbose=verbose, **solve_ivp_kwargs)

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
                        states_t = {key: self.variables[key][j] for key in self.variable_names}
                        inst_states_t = dif_eq_int_values(t=t, x = states_t, operating_inputs = self.operating_inputs, parameters = self.parameters)
                        flux_t = calculate_flows(t=t, x = states_t, operating_inputs = self.operating_inputs, parameters = self.parameters, **inst_states_t)
                        for index, key in enumerate(self.flux_names):
                                self.fluxes[key].append(flux_t[key])
                        #  recovery of Ucell.
                        Rmem_t, Rccl_t, Racl_t = Rproton(states_t, self.parameters, self.operating_inputs)
                        Ueq_t = Ueq(states_t, self.operating_inputs)
                        eta_c_t = eta_ccl(states_t, self.operating_inputs, self.parameters)
                        f_drop_t = fdrop(states_t, self.operating_inputs, self.parameters)
                        if f_drop_t == 1:
                                self.echem_traj["eta_act"].append(eta_c_t)
                                self.echem_traj["eta_conc"].append(0)
                        else:
                                eta_conc_t = eta_c_t * (1 - f_drop_t)/f_drop_t
                                eta_act_t = eta_c_t - eta_conc_t
                                self.echem_traj["eta_act"].append(eta_act_t)
                                self.echem_traj["eta_conc"].append(eta_conc_t)  
                        self.echem_traj["i_fc"].append(inst_states_t["i_fc"])
                        self.echem_traj["fdrop"].append(f_drop_t)
                        self.echem_traj["Ueq"].append(Ueq_t)
                        self.echem_traj["Rmem"].append(Rmem_t)
                        self.echem_traj["Rccl"].append(Rccl_t)
                        self.echem_traj["Racl"].append(Racl_t)
                        self.echem_traj["Ucell"].append(Ucell(t=t, variables=states_t, operating_inputs=self.operating_inputs, parameters=self.parameters))
                        PRD_t = [states_t[f'S_N_ccl_{i}'] for i in range(1, self.parameters['n_group_pt'] + 1)] 
                        ECSA_t = getECSA(PRD_t, self.parameters['r_m']) / getECSA(self.parameters['prd0'], self.parameters['r_m'])
                        self.echem_traj["S_N"].append(ECSA_t)
                        self.echem_traj["PRD"].append(PRD_t)

        def _flush(self):
                for key in self.variables.keys():
                        self.variables[key] = []


def event_negative(t, y, operating_inputs, parameters, solver_variable_names, control_variables):
    """This function creates an event that will be checked at each step of solve_ivp integration. The integration stops
    if one of the crucial variables (C_v, lambda, C_O2, C_H2) becomes negative (or smaller than 1e-5).

    Parameters
    ----------
    t : float
        Time (s).
    y : numpy.ndarray
        Numpy list of the solver variables.
    operating_inputs : dict
        Operating inputs of the fuel cell.
    parameters : dict
        Parameters of the fuel cell model.
    solver_variable_names : list
        Names of the solver variables.
    control_variables : dict
        Variables controlled by the user.

    Returns
    -------
    The difference between the minimum value of the crucial variables and 1e-5.
    """

    negative_solver_variables = {} # Dictionary to store the crucial variables
    for index, key in enumerate(solver_variable_names):
        if (key.startswith("C_v_")) or (key.startswith("lambda_")) or \
                (key.startswith("C_O2_")) or (key.startswith("C_H2_")):
            negative_solver_variables[key] = y[index]
    return min(negative_solver_variables.values()) - 1e-5  # 1e-5 is a control parameter to stop the program before
    #                                                        having negative values.