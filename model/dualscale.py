from model.coefficients import *
from model.inst_values import *
from model.state_eq import *
from model.kinetic_eq import *

class PEMFC:

        def __init__(self, param, operating_inputs, 
                             variable_names, flux_names):

                # Defensive copies: __init__ mutates `variable_names` (pop/slice insert
                # for discretization) and writes new keys ('r_m', 'prd0') into `parameters`.
                # Without copies, those mutations leak back to the caller's imported
                # `solver_variable_names` / `parameters`, breaking any subsequent
                # PEMFC(...) call ("'C_v_agdl' is not in list").
                self.parameters = dict(param)
                self.variable_names = list(variable_names)
                self.flux_names = list(flux_names)
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

                # Cached lookups for the dxdt hot path: O(1) state mapping and pre-built
                # output keys, so each ODE eval avoids O(N^2) list.index loops and
                # per-call string concatenation. Built after all discretization is done
                # so variable_names is in its final form.
                self._idx = {n: i for i, n in enumerate(self.variable_names)}
                self._dif_keys = tuple('d' + n + ' / dt' for n in self.variable_names)
                self._n_states = len(self.variable_names)

        def dxdt(self, t, x):

                # State gradients dict (pre-cached keys; values default to 0.0)
                dif = dict.fromkeys(self._dif_keys, 0.0)

                # O(N) state mapping via cached index map
                states = {n: x[i] for n, i in self._idx.items()}

                inst_states = dif_eq_int_values(t=t, x=states, operating_inputs=self.operating_inputs, parameters=self.parameters)
                massflow = calculate_flows(t, states, self.operating_inputs, self.parameters, **inst_states)

                # Merge once per dxdt call so each dxdt_X site unpacks one dict.
                common = {**inst_states, **self.parameters, **massflow}
                common_op = {**common, **self.operating_inputs}

                dxdt_AGC(dif, **common)
                dxdt_CGC(dif, **common)
                dxdt_AGDL(dif, states, **common)
                dxdt_CGDL(dif, states, **common)
                dxdt_ACL(dif, states, **common)
                dxdt_CCL(dif, states, **common)
                dxdt_MEM(dif, **common)
                dxdt_CP(dif, **states, **common)
                dxdt_Manifold(dif, **common_op)
                dxdt_TH(dif, **states, **common_op)
                dxdt_U(dif, **states, **common)
                dxdt_N2(dif, **common)
                dxdt_PRD(dif=dif, **states, **common)

                return np.fromiter(dif.values(), dtype=float, count=self._n_states)

        def compute_jac_sparsity(self, x0, t=0.0, eps=1e-7, n_probes=3, seed=0):
                """Numerically detect the Jacobian sparsity pattern at x0.

                Probes the initial state plus a few small random perturbations and
                marks any (i, j) where dxdt[i] changes when x[j] is perturbed. The
                diagonal is forced on. Conservative: a False entry only forms if
                NO probe detected a change.
                """
                from scipy.sparse import lil_matrix, csr_matrix

                n = len(x0)
                sparsity = lil_matrix((n, n), dtype=np.int8)
                rng = np.random.default_rng(seed)
                x_base = np.asarray(x0, dtype=float)

                probes = [x_base.copy()]
                for _ in range(max(0, n_probes - 1)):
                        delta = rng.uniform(-1e-3, 1e-3, size=n) * np.abs(x_base) + 1e-9
                        probes.append(x_base + delta)

                for x_probe in probes:
                        f0 = np.asarray(self.dxdt(t, x_probe), dtype=float)
                        for j in range(n):
                                x_p = x_probe.copy()
                                step = max(eps * abs(x_p[j]), eps)
                                x_p[j] += step
                                f1 = np.asarray(self.dxdt(t, x_p), dtype=float)
                                changed = np.flatnonzero(f1 != f0)
                                for i in changed:
                                        sparsity[i, j] = 1

                for i in range(n):
                        sparsity[i, i] = 1

                return csr_matrix(sparsity)
                
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

