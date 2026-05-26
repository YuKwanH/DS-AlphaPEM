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
                self.parameters["Hmem"] = states["delta_mem"]
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
                if self.parameters.get("aux_system", True):
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


class PEMFC_0D:

    def __init__(self, parameters, operating_inputs):
        """
        Dual-scale mathematical model of a fuel cell
        Walid, AMU
        Jay Tawee Pukrushpan, Michigan University

        Default unit:
        Pressure: Pa
        surface: cm2
        volume: cm^3
        temperature: K
        weight: kg
        time: second

        :param P_nom: the rated power of the fuel cell
        """
        self.parameters = parameters
        self.operating_inputs = operating_inputs
        self.states = {"Ucell": 0.8}
        self.fluxes = {"Qia_H2": 0, "Qia_vp": 0, "Qic_O2": 0, "Qic_N2": 0, "Qic_vp": 0, "Qr_H2": 0, "Qr_O2": 0, "Qr_H2O": 0, "Qoa_H2": 0, "Qoa_vp": 0, "Qoc_O2": 0, "Qoc_N2": 0, "Qoc_vp": 0}
        self.variable_names = ["MH2_an", "MH2O_an", "MO2_ca", "MN2_ca", "MH2O_ca", "Prm", "c_pt2_ccl", "Hmem", "S_N_ccl", "theta_ccl"]
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
        self.record = {"Poa": [], "Poc": []}

    def dxdt(self, t, state):

        # Map the state vector to the corresponding variables
        for index, key in enumerate(self.variable_names):
            self.states[key] = state[index]

        MH2_an, MH2O_an = self.states["MH2_an"], self.states["MH2O_an"]
        MO2_ca, MN2_ca, MH2O_ca = self.states["MO2_ca"], self.states["MN2_ca"], self.states["MH2O_ca"]
        Prm = self.states["Prm"]
        c_pt2_ccl = self.states["c_pt2_ccl"]
        Hmem = self.states["Hmem"]
        PRD = [self.states[f'S_N_ccl_{i}'] for i in range(1, self.parameters['n_group_pt'] + 1)]
        theta_ccl = [self.states[f'theta_ccl_{i}'] for i in range(1, self.parameters['n_group_pt'] + 1)]
        Tdes, Pa_des, Pc_des = self.operating_inputs["Tfc"], self.operating_inputs["Pa_des"], self.operating_inputs["Pc_des"]
        Phi_a_des, Phi_c_des = self.operating_inputs["Phi_a_des"], self.operating_inputs["Phi_c_des"]
        Ucell = self.states["Ucell"]
        iload = self.operating_inputs["current_density"](t) / 1e4  # A/cm2
        Psat_ = Psat(Tdes)
        Vol_an = self.parameters["Aact"] * (self.parameters["Hcl"] + self.parameters["Hgdl"]) * 1e-6  # cm3
        Vol_ca = self.parameters["Aact"] * (self.parameters["Hcl"] + self.parameters["Hgdl"]) * 1e-6  # cm3


        # Instantaneous values calculation
        # ------------- Anode inlet conditions ------------- #
        Pa_in = Pa_des
        P_im = Pa_des
        pca = (0.0009225603028 * P_im + 2.84870103138421) * 1e3  # 141232.746
        Ja_in = (pca - Pa_des) * 10e-6  # kg/s
        Pin_vp_a = Phi_a_des * Psat_  # Pressure of the inlet vapor mass flow 3.60
        Pin_H2 = Pa_in - Pin_vp_a  # Pressure of the inlet vapor mass flow 3.61
        humidityRatio = (Pin_vp_a / Pin_H2) * (M_H2O / M_H2)
        J_H2_in =Ja_in / (1 + humidityRatio)
        J_vp_in_a = Ja_in - J_H2_in
        Pa_H2_in = self.states["MH2_an"] * R_H2 * Tdes / Vol_an
        # ------------- Cathode inlet conditions ------------- #
        Jc_in = 0.0289 # kg/s fixed air supply
        Pc_in = Pc_des
        Phi_c_in = self.operating_inputs["Phi_c_des"]
        Pin_vp_c = Phi_c_des * Psat_  
        Pair = Pc_des - Pin_vp_c
        Pc_O2 = self.states["MO2_ca"] * R_O2 * Tdes / Vol_ca
        Kh = (Pin_vp_c / Pair) * (M_H2O / (yO2_ext * M_O2 + (1 - yO2_ext) * M_N2))# Humidity ratio
        KM_O2 = yO2_ext * M_O2 / (yO2_ext * M_O2 + (1 - yO2_ext) * M_N2)
        J_air_in = Jc_in / (Kh + 1)
        J_O2_in = J_air_in * KM_O2
        J_N2_in = J_air_in * (1 - KM_O2)
        J_vp_in_c = Jc_in - J_air_in  # Vapor
        # ------------- Source terms ------------- #
        S_H2 = -iload / (2 * F) * M_H2 # mol/s
        S_O2 = -iload / (4 * F) * M_O2 # mol/s
        S_H2O = -iload / (2 * F) * M_H2O  # mol/s
                # ------------- Anode outlet conditions ------------- #
        Pa_H2_out = self.states["MH2_an"] * R_H2 * Tdes / Vol_an
        Kexaust_H2 = J_H2_in / S_H2
        MH2O_sat_a = Psat_ * Vol_an / (R_H2 * Tdes)