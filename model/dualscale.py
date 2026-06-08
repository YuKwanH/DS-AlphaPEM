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
    """0D PEMFC model following Pukrushpan's lumped-parameter formulation
    (Pukrushpan, Stefanopoulou & Peng 2003, Ch.3 + Ch.5.1), with variable names
    in Raphael Gass's notation.

    Pukrushpan's macro state vector (his eq. 5.1) is:
        x = [m_O2, m_H2, m_N2, w_cp, p_sm, m_sm, m_w,an, m_w,ca, p_rm]^T

    Translated to Raphael's units (mol.m-3 for concentrations, kg.s-1 for flows):

    =====================  =====================  ===========================
    Pukrushpan symbol      This model symbol      Description
    =====================  =====================  ===========================
    m_H2                   C_H2_a                 H2 in the anode chamber
    m_w,an                 C_v_a                  vapour in the anode chamber
    m_O2                   C_O2_c                 O2 in the cathode chamber
    m_N2                   C_N2_c                 N2 in the cathode chamber
    m_w,ca                 C_v_c                  vapour in the cathode chamber
    w_cp                   Wcp                    air-compressor mass flow (kg.s-1)
    p_sm                   Pcsm                   cathode supply manifold pressure
    p_rm                   Pcem                   cathode return manifold pressure
    (algebraic)            lambda_mem             membrane water content
    =====================  =====================  ===========================

    There is no anode manifold (Pukrushpan assumes a pressure-regulated H2 tank
    feeding a dead-end anode with no purge), no manifold humidity state, no
    humidifier injection state (humidifier is a static algebraic model), and no
    back-pressure-valve throttle state (fixed return-manifold orifice area A_T).

    Additional states beyond Pukrushpan -- inherited from the dual-scale model
    for degradation studies, can be ignored for control work:
        Hmem            membrane thickness (chemical degradation)
        C_Pt2_ccl       platinum-ion concentration
        S_N_ccl_i       Pt particle-radius distribution per bin
        theta_ccl_i     Pt-oxide surface coverage per bin
    """

    def __init__(self, parameters, operating_inputs,
                 variable_names=None, flux_names=None):
        self.parameters = dict(parameters)
        self.operating_inputs = operating_inputs

        default_vars = [
            # Cell chambers (lumped GC + GDL + CL; vapour only, no liquid s)
            'C_H2_a', 'C_v_a',
            'C_O2_c', 'C_v_c', 'C_N2_c',
            # Membrane
            'lambda_mem',
            # Cathode air-supply BoP (Pukrushpan: p_sm, p_rm, w_cp)
            'Pcsm', 'Pcem', 'Wcp',
            # Degradation (dual-scale extensions, kept for compatibility)
            'Hmem',
            'C_Pt2_ccl', 'S_N_ccl', 'theta_ccl',
        ]
        self.variable_names = list(default_vars if variable_names is None else variable_names)

        default_fluxes = [
            'W_H2_an_in', 'W_v_an_in', 'W_v_mem',
            'Wcsm_out', 'W_O2_csm_out', 'W_N2_csm_out', 'W_v_csm_out',
            'Wcem_in', 'W_O2_cem_in', 'W_N2_cem_in', 'W_v_cem_in',
            'Wcem_out', 'W_v_cem_out',
            'Wcp_des',
        ]
        self.flux_names = list(default_fluxes if flux_names is None else flux_names)

        # Discretise the microscale PSD variables
        dr = 1e-6 / self.parameters['n_group_pt']
        self.parameters['r_m'] = (np.linspace(1e-8, 1e-6, self.parameters['n_group_pt'] + 1) + dr / 2)[1:]
        self.parameters['prd0'] = initPRD(resolution=self.parameters['n_group_pt'])
        for name in ('S_N_ccl', 'theta_ccl'):
            idx = self.variable_names.index(name)
            self.variable_names.pop(idx)
            self.variable_names[idx:idx] = [f'{name}_{i}' for i in range(1, len(self.parameters['r_m']) + 1)]

        # Storage
        self.variables = {key: [] for key in self.variable_names}
        self.variables['t'] = []
        self.fluxes = {key: [] for key in self.flux_names}
        self.fluxes['t'] = []
        self.echem_traj = {key: [] for key in
                           ['Ucell', 'eta_act', 'eta_conc', 'i_fc', 'fdrop',
                            'Ueq', 'Rmem', 'Rccl', 'Racl', 'S_N', 'PRD']}
        self.echem_traj['t'] = []

        self.inst_constant = {
            'rho_H2O': rho_H2O(self.operating_inputs['Tfc']),
            'M_Pt0': 4 / 3 * np.pi * rho_Pt * trapezoid(
                y=self.parameters['prd0'] * self.parameters['r_m'] ** 3,
                x=self.parameters['r_m']),
            'ECSA0': getECSA(self.parameters['prd0'], radius=self.parameters['r_m']),
        }

        self._idx = {n: i for i, n in enumerate(self.variable_names)}
        self._dif_keys = tuple('d' + n + ' / dt' for n in self.variable_names)
        self._n_states = len(self.variable_names)
        self.t = 0

    @staticmethod
    def Ucell(i_fc, Tfc, P_H2, P_O2, P_c, lambda_mem, Hmem):
        """Pukrushpan's cell terminal voltage (Stefanopoulou & Peng 2003 eq.
        3.18 with the regression coefficients of eq. 3.21).

        ``Vfc = E - vact - vohm - vconc``, where each loss is computed from the
        polynomial regression Pukrushpan fitted on polarization data:

        - Open-circuit (Nernst) voltage E
        - Activation loss ``vact = v0 + va * (1 - exp(-c1*i))``
        - Ohmic loss    ``vohm = i * Rohm``  with Rohm = Hmem / sigma_m
        - Concentration loss ``vconc = i * (c2*i/imax)^c3``

        Inputs are in SI: ``i_fc`` in A.m-2, ``Tfc`` in K, pressures in Pa,
        ``Hmem`` in m. Internally the function rescales to Pukrushpan's
        CGS / bar units (current in A.cm-2, pressures in bar / atm, resistance
        in Ohm.cm2).

        Returns a dict with the voltage breakdown plus the membrane resistance
        in SI (Ohm.m2) so the caller can report HFR consistently with the rest
        of the model. Intended as a private helper of ``PEMFC_0D``; callers
        inside the class should invoke it through ``self.Ucell(...)``.
        """
        # ---- unit conversions ----
        i = max(i_fc * 1e-4, 1e-6)            # A.cm-2, clamp to avoid log(0)
        pH2 = max(P_H2 / 1e5, 1e-9)           # bar
        pO2 = max(P_O2 / 1e5, 1e-9)           # bar
        pca = max(P_c  / 1e5, 1e-9)           # bar
        psat = Psat(Tfc) / 1e5                # bar
        pa_dry_atm = max(pca - psat, 1e-9) / 1.01325  # atm

        # ---- E : Nernst open-circuit (eq. 3.21) ----
        E = (1.229
             - 8.5e-4 * (Tfc - 298.15)
             + 4.308e-5 * Tfc * (np.log(pH2 / 1.01325)
                                 + 0.5 * np.log(pO2 / 1.01325)))

        # ---- vact : activation loss (eq. 3.21) ----
        v0 = (0.279
              - 8.5e-4 * (Tfc - 298.15)
              + 4.308e-5 * Tfc * (np.log(pa_dry_atm)
                                  + 0.5 * np.log(0.1173 * pa_dry_atm)))
        arg = pO2 / 0.1173 + psat        # atm-ish ratio used in c2 / va regressions
        va = ((-1.618e-5 * Tfc - 1.618e-2) * arg ** 2
              + (1.8e-4 * Tfc - 0.166) * arg
              + (-5.8e-4 * Tfc + 0.5736))
        c1 = 10.0
        vact = v0 + va * (1.0 - np.exp(-c1 * i))

        # ---- vohm : ohmic loss (eq. 3.21) ----
        # Springer-style membrane conductivity (b1 fitted to Nafion 117, b2
        # fitted by Pukrushpan to his polarisation data).
        b1 = 0.005139 * lambda_mem - 0.00326
        b2 = 350.0
        sigma_m = max(b1, 1e-9) * np.exp(b2 * (1.0 / 303.0 - 1.0 / Tfc))   # 1/(Ohm.cm)
        tm = Hmem * 100.0                                                  # m -> cm
        Rohm = tm / max(sigma_m, 1e-12)                                    # Ohm.cm2
        vohm = i * Rohm

        # ---- vconc : concentration loss (eq. 3.21) ----
        if arg < 2.0:
            c2 = (7.16e-4 * Tfc - 0.622) * arg + (-1.45e-3 * Tfc + 1.68)
        else:
            c2 = (8.66e-5 * Tfc - 0.068) * arg + (-1.6e-4 * Tfc + 0.54)
        imax = 2.2          # A.cm-2 (Pukrushpan's limit current density)
        c3 = 2.0
        vconc = i * (c2 * i / imax) ** c3

        Vfc = E - vact - vohm - vconc
        return {
            'Ucell':   float(Vfc),
            'E':       float(E),
            'vact':    float(vact),
            'vohm':    float(vohm),
            'vconc':   float(vconc),
            'Rohm_m2': float(Rohm * 1e-4),   # Ohm.cm2 -> Ohm.m2 for HFR storage
        }

    def dxdt(self, t, x):
        self.t = t
        states = {n: x[i] for n, i in self._idx.items()}
        self.parameters['Hmem'] = states['Hmem']
        dif = dict.fromkeys(self._dif_keys, 0.0)

        # ---- Operating inputs and parameters ----
        Tfc = self.operating_inputs['Tfc']
        Pa_des = self.operating_inputs['Pa_des']
        Pc_des = self.operating_inputs['Pc_des']
        Phi_a_des = self.operating_inputs['Phi_a_des']
        Phi_c_des = self.operating_inputs['Phi_c_des']
        Sa = self.operating_inputs['Sa']
        Sc = self.operating_inputs['Sc']
        i_fc = self.operating_inputs['current_density'](t)        # A.m-2
        I_load = i_fc * self.parameters['Aact']                   # A
        Imin_aux = self.operating_inputs.get('Imin_aux', 0.0)

        Aact = self.parameters['Aact']
        Hcl = self.parameters['Hcl']
        Hgdl = self.parameters['Hgdl']
        Hgc = self.parameters['Hgc']
        Hmem = states['Hmem']
        epsilon_cl = self.parameters['epsilon_cl']
        epsilon_gdl = self.parameters['epsilon_gdl']
        epsilon_mc = self.parameters['epsilon_mc']

        Psat_fc = Psat(Tfc)
        Psat_ext = Psat(Text)

        # Lumped gas-phase volume (m3): GC + porous GDL + porous CL
        Vol_a = Aact * (Hgc + epsilon_gdl * Hgdl + epsilon_cl * Hcl)
        Vol_c = Aact * (Hgc + epsilon_gdl * Hgdl + epsilon_cl * Hcl)

        # ---- Cell chamber state ----
        C_H2_a, C_v_a = states['C_H2_a'], states['C_v_a']
        C_O2_c, C_v_c, C_N2_c = states['C_O2_c'], states['C_v_c'], states['C_N2_c']
        lambda_mem = states['lambda_mem']
        C_Pt2_ccl = states['C_Pt2_ccl']

        Pa = (C_H2_a + C_v_a) * R * Tfc
        Pc = (C_O2_c + C_v_c + C_N2_c) * R * Tfc
        Phi_c = C_v_c * R * Tfc / max(Psat_fc, 1.0)
        y_c = C_O2_c / max(C_O2_c + C_N2_c, 1e-30)
        x_v_c = Phi_c * Psat_fc / max(Pc, 1.0)

        # ---- Cathode manifolds (Pukrushpan eqs. 3.4 / 5.1) ----
        Pcsm = states['Pcsm']
        Pcem = states['Pcem']
        # Supply manifold composition: humidified air at Phi_c_des
        x_v_csm = Phi_c_des * Psat_fc / max(Pcsm, 1.0)
        M_dry_air = yO2_ext * M_O2 + (1.0 - yO2_ext) * M_N2
        Mcsm = x_v_csm * M_H2O + (1.0 - x_v_csm) * M_dry_air
        # Return manifold composition: same as cell cathode
        Mcem = x_v_c * M_H2O + y_c * (1.0 - x_v_c) * M_O2 + (1.0 - y_c) * (1.0 - x_v_c) * M_N2

        # Ambient air molar mass
        x_v_ext = Phi_ext * Psat_ext / Pext
        Mext = x_v_ext * M_H2O + yO2_ext * (1.0 - x_v_ext) * M_O2 + (1.0 - yO2_ext) * (1.0 - x_v_ext) * M_N2

        # ---- Voltage and electrochemistry (Pukrushpan eq. 3.18 / 3.21) ----
        P_H2_a = C_H2_a * R * Tfc
        P_O2_c = C_O2_c * R * Tfc
        v_breakdown = self.Ucell(
            i_fc=i_fc, Tfc=Tfc,
            P_H2=P_H2_a, P_O2=P_O2_c, P_c=Pc,
            lambda_mem=lambda_mem, Hmem=Hmem,
        )
        Ucell_val = v_breakdown['Ucell']

        # ---- Membrane water transport (Springer-style, lumped) ----
        lambda_eq_a = lambda_eq(C_v_a, 0.0, Tfc, Kshape)
        lambda_eq_c = lambda_eq(C_v_c, 0.0, Tfc, Kshape)
        S_sorp_a = (gamma_sorp(C_v_a, 0.0, lambda_mem, Tfc, Hcl, Kshape)
                    * rho_mem / M_eq * (lambda_eq_a - lambda_mem))
        S_sorp_c = (gamma_sorp(C_v_c, 0.0, lambda_mem, Tfc, Hcl, Kshape)
                    * rho_mem / M_eq * (lambda_eq_c - lambda_mem))
        Dw_val = Dw(lambda_mem, Tfc)
        N_v_mem = ((2.5 / 22.0) * i_fc / F * lambda_mem
                   - rho_mem / M_eq * Dw_val * (lambda_eq_c - lambda_eq_a) / max(Hmem, 1e-9))
        W_v_mem = N_v_mem * M_H2O * Aact

        # ---- Compressor setpoint ----
        Wcp_des = (n_cell * Mext * Pext / max(Pext - Phi_ext * Psat_ext, 1.0)
                   * (1.0 / yO2_ext) * Sc * max(I_load, Imin_aux) / (4.0 * F))

        # ============= ANODE -- dead-end, stoich-based H2 supply (Pukrushpan eq. 3.51) =============
        W_H2_an_in = M_H2 * Sa * max(I_load, Imin_aux) / (2.0 * F)
        omega_an_in = (Phi_a_des * Psat_fc / max(Pa_des - Phi_a_des * Psat_fc, 1.0)
                       * (M_H2O / M_H2))
        W_v_an_in = omega_an_in * W_H2_an_in
        S_H2_react = I_load / (2.0 * F)

        # ============= CATHODE -- SM + RM with fixed throttle (Pukrushpan eqs. 3.25-3.27) =============
        # Static humidifier injection
        Wv_hum_in = M_H2O * (Phi_ext * Psat_ext / Pext) * (states['Wcp'] / max(Mext, 1e-9))
        Wc_v_des = M_H2O * Phi_c_des * Psat_fc / max(Pext, 1.0) * (states['Wcp'] / max(Mext, 1e-9))
        Wc_inj = max(Wc_v_des - Wv_hum_in, 0.0)
        Wcsm_in = states['Wcp'] + Wc_inj
        # SM -> cell
        Wcsm_out = Ksm_out * max(Pcsm - Pc, 0.0)
        omega_csm = x_v_csm * M_H2O / max((1.0 - x_v_csm) * M_dry_air, 1e-30)
        W_air_csm_out = Wcsm_out / (1.0 + omega_csm)
        W_v_csm_out = Wcsm_out - W_air_csm_out
        W_O2_csm_out = W_air_csm_out * (yO2_ext * M_O2) / M_dry_air
        W_N2_csm_out = W_air_csm_out - W_O2_csm_out
        # Cell -> RM
        Wcem_in = Kem_in * max(Pc - Pcem, 0.0)
        M_dry_c = y_c * M_O2 + (1.0 - y_c) * M_N2
        omega_c_cell = (Phi_c * Psat_fc / max(Pc - Phi_c * Psat_fc, 1.0)
                        * (M_H2O / max(M_dry_c, 1e-30)))
        W_air_cem_in = Wcem_in / (1.0 + omega_c_cell)
        W_v_cem_in = Wcem_in - W_air_cem_in
        W_O2_cem_in = W_air_cem_in * (y_c * M_O2) / max(M_dry_c, 1e-30)
        W_N2_cem_in = W_air_cem_in - W_O2_cem_in
        # RM -> atmosphere via linear orifice (Pukrushpan eq. 3.47 form).
        # The conductance K_em_eff is self-tuned to match Pc_des at the demanded
        # compressor flow -- keeps the open-loop, static-throttle character of
        # Pukrushpan's model while letting the user pick a Pc_des setpoint
        # without introducing a back-pressure-valve state.
        Kem_out_eff = max(Wcp_des, 1e-12) / max(Pc_des - Pext, 100.0)
        Wcem_out = Kem_out_eff * max(Pcem - Pext, 0.0)

        # ============= STATE DERIVATIVES =============
        # Anode cell chamber
        dif['dC_H2_a / dt'] = (W_H2_an_in / M_H2 - S_H2_react) / Vol_a
        dif['dC_v_a / dt'] = (W_v_an_in / M_H2O - W_v_mem / M_H2O) / Vol_a - S_sorp_a

        # Cathode cell chamber
        dif['dC_O2_c / dt'] = ((W_O2_csm_out - W_O2_cem_in) / M_O2 - I_load / (4.0 * F)) / Vol_c
        dif['dC_N2_c / dt'] = ((W_N2_csm_out - W_N2_cem_in) / M_N2) / Vol_c
        dif['dC_v_c / dt'] = ((W_v_csm_out - W_v_cem_in + W_v_mem) / M_H2O / Vol_c
                              + I_load / (2.0 * F * Vol_c) - S_sorp_c)

        # Membrane water content
        dif['dlambda_mem / dt'] = (M_eq / (rho_mem * epsilon_mc)) * (S_sorp_a + S_sorp_c)

        # Membrane thickness degradation
        dif['dHmem / dt'] = -20.8 / (0.82 * 1980e3) * flourideReleaseRate(
            MT=Hmem, U=Ucell_val, Tmem=Tfc, PO2_ca=P_O2_c, Hmem_init=1.2e-5)

        # Manifold pressures
        dif['dPcsm / dt'] = R * Tfc / (Vsm * Mcsm) * (Wcsm_in - n_cell * Wcsm_out)
        dif['dPcem / dt'] = R * Tfc / (Vem * Mcem) * (n_cell * Wcem_in - Wcem_out)

        # Compressor first-order dynamics
        if self.parameters.get('aux_system', True):
            dif['dWcp / dt'] = (Wcp_des - states['Wcp']) / tau_cp

        # ---- Pt particle dynamics ----
        n_pt = self.parameters['n_group_pt']
        r_m = self.parameters['r_m']
        prd = np.fromiter((states[f'S_N_ccl_{i + 1}'] for i in range(n_pt)),
                          dtype=float, count=n_pt)
        theta_ccl = np.fromiter((states[f'theta_ccl_{i + 1}'] for i in range(n_pt)),
                                dtype=float, count=n_pt)
        C_Proton = Cproton_CCL(lambda_w=lambda_mem)
        kdis = PtDissolution(Ucell_val, Tfc, C_Pt2_ccl, theta_ccl)
        kox = PtOxidation(Ucell_val, Tfc, C_Proton, theta_ccl)
        kcdis = PtOxideDissolution(theta_ccl, C_Proton)
        kdet = PtDetachment(Ucell_val, Tfc, r_m)
        drdt = (Vm_Pt * krdp * C_Pt2_ccl * np.exp(R0 / r_m)
                - Vm_Pt * (kdis + kox) * Cpt2_ref * np.exp(R0 / r_m))
        dMdisdt = 4.0 * np.pi * rho_Pt * trapezoid(y=prd * r_m ** 2 * drdt, x=r_m)
        dMcdisdt = 4.0 * np.pi * rho_Pt * trapezoid(y=prd * r_m ** 2 * kcdis, x=r_m)
        dfdt = -np.gradient(prd * drdt, r_m) - kdet * prd
        dthetadt = ((kox - kcdis) / GAMMA_max) - (2.0 * theta_ccl / r_m) * drdt
        dif['dC_Pt2_ccl / dt'] = -3.33 / M_Pt * (dMdisdt - dMcdisdt) / self.inst_constant['M_Pt0']
        for i in range(n_pt):
            dif[f'dS_N_ccl_{i + 1} / dt'] = dfdt[i]
            dif[f'dtheta_ccl_{i + 1} / dt'] = dthetadt[i]

        return np.fromiter(dif.values(), dtype=float, count=self._n_states)

    def jac_sparsity(self, y0, t=0.0, n_probe=10, rel_step=1e-6, seed=0, extra_states=None):
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
                continue
            for j in range(N):
                h = rel_step * max(abs(y[j]), 1e-3)
                yp = y.copy()
                yp[j] += h
                df = self.dxdt(t, yp) - f0
                S[:, j] |= ~(df == 0.0)
        np.fill_diagonal(S, True)
        self._jac_sparsity = csr_matrix(S)
        return self._jac_sparsity

    def solve(self, t_span, y0, method='BDF', max_step=None, verbose=True,
              sparsity=True, **solve_ivp_kwargs):
        if (sparsity and method in ('BDF', 'Radau')
                and 'jac' not in solve_ivp_kwargs
                and 'jac_sparsity' not in solve_ivp_kwargs):
            if getattr(self, '_jac_sparsity', None) is None:
                self.jac_sparsity(y0)
            solve_ivp_kwargs['jac_sparsity'] = self._jac_sparsity
        return trace_nan(self, t_span=t_span, y0=y0, method=method,
                         max_step=max_step, verbose=verbose, **solve_ivp_kwargs)

    def _recovery(self, sol):
        self.variables['t'].extend(list(sol.t))
        self.echem_traj['t'].extend(list(sol.t))
        for index, key in enumerate(self.variable_names):
            self.variables[key].extend(list(sol.y[index]))
        Tfc = self.operating_inputs['Tfc']
        for j in range(len(sol.t)):
            t = sol.t[j]
            states_t = {key: self.variables[key][j] for key in self.variable_names}
            lambda_mem_t = states_t['lambda_mem']
            Hmem_t = states_t['Hmem']
            # Pukrushpan voltage breakdown (eq. 3.18 / 3.21)
            P_H2_a_t = states_t['C_H2_a'] * R * Tfc
            P_O2_c_t = states_t['C_O2_c'] * R * Tfc
            P_c_t = (states_t['C_O2_c'] + states_t['C_v_c'] + states_t['C_N2_c']) * R * Tfc
            i_fc_t = self.operating_inputs['current_density'](t)
            v_breakdown = self.Ucell(
                i_fc=i_fc_t, Tfc=Tfc,
                P_H2=P_H2_a_t, P_O2=P_O2_c_t, P_c=P_c_t,
                lambda_mem=lambda_mem_t, Hmem=Hmem_t,
            )
            self.echem_traj['eta_act'].append(v_breakdown['vact'])
            self.echem_traj['eta_conc'].append(v_breakdown['vconc'])
            self.echem_traj['i_fc'].append(i_fc_t)
            # Pukrushpan has no flooding factor; report fdrop = 1 for backward
            # compatibility with the 1D notebooks' plotting helpers.
            self.echem_traj['fdrop'].append(1.0)
            self.echem_traj['Ueq'].append(v_breakdown['E'])
            # Map Pukrushpan's lumped Rohm onto the (Rmem, Rccl, Racl) triple so
            # the HFR notebooks keep working. Rohm sits in Rmem, the catalyst-
            # layer protonic resistances are zero in this formulation.
            self.echem_traj['Rmem'].append(v_breakdown['Rohm_m2'])
            self.echem_traj['Rccl'].append(0.0)
            self.echem_traj['Racl'].append(0.0)
            self.echem_traj['Ucell'].append(v_breakdown['Ucell'])
            PRD_t = [states_t[f'S_N_ccl_{i + 1}'] for i in range(self.parameters['n_group_pt'])]
            ECSA_t = getECSA(PRD_t, self.parameters['r_m']) / self.inst_constant['ECSA0']
            self.echem_traj['S_N'].append(ECSA_t)
            self.echem_traj['PRD'].append(PRD_t)

    def _flush(self):
        for key in self.variables.keys():
            self.variables[key] = []
        for key in self.fluxes.keys():
            self.fluxes[key] = []
        for key in self.echem_traj.keys():
            self.echem_traj[key] = []

    @staticmethod
    def default_initial_state(parameters, operating_inputs):
        """Default initial state for the Pukrushpan-style 0D model."""
        Tfc = operating_inputs['Tfc']
        Pa_des = operating_inputs['Pa_des']
        Pc_des = operating_inputs['Pc_des']
        Phi_a_des = operating_inputs['Phi_a_des']
        Phi_c_des = operating_inputs['Phi_c_des']
        Sc = operating_inputs.get('Sc', 2.0)
        Imin_aux = operating_inputs.get('Imin_aux', 0.0)
        Hmem0 = parameters['Hmem']
        Aact = parameters['Aact']
        n_pt = parameters['n_group_pt']

        Psat_fc = Psat(Tfc)
        P_a_dry = max(Pa_des - Phi_a_des * Psat_fc, 0.0)
        P_c_dry = max(Pc_des - Phi_c_des * Psat_fc, 0.0)
        C_H2_a0 = P_a_dry / (R * Tfc)
        C_v_a0 = Phi_a_des * Psat_fc / (R * Tfc)
        C_O2_c0 = yO2_ext * P_c_dry / (R * Tfc)
        C_N2_c0 = (1.0 - yO2_ext) * P_c_dry / (R * Tfc)
        C_v_c0 = Phi_c_des * Psat_fc / (R * Tfc)
        lambda_mem0 = lambda_eq(C_v_c0, 0.0, Tfc, Kshape)
        prd0 = initPRD(resolution=n_pt).tolist()
        theta0 = [0.0] * n_pt

        # Seed compressor at the steady-state setpoint implied by current_density(0)
        cd = operating_inputs.get('current_density')
        i_fc0 = float(cd(0.0)) if callable(cd) else 0.0
        I_load0 = max(i_fc0 * Aact, Imin_aux)
        Psat_ext = Psat(Text)
        x_v_ext = Phi_ext * Psat_ext / Pext
        Mext = (x_v_ext * M_H2O
                + yO2_ext * (1.0 - x_v_ext) * M_O2
                + (1.0 - yO2_ext) * (1.0 - x_v_ext) * M_N2)
        Wcp0 = (n_cell * Mext * Pext / max(Pext - Phi_ext * Psat_ext, 1.0)
                * (1.0 / yO2_ext) * Sc * I_load0 / (4.0 * F))

        # Pcsm slightly above Pc_des, Pcem slightly below, so the cell starts with
        # a non-zero pressure gradient driving Wcsm_out and Wcem_in -- this breaks
        # the singular zero-flow initial condition that otherwise makes BDF's
        # Jacobian degenerate at low currents.
        Pcsm0 = Pc_des * 1.005
        Pcem0 = Pc_des * 0.995

        # Order MUST match self.variable_names exactly:
        #   [C_H2_a, C_v_a, C_O2_c, C_v_c, C_N2_c,
        #    lambda_mem, Pcsm, Pcem, Wcp,
        #    Hmem, C_Pt2_ccl, S_N_ccl_*, theta_ccl_*]
        return ([C_H2_a0, C_v_a0,
                 C_O2_c0, C_v_c0, C_N2_c0,
                 lambda_mem0,
                 Pcsm0, Pcem0, Wcp0,
                 Hmem0,
                 0.0] + prd0 + theta0)
