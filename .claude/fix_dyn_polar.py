"""Patch simulation/Test_polar/all_cond 50A dyn.ipynb so its initial run
finishes in minutes (not hours) and never crashes on a NaN."""
import json

PATH = r"D:\MFC2024\simulation\Test_polar\all_cond 50A dyn.ipynb"

# --- Replacement for cell 2 (the simulation loop) ---
NEW_CELL2 = '''# ---- Validation run: small grid, robust per-point error handling ----
# The dynamic model (181 states, full BoP) is ~5 s wall per simulated second
# at default mesh, so the full 9-condition x 7-current x 600-s grid would
# take ~5 hours. For an INITIAL validation run we use a reduced subset
# (1 condition, 3 currents, t_span = 30 s) so we can confirm the model
# produces sensible polarisation values in ~5 minutes. After this baseline
# is confirmed, expand the grid for a production run.
import time as _time

pola_tests_sim = {}
load_points    = [10, 20, 30, 35, 40, 45, 50]   # full sweep for reference plots
# Validation subset -- comment the next 3 lines out for the full grid
_VALIDATION_RHC = [0.5]
_VALIDATION_P   = [1.4]               # bar (was [1.3, 1.4, 1.5])
_VALIDATION_T   = [333.15]            # K   (was [323.15, 333.15, 343.15])
_VALIDATION_I   = [10, 30, 50]        # A   (was load_points)

T_SPAN = (0, 30)                      # was (0, 600); 30 s is enough for steady-state Ucell
MAX_STEP = 1e-1

t0_total = _time.perf_counter()
for RHC in _VALIDATION_RHC:
    for P_des in _VALIDATION_P:
        for T_des in _VALIDATION_T:
            cond_key = f"RHC{RHC}_P{P_des}_T{T_des}"
            print(f"\\n=== {cond_key} ===")
            states_test = []           # list of dicts (one per successful current)
            i_kept      = []
            for I_LOAD in _VALIDATION_I:
                op = dict(operating_inputs)
                op["Phi_c_des"] = RHC
                op["Pa_des"]    = P_des * 1e5
                op["Pc_des"]    = P_des * 1e5
                op["Tfc"]       = T_des
                op["current_density"] = (lambda x, _I=I_LOAD: _I / parameters["Aact"])

                t0 = _time.perf_counter()
                try:
                    x_init = init_x_for("dynamic", op, parameters)
                    if RHC == 0:
                        x_init[model_dummy_var_names.index("Wc_inj") if False else 0] = 0.0
                    # Use the same initial state both for model construction and the solver
                    model = PEMFC_dyn(parameters, op, x_init)
                    # Override Wc_inj setpoint by name (works for non-zero RHC)
                    if RHC > 0:
                        idx_wc = model.solver_variable_names.index("Wc_inj")
                        x_init[idx_wc] = 4.0e-5
                    sol = solve_ivp(model.dxdt, T_SPAN, x_init,
                                    method="BDF", max_step=MAX_STEP)
                    if not sol.success:
                        print(f"  I = {I_LOAD:5.1f} A  ->  solver failed: {sol.message}")
                        continue
                    model._recovery(sol)
                    # Gather last-step values from both variables and ec_kinetics
                    states = {}
                    for var_name in model.variables:
                        v = model.variables[var_name]
                        states[var_name] = v[-1] if hasattr(v, "__len__") and len(v) else float("nan")
                    for var_name in model.ec_kinetics:
                        v = model.ec_kinetics[var_name]
                        states[var_name] = v[-1] if hasattr(v, "__len__") and len(v) else float("nan")
                    # Reject if Ucell is non-finite or unphysical
                    u = states.get("Ucell", float("nan"))
                    try:
                        u = float(u)
                    except Exception:
                        u = float("nan")
                    if not (np.isfinite(u) and 0.0 < u < 1.3):
                        print(f"  I = {I_LOAD:5.1f} A  ->  unphysical Ucell = {u}")
                        continue
                    states_test.append(states)
                    i_kept.append(I_LOAD)
                    print(f"  I = {I_LOAD:5.1f} A  ->  Ucell = {u:.3f} V"
                          f"   ({_time.perf_counter() - t0:.1f} s wall)")
                except Exception as exc:
                    print(f"  I = {I_LOAD:5.1f} A  ->  {type(exc).__name__}: {exc}")

            if not states_test:
                print(f"  -> {cond_key}: no successful points, skipping")
                continue

            # Pack into the same shape the downstream cells expect:
            # {cond_key: {"states": {var_name: [val_per_i]}}}
            states_profile = {var: [s.get(var, float("nan")) for s in states_test]
                              for var in states_test[0].keys()}
            states_profile["_i_kept"] = i_kept
            pola_tests_sim[cond_key] = {"states": states_profile, "i_kept": i_kept}

print(f"\\nDone. {len(pola_tests_sim)} condition(s) with "
      f"successful points. Total wall time: {_time.perf_counter() - t0_total:.1f} s")
'''


with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace cell 2
nb["cells"][2]["source"]          = NEW_CELL2.splitlines(keepends=True)
nb["cells"][2]["outputs"]         = []
nb["cells"][2]["execution_count"] = None
if "id" not in nb["cells"][2]:
    nb["cells"][2]["id"] = f"cell-{abs(hash(NEW_CELL2)) % 10**8:08d}"

# Also patch every downstream plot cell to use value.get("i_kept", load_points)
# so they plot against the actual current points that succeeded (gives correct
# x-axis when the simulator skipped a current that hit NaN).
for ci in range(3, len(nb["cells"])):
    cell = nb["cells"][ci]
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "load_points" not in src or "pola_tests_sim" not in src:
        continue
    # Replace ", load_points," with ", value.get(\"i_kept\", load_points),"
    # only when the value variable is in scope (heuristic: cell loops over pola_tests_sim.items())
    if "for cond_key, value in pola_tests_sim.items()" in src or "for key, value in pola_tests_sim.items()" in src:
        new_src = src.replace(", load_points,", ", value.get(\"i_kept\", load_points),")
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            print(f"  patched plot in cell {ci}")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("Notebook updated.")
