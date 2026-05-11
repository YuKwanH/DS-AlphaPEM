"""Patch both polar.ipynb files (dynamic + static model) so they run four
calibration scenarios (single condition / fixed pressure / fixed temperature /
all data) and compare the results side-by-side."""
import json
import uuid
from pathlib import Path


def make_md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


def make_code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def set_source(cell, src):
    cell["source"] = src.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


# ----------------------------------------------------------------------
# Cell texts shared between both notebooks
# ----------------------------------------------------------------------

SCENARIO_MD = (
    "## Calibration scenarios\n"
    "Compare the calibration result obtained when fitting the model against "
    "different subsets of the experimental polarization data. Edit "
    "`SINGLE_T`, `SINGLE_P`, `SINGLE_RHC`, `FIXED_P`, `FIXED_T` below if you "
    "want different anchor conditions.\n"
)

SCENARIO_DEFS = r'''# Anchor conditions for the partial scenarios -- edit as needed.
SINGLE_T   = 60       # degrees C
SINGLE_P   = 1.4e5    # Pa
SINGLE_RHC = 0        # %
FIXED_P    = 1.4e5    # Pa  -- scenario "fixed pressure"
FIXED_T    = 60       # degrees C -- scenario "fixed temperature"

scenarios = {
    "one_condition":      {
        "label":  f"One condition (T={SINGLE_T}, P={SINGLE_P/1e5:.1f} bar, RHC={SINGLE_RHC})",
        "filter": lambda T, P, RHC: (T == SINGLE_T) and (P == SINGLE_P) and (RHC == SINGLE_RHC),
    },
    "fixed_pressure":     {
        "label":  f"Fixed pressure (P={FIXED_P/1e5:.1f} bar, all T, all RHC)",
        "filter": lambda T, P, RHC: P == FIXED_P,
    },
    "fixed_temperature":  {
        "label":  f"Fixed temperature (T={FIXED_T}, all P, all RHC)",
        "filter": lambda T, P, RHC: T == FIXED_T,
    },
    "all_data":           {
        "label":  "All conditions",
        "filter": lambda T, P, RHC: True,
    },
}


def conditions_in_scenario(filter_fn):
    """Return the list of cond_keys actually present in polardata_exp that pass the filter."""
    keys = []
    for RHC in RHC_tested:
        for P_des in PAC_tested:
            for T_des in TFC_tested:
                cond_key = "T" + str(T_des) + "_P" + str(int(P_des/1e2 - 1e3)) + "_HRC" + str(RHC)
                if cond_key in polardata_exp and filter_fn(T_des, P_des, RHC):
                    keys.append(cond_key)
    return keys


for name, info in scenarios.items():
    matches = conditions_in_scenario(info["filter"])
    print(f"  {name:20s}  {len(matches):2d} conditions  -> {matches}")
'''


# ----------------------------------------------------------------------
# Cell texts that are model-specific
# ----------------------------------------------------------------------

# --- DYNAMIC model: polartest_sim signature uses model_calib ---
DYN_POLARTEST = r'''def polartest_sim(model_calib, cond_filter=None):
    """Run the dynamic model across every (RHC, P, T) condition for which we
    have experimental data and return ``{cond_key: [Ucell at I_tested]}``.
    Returns ``False`` if any cell voltage came back NaN.

    Pass ``cond_filter(T, P, RHC) -> bool`` to restrict which conditions are
    simulated (used by the calibration-scenario loop).
    """
    OCV  = model_calib.parameters["OCV"]
    Aact = model_calib.parameters["Aact"]
    op   = model_calib.operating_inputs

    result = {}
    for RHC in RHC_tested:
        for P_des in PAC_tested:
            for T_des in TFC_tested:
                cond_key = "T" + str(T_des) + "_P" + str(int(P_des/1e2 - 1e3)) + "_HRC" + str(RHC)
                if cond_key not in polardata_exp:
                    continue
                if cond_filter is not None and not cond_filter(T_des, P_des, RHC):
                    continue
                op["Phi_c_des"] = RHC / 100
                op["Pa_des"] = P_des
                op["Pc_des"] = P_des
                op["Tfc"]    = T_des + 273.15

                Ucell_test = []
                for I_LOAD in I_tested:
                    i_density = I_LOAD / Aact
                    op["current_density"] = lambda t, _i=i_density: _i

                    x_init = init_x(op, model_calib.parameters)
                    sol = solve_ivp(model_calib.dxdt, (0, 60), x_init,
                                    method="BDF", max_step=1e-1)
                    last = {k: sol.y[idx, -1] for idx, k in enumerate(model_calib.solver_variable_names)}

                    Rmem_t, Rccl_t, Racl_t = Rproton(last, model_calib.parameters)
                    Rp = sum(Rmem_t) + Rccl_t + Racl_t
                    Ueq_t = Ueq(last)
                    Ucell_test.append(Ueq_t - OCV
                                      - i_density * (Rp + model_calib.parameters["Re"])
                                      - last["eta_c"])
                    if any(math.isnan(v) for v in Ucell_test):
                        return False

                result[cond_key] = Ucell_test
    return result
'''

DYN_OBJECTIVE = r'''from copy import deepcopy

def _experimental_ucell(cond_key):
    """Experimental cell voltage at every I_tested for one condition."""
    df = polardata_exp[cond_key]
    i_exp = df["I_LOAD"].to_numpy(dtype=float)
    v_exp = df["VFC"].to_numpy(dtype=float) / n_cell
    idx = [np.argmin((i_test - i_exp) ** 2) for i_test in I_tested]
    return v_exp[idx]


def make_objective(cond_filter):
    """Return an optuna objective that fits only the conditions selected
    by ``cond_filter`` -- so a single objective function can be reused
    across the four calibration scenarios."""
    def objective(trial):
        params_trial = deepcopy(parameters)
        op_trial     = deepcopy(operating_inputs)

        params_trial["OCV"]          = trial.suggest_float("OCV",          0.2,   0.4,  log=True)
        params_trial["i0_c_ref"]     = trial.suggest_float("i0_c_ref",     1e-2,  10,   log=True)
        params_trial["kappa_c"]      = trial.suggest_float("kappa_c",      1,     10,   log=True)
        params_trial["tau"]          = trial.suggest_float("tau",          1,     4,    log=True)
        params_trial["Re"]           = trial.suggest_float("Re",           1e-7,  1e-5, log=True)
        params_trial["epsilon_mc"]   = trial.suggest_float("epsilon_mc",   0.15,  0.4,  log=True)
        params_trial["epsilon_gdl"]  = trial.suggest_float("epsilon_gdl",  0.5,   0.7,  log=True)
        params_trial["epsilon_c"]    = trial.suggest_float("epsilon_c",    0.1,   0.3,  log=True)
        params_trial["epsilon_cl"]   = trial.suggest_float("epsilon_cl",   0.1,   0.4,  log=True)
        params_trial["a_slim"]       = trial.suggest_float("a_slim",       1e-2,  0.5,  log=True)
        params_trial["b_slim"]       = trial.suggest_float("b_slim",       1e-2,  0.5,  log=True)
        params_trial["a_switch"]     = trial.suggest_float("a_switch",     1e-2,  0.5,  log=True)
        params_trial["Hcl"]          = trial.suggest_float("Hcl",          1e-5,  2e-5, log=True)
        params_trial["Hgdl"]         = trial.suggest_float("Hgdl",         2e-5,  5e-5, log=True)

        model_trial = PEMFC_dyn(params_trial, op_trial,
                                init_x(op_trial, params_trial))
        try:
            polardata_sim = polartest_sim(model_trial, cond_filter=cond_filter)
        except Exception:
            return 100.0
        if polardata_sim is False or not polardata_sim:
            return 100.0

        error_sum = 0.0
        for cond_key, ucell_sim in polardata_sim.items():
            ucell_exp = _experimental_ucell(cond_key)
            error_sum += float(np.sum((np.array(ucell_sim) - ucell_exp) ** 2))
        return error_sum

    return objective
'''


# --- STATIC model: polartest_sim signature uses parameters dict ---
STAT_POLARTEST = r'''def polartest_sim(parameters, cond_filter=None):
    """Run the static (algebraic) model across every (RHC, P, T) condition
    for which we have experimental data and return
    ``{cond_key: [Ucell at I_tested]}``. Returns ``False`` if any cell
    voltage came back NaN.

    Pass ``cond_filter(T, P, RHC) -> bool`` to restrict which conditions are
    simulated (used by the calibration-scenario loop).
    """
    result = {}
    for RHC in RHC_tested:
        for P_des in PAC_tested:
            for T_des in TFC_tested:
                cond_key = "T" + str(T_des) + "_P" + str(int(P_des/1e2 - 1e3)) + "_HRC" + str(RHC)
                if cond_key not in polardata_exp:
                    continue
                if cond_filter is not None and not cond_filter(T_des, P_des, RHC):
                    continue

                Ucell_test = []
                for I_LOAD in I_tested:
                    Wout_c = 20 + I_LOAD * parameters["K_wout"]
                    Win_c  = 3  + I_LOAD ** parameters["K_win"]
                    operating_inputs_stat = {
                        "Tfc": T_des + 273.15, "Phi_a_des": 0.0, "Phi_c_des": RHC / 100,
                        "Pa_des": P_des, "Pc_des": P_des,
                        "Win_c": Win_c, "Wout_c": Wout_c, "Win_a": 4.8, "Wout_a": 4.8,
                    }
                    model = PEMFC_stat(parameters, operating_inputs_stat)
                    sol = model.solve(I_LOAD / parameters["Aact"])
                    Ucell_test.append(model.cell_voltage(
                        I_LOAD / model.parameters["Aact"], sol, parameters, operating_inputs))
                    if any(math.isnan(v) for v in Ucell_test):
                        return False
                result[cond_key] = Ucell_test
    return result
'''

STAT_OBJECTIVE = r'''def _experimental_ucell(cond_key):
    """Experimental cell voltage at every I_tested for one condition."""
    df = polardata_exp[cond_key]
    i_exp = df["I_LOAD"].to_numpy(dtype=float)
    v_exp = df["VFC"].to_numpy(dtype=float) / n_cell
    idx = [np.argmin((i_test - i_exp) ** 2) for i_test in I_tested]
    return v_exp[idx]


def make_objective(cond_filter):
    """Return an optuna objective that fits only the conditions selected
    by ``cond_filter`` -- so a single objective function can be reused
    across the four calibration scenarios."""
    def objective(trial):
        params_trial = deepcopy(parameters)

        params_trial["OCV"]          = trial.suggest_float("OCV",          0.9,   1.0,  log=True)
        params_trial["i0_c_ref"]     = trial.suggest_float("i0_c_ref",     1e-2,  10,   log=True)
        params_trial["kappa_c"]      = trial.suggest_float("kappa_c",      1,     10,   log=True)
        params_trial["tau"]          = trial.suggest_float("tau",          1,     4,    log=True)
        params_trial["Re"]           = trial.suggest_float("Re",           1e-7,  1e-5, log=True)
        params_trial["epsilon_mc"]   = trial.suggest_float("epsilon_mc",   0.15,  0.4,  log=True)
        params_trial["epsilon_gdl"]  = trial.suggest_float("epsilon_gdl",  0.5,   0.7,  log=True)
        params_trial["epsilon_c"]    = trial.suggest_float("epsilon_c",    0.1,   0.3,  log=True)
        params_trial["epsilon_cl"]   = trial.suggest_float("epsilon_cl",   0.1,   0.4,  log=True)
        params_trial["a_slim"]       = trial.suggest_float("a_slim",       1e-2,  0.5,  log=True)
        params_trial["b_slim"]       = trial.suggest_float("b_slim",       1e-2,  0.5,  log=True)
        params_trial["a_switch"]     = trial.suggest_float("a_switch",     1e-2,  0.5,  log=True)
        params_trial["Hcl"]          = trial.suggest_float("Hcl",          1e-5,  2e-5, log=True)
        params_trial["Hgdl"]         = trial.suggest_float("Hgdl",         2e-4,  5e-4, log=True)
        params_trial["K_wout"]       = trial.suggest_float("K_wout",       0.1,   10,   log=True)
        params_trial["K_win"]        = trial.suggest_float("K_win",        0.1,   1,    log=True)

        try:
            polardata_sim = polartest_sim(params_trial, cond_filter=cond_filter)
        except Exception:
            return 100.0
        if polardata_sim is False or not polardata_sim:
            return 100.0

        error_sum = 0.0
        for cond_key, ucell_sim in polardata_sim.items():
            ucell_exp = _experimental_ucell(cond_key)
            error_sum += float(np.sum((np.array(ucell_sim) - ucell_exp) ** 2))
        return error_sum

    return objective
'''

# ----------------------------------------------------------------------
# Run-the-scenarios cells (model-specific, slightly different setup)
# ----------------------------------------------------------------------

RUN_DYN = r'''# Run a separate optuna study for each scenario and store the best params.
# Each trial does up to ~30 short BDF integrations, so adjust timeout / n_trials
# to the wall-time you can afford. The total budget is roughly
# (timeout per scenario) * 4.
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PER_SCENARIO_TIMEOUT = 600    # seconds
PER_SCENARIO_TRIALS  = 2000
N_JOBS               = 6

results = {}
for name, info in scenarios.items():
    print(f"\n=== Calibrating scenario: {name} ===")
    print(f"    {info['label']}")
    print(f"    conditions used: {conditions_in_scenario(info['filter'])}")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(info["filter"]),
        n_trials=PER_SCENARIO_TRIALS,
        timeout=PER_SCENARIO_TIMEOUT,
        n_jobs=N_JOBS,
        show_progress_bar=False,
    )

    # Re-run polartest_sim with the best params over ALL conditions so we
    # can later check how well each scenario generalises.
    params_best = deepcopy(parameters)
    op_best     = deepcopy(operating_inputs)
    params_best.update(study.best_params)
    model_best  = PEMFC_dyn(params_best, op_best, init_x(op_best, params_best))
    sim_all     = polartest_sim(model_best, cond_filter=None)

    results[name] = {
        "label":           info["label"],
        "calibrated_keys": conditions_in_scenario(info["filter"]),
        "best_params":     study.best_params,
        "best_value":      study.best_value,
        "sim_all":         sim_all if sim_all is not False else {},
    }
    print(f"    best objective on the calibrated subset: {study.best_value:.4f}")
'''

RUN_STAT = r'''# Run a separate optuna study for each scenario. The static model is
# algebraic so each trial is fast -- a short timeout already covers many
# trials. Total wall time is roughly (timeout per scenario) * 4.
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PER_SCENARIO_TIMEOUT = 180    # seconds
PER_SCENARIO_TRIALS  = 2000
N_JOBS               = 6

results = {}
for name, info in scenarios.items():
    print(f"\n=== Calibrating scenario: {name} ===")
    print(f"    {info['label']}")
    print(f"    conditions used: {conditions_in_scenario(info['filter'])}")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(info["filter"]),
        n_trials=PER_SCENARIO_TRIALS,
        timeout=PER_SCENARIO_TIMEOUT,
        n_jobs=N_JOBS,
        show_progress_bar=False,
    )

    # Re-run polartest_sim with the best params over ALL conditions so we
    # can later check how well each scenario generalises.
    params_best = deepcopy(parameters)
    params_best.update(study.best_params)
    sim_all     = polartest_sim(params_best, cond_filter=None)

    results[name] = {
        "label":           info["label"],
        "calibrated_keys": conditions_in_scenario(info["filter"]),
        "best_params":     study.best_params,
        "best_value":      study.best_value,
        "sim_all":         sim_all if sim_all is not False else {},
    }
    print(f"    best objective on the calibrated subset: {study.best_value:.4f}")
'''

# ----------------------------------------------------------------------
# Comparison plotting cells (shared)
# ----------------------------------------------------------------------

COMPARE_FIT_MD = (
    "## Compare best-fit polarization curves across scenarios\n"
    "Each panel below shows, for one calibration scenario, the experimental "
    "polarization curves (grey dots) overlaid with the simulated curves "
    "produced by that scenario's best parameters (coloured lines using the "
    "project's `plot_condition` convention). Conditions used in calibration "
    "are highlighted in bold.\n"
)

COMPARE_FIT_CELL = r'''# 2x2 grid -- one panel per scenario, all conditions on every panel.
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
for ax, (name, info) in zip(axes.flatten(), results.items()):
    cal_set = set(info["calibrated_keys"])

    # Background: every experimental polar curve, in light grey.
    for cond_key in polardata_exp.keys():
        ucell_exp = _experimental_ucell(cond_key)
        ax.plot(I_tested, ucell_exp, "o-", color="0.75",
                linewidth=0.9, markersize=3, alpha=0.7)

    # Simulated curves with the project colour / linestyle / marker mapping.
    for cond_key, ucell_sim in info["sim_all"].items():
        lw = 2.2 if cond_key in cal_set else 1.0
        plot_condition(ax, I_tested, ucell_sim, cond_key,
                       linewidth=lw, markersize=4)

    ax.set_title(f"{name}\n{info['label']}\nobjective on calibrated subset = "
                 f"{info['best_value']:.4f}", fontsize=9)
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("U_cell (V)")
    ax.grid(True, alpha=0.3)

# Custom legend explaining what bold / thin mean.
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="0.4", marker="o", linewidth=2.2, label="simulated, USED for calibration"),
    Line2D([0], [0], color="0.4", marker="o", linewidth=1.0, label="simulated, held-out condition"),
    Line2D([0], [0], color="0.75", marker="o", linewidth=0.9, label="experiment (all conditions)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.tight_layout(rect=(0, 0.03, 1, 1))
plt.show()
'''

COMPARE_HEATMAP_MD = (
    "## Residual heatmap\n"
    "Each row is a calibration scenario, each column is one operating "
    "condition; the cell value is the sum of squared errors over the "
    "tested current points. Bright (low residual) on a column the scenario "
    "did NOT use during calibration tells you the scenario generalised; "
    "dark (large residual) tells you it did not.\n"
)

COMPARE_HEATMAP_CELL = r'''# Residual heatmap: scenarios (rows) x conditions (columns).
all_cond_keys = list(polardata_exp.keys())
n_scen = len(results)
residual_matrix = np.full((n_scen, len(all_cond_keys)), np.nan)
for i_scen, (name, info) in enumerate(results.items()):
    for i_cond, cond_key in enumerate(all_cond_keys):
        if cond_key not in info["sim_all"]:
            continue
        ucell_sim = np.array(info["sim_all"][cond_key])
        ucell_exp = _experimental_ucell(cond_key)
        residual_matrix[i_scen, i_cond] = float(np.sum((ucell_sim - ucell_exp) ** 2))

fig, ax = plt.subplots(figsize=(max(8, len(all_cond_keys) * 0.5), 0.5 + 0.5 * n_scen))
im = ax.imshow(residual_matrix, aspect="auto", cmap="cividis_r")
ax.set_yticks(range(n_scen)); ax.set_yticklabels(list(results.keys()), fontsize=9)
ax.set_xticks(range(len(all_cond_keys)))
ax.set_xticklabels(all_cond_keys, rotation=45, ha="right", fontsize=8)
# Mark the cells that were USED for calibration with a black border.
for i_scen, (name, info) in enumerate(results.items()):
    cal_set = set(info["calibrated_keys"])
    for i_cond, cond_key in enumerate(all_cond_keys):
        if cond_key in cal_set:
            ax.add_patch(plt.Rectangle((i_cond - 0.5, i_scen - 0.5), 1, 1,
                                       fill=False, edgecolor="white", linewidth=2))
ax.set_title("Sum-of-squared-errors per condition (white box = used during calibration)")
fig.colorbar(im, ax=ax, label=r"$\Sigma(\Delta U)^2$")
plt.tight_layout()
plt.show()
'''

COMPARE_PARAMS_MD = (
    "## Best parameters side-by-side\n"
    "How much do the calibrated parameters change as you broaden the "
    "training set? Big swings here indicate the parameter is poorly "
    "constrained by the partial dataset.\n"
)

COMPARE_PARAMS_CELL = r'''# Side-by-side parameter comparison.
param_df = pd.DataFrame({name: info["best_params"] for name, info in results.items()})
print(param_df.to_string(float_format=lambda v: f"{v:.4g}"))
'''


# ----------------------------------------------------------------------
# Patch logic
# ----------------------------------------------------------------------

def patch_dynamic(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Replace polartest_sim cell (id fb007ff6).
    set_source(next(c for c in nb["cells"] if c.get("id") == "fb007ff6"), DYN_POLARTEST)
    # Replace objective cell (id 640febd2).
    set_source(next(c for c in nb["cells"] if c.get("id") == "640febd2"), DYN_OBJECTIVE)

    # Delete the obsolete "sanity check" / "visual comparison" / single-study /
    # single-post-optim cells (they referenced `objective`, `study`, `sim`).
    OBSOLETE_IDS = {"7b08231b", "1d18c27c", "1f0e6d0a", "107d4033", "4e28fc4d"}
    nb["cells"] = [c for c in nb["cells"] if c.get("id") not in OBSOLETE_IDS]

    # Drop any trailing empty code cells before appending the new section.
    while nb["cells"] and nb["cells"][-1].get("cell_type") == "code" \
            and not "".join(nb["cells"][-1].get("source", [])).strip():
        nb["cells"].pop()

    nb["cells"].extend([
        make_md(SCENARIO_MD),
        make_code(SCENARIO_DEFS),
        make_code(RUN_DYN),
        make_md(COMPARE_PARAMS_MD),
        make_code(COMPARE_PARAMS_CELL),
        make_md(COMPARE_FIT_MD),
        make_code(COMPARE_FIT_CELL),
        make_md(COMPARE_HEATMAP_MD),
        make_code(COMPARE_HEATMAP_CELL),
    ])
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(nb["cells"])


def patch_static(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Replace polartest_sim cell (id 9639c0bb).
    set_source(next(c for c in nb["cells"] if c.get("id") == "9639c0bb"), STAT_POLARTEST)
    # Replace objective cell (id f234b5c1).
    set_source(next(c for c in nb["cells"] if c.get("id") == "f234b5c1"), STAT_OBJECTIVE)

    # Delete the obsolete single-study cell (id 78ab7258) and the single
    # post-optim plot (id a28b26ce).
    OBSOLETE_IDS = {"78ab7258", "a28b26ce"}
    nb["cells"] = [c for c in nb["cells"] if c.get("id") not in OBSOLETE_IDS]

    while nb["cells"] and nb["cells"][-1].get("cell_type") == "code" \
            and not "".join(nb["cells"][-1].get("source", [])).strip():
        nb["cells"].pop()

    nb["cells"].extend([
        make_md(SCENARIO_MD),
        make_code(SCENARIO_DEFS),
        make_code(RUN_STAT),
        make_md(COMPARE_PARAMS_MD),
        make_code(COMPARE_PARAMS_CELL),
        make_md(COMPARE_FIT_MD),
        make_code(COMPARE_FIT_CELL),
        make_md(COMPARE_HEATMAP_MD),
        make_code(COMPARE_HEATMAP_CELL),
    ])
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(nb["cells"])


dyn_path  = Path("simulation/parameter calibration/dynamic model/polar.ipynb")
stat_path = Path("simulation/parameter calibration/static model/polar.ipynb")

n_dyn = patch_dynamic(dyn_path)
print(f"dyn  polar.ipynb: total cells = {n_dyn}")

n_stat = patch_static(stat_path)
print(f"stat polar.ipynb: total cells = {n_stat}")

import nbformat
nbformat.validate(nbformat.read(str(dyn_path),  as_version=4))
nbformat.validate(nbformat.read(str(stat_path), as_version=4))
print("Both notebooks valid.")
