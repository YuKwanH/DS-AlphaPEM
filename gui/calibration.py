"""Calibration page — three-column layout.

  * **Left  (data viewer)** — pick a dataset (Polarization / HFR / EIS) and a
    set of conditions, render the corresponding measurement curves.
  * **Middle (optimizer)** — calibration target, model variant, Optuna
    settings, parameter-bounds table, and a big Start button.
  * **Right (results)**   — placeholder for the convergence curve, best-fit
    parameter table, and overlay-vs-experiment plot.

The optimizer backend is intentionally a stub for now — the panel records
the requested run into ``state["calib_request"]`` so the actual Optuna
loop can be wired in once the UI feels right. Everything else (data
loading, plotting, parameter-bounds editor) works end-to-end.
"""
import ast
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data.export import export_experiment_data
from gui import style as _style
from gui import calib_backend


def _is_optuna_available():
    """Re-probe on every rerun so a mid-session `pip install optuna`
    is picked up the next time the page is refreshed (we cannot rely on
    a module-level constant — Python only imports the module once per
    process, so the flag would stay False until a full Streamlit restart).
    """
    import importlib.util
    return importlib.util.find_spec("optuna") is not None


# ----------------------------------------------------------------------------
# Dataset catalogue
# ----------------------------------------------------------------------------
DATASETS = ("Polarization", "HFR", "EIS")

# Physical parameters worth varying in a calibration. Two flavours are
# supported:
#
#   CONTINUOUS — (label, low, high, log, unit)
#   DISCRETE   — (label, None, None, False, unit, [c1, c2, ...])
#
# Discrete params (a non-empty ``choices`` 6th element) sample over a
# fixed set of values; the bounds editor swaps the low/high pair for a
# multiselect. Defaults and bounds match the recipe used in the
# project's calibration notebooks
# (`simulation/parameter calibration/.../*.ipynb`).
CALIB_PARAMS = {
    # name          (display label,                  low,    high,   log,   unit,  [choices])
    "OCV":          ("Open-circuit voltage",         0.90,   1.00,   False, "V"),
    "i0_c_ref":     ("Cathode exchange i0",          0.01,   15.0,   True,  "A/m²"),
    "kappa_c":      ("Cathode Tafel slope",          0.5,    3.0,    False, "-"),
    "epsilon_mc":   ("Membrane / CL contact",        0.15,   0.5,    False, "-"),
    "epsilon_c":    ("Cathode ionomer fraction",     0.05,   0.35,   False, "-"),
    "epsilon_cl":   ("Catalyst-layer porosity",      0.1,    0.5,    False, "-"),
    "Hcl":          ("Catalyst-layer thickness",     1e-5,   3e-5,   True,  "m"),
    "Hgdl":         ("GDL thickness",                2e-4,   4e-4,   True,  "m"),
    "e":            ("Bruggeman exponent (discrete)", None,  None,   False, "-", [3, 4, 5]),
    # Additional supported parameters. All entries are selected by default.
    "tau":          ("Tortuosity exponent",          1.0,    4.0,    False, "-"),
    "Re":           ("Electronic resistance",        5e-7,   1e-5,   True,  "Ω·m²"),
    "epsilon_gdl":  ("GDL porosity",                 0.5,    0.8,    False, "-"),
    "a_slim":       ("Saturation a_slim",            5e-3,   0.3,    True,  "-"),
    "b_slim":       ("Saturation b_slim",            0.1,    0.9,    False, "-"),
    "a_switch":     ("Switch coverage a",            0.05,   0.5,    False, "-"),
}

# Calibrate every supported parameter unless the user removes one.
DEFAULT_PARAMS = list(CALIB_PARAMS)


def _is_discrete(name):
    """True if the param sweeps a fixed list of values instead of a (low, high) range."""
    spec = CALIB_PARAMS.get(name)
    return spec is not None and len(spec) >= 6 and spec[5] is not None


def _default_bounds(name):
    """Initial bounds value for a parameter — list of choices for discrete,
    ``(low, high)`` tuple for continuous."""
    spec = CALIB_PARAMS[name]
    if _is_discrete(name):
        return list(spec[5])
    return (spec[1], spec[2])

OPTIMIZERS = ("TPE (Optuna)", "Genetic algorithm", "Grid search")
AUX_CHOICES = ("With auxiliary (BoP)", "Without auxiliary")
# ODE integrator choices for transient calibration. Mirrors the set the
# Simulation page exposes so the user calibrates with the same solver
# they will later simulate with. The backend does NOT fall back to a
# different solver if the chosen one hits a NaN — a failed trial stays
# failed (per the strict-fail policy in ``calib_backend.make_objective``).
PDE_SOLVERS = ("BDF", "Radau", "LSODA", "RK45")
# TPE is the recommended default: sample-efficient for 2-6 continuous
# parameters with log/linear scales, handles the kind of budget (~50
# trials) that PEMFC calibration runs typically use, and it's the engine
# already wired into the calibration scripts in `.claude/patch_hfr.py`.
OPTIMIZER_DEFAULT = "TPE (Optuna)"

# All three variants are wired end-to-end in gui/calib_backend.py.
# Static is fastest (~50 ms / point, algebraic); Dual-scale and Dynamic
# run a transient settle at every measurement current and are honest about
# the cost in the optimizer panel (caption + per-trial estimate).
MODEL_VARIANTS = ("Static", "Dual-scale", "Dynamic")

# Default calibration case. ``Dynamic`` with auxiliary disabled is routed to
# the same PEMFC backend, but displaying that combination is misleading; use
# the backend's honest model label directly.
DEFAULT_MODEL_VARIANT = "Dual-scale"
DEFAULT_AUX_SYSTEM = False
DEFAULT_CONDITIONS = (
    "T50_P300_HRC0",
    "T50_P300_HRC50",
    "T50_P500_HRC50",
)


# ----------------------------------------------------------------------------
# Cached loaders (Excel reads are slow; cache them per session).
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(data_type):
    """Cache-wrapped ``export_experiment_data`` so repeat picks are fast."""
    return export_experiment_data(data_type)


def _find_col(df, prefix, *, exclude=None):
    """Return the first column whose name starts with ``prefix`` (case-
    sensitive, accent-tolerant) and does not contain ``exclude``.

    The EIS sheet ships column names like ``Partie réelle`` and
    ``Partie réelle stack`` with a French é. Some toolchains decode the
    bytes differently — matching by prefix avoids hard-coding the accent.
    """
    for col in df.columns:
        s = str(col)
        if s.startswith(prefix) and (exclude is None or exclude not in s):
            return col
    return None


def _parse_hfr_R(cell):
    """HFR.xlsx stores R as a stringified ``(R_real, I_load)`` tuple."""
    if isinstance(cell, (int, float)):
        return float(cell)
    try:
        val = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return float("nan")
    if isinstance(val, tuple) and len(val) >= 1:
        return float(val[0])
    return float(val)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def render(state, *, section_height=820):
    """Render the full three-column calibration page into ``state``."""
    state.setdefault("calib", {
        "dataset":      "Polarization",
        "conditions":   list(DEFAULT_CONDITIONS),
        "target":       "Polarization",
        "model":        DEFAULT_MODEL_VARIANT,
        "aux_system":   DEFAULT_AUX_SYSTEM,
        "method":       "BDF",          # PDE solver — locked-in during a run
        "n_trials":     50,
        "optimizer":    OPTIMIZER_DEFAULT,
        "seed":         42,
        "params":       list(DEFAULT_PARAMS),
        "bounds":       {k: _default_bounds(k) for k in CALIB_PARAMS},
        "dwell_pola":   DWELL_POLA_S,
        "dwell_hfr":    DWELL_HFR_S,
    })
    # Migrate any stale state from earlier sessions so a returning user
    # never crashes on `MODEL_VARIANTS.index(...)`.
    if state["calib"].get("model") not in MODEL_VARIANTS:
        state["calib"]["model"] = DEFAULT_MODEL_VARIANT

    col_data, col_opt, col_res = st.columns([1.15, 0.85, 1.45], gap="medium")
    with col_data:
        with st.container(height=section_height, border=True):
            _render_data_viewer(state)
    with col_opt:
        with st.container(height=section_height, border=True):
            _render_optimizer_panel(state)
    with col_res:
        with st.container(height=section_height, border=True):
            _render_results_panel(state)


# ----------------------------------------------------------------------------
# Column 1 — Experiment data viewer
# ----------------------------------------------------------------------------
def _render_data_viewer(state):
    st.markdown("#### § Experiment data")

    cfg = state["calib"]
    cfg["dataset"] = st.selectbox(
        "Dataset",
        options=DATASETS,
        index=DATASETS.index(cfg.get("dataset", "Polarization")),
        key="calib_dataset",
        help="Pick which measurement file to display.",
    )

    try:
        data = _load(_dataset_key(cfg["dataset"]))
    except Exception as exc:
        st.error(f"Could not load {cfg['dataset']} data: {exc}")
        return
    if not data:
        st.info("No data found.")
        return

    cond_keys = sorted(data.keys())
    # EIS has a lot of conditions; default to a single pick to keep the
    # plot readable. The other two are small enough to multi-select.
    default = cfg.get("conditions") or cond_keys[:1]
    default = [c for c in default if c in cond_keys] or cond_keys[:1]
    cfg["conditions"] = st.multiselect(
        "Conditions",
        options=cond_keys,
        default=default,
        key="calib_conditions",
        help="Select one or more operating points to overlay.",
    )

    if cfg["conditions"]:
        fig = _build_data_figure(cfg["dataset"], data, cfg["conditions"])
        st.pyplot(fig, clear_figure=True)
    else:
        st.caption("Select at least one condition to plot.")
        return

    # ---- Simulation load profile ----------------------------------------
    # Show the currents (or EIS frequencies) the model will be sampled at
    # during each calibration trial. Polarization / HFR are drawn as a
    # staircase over the test-bench dwell convention; EIS uses the actual
    # measurement timestamps from the data file.
    st.divider()
    st.markdown("**Load profile preview** *(visualization only — does not change the loss)*")
    st.caption(
        "The calibration objective evaluates the experimental I_LOAD values "
        "directly. The dwell setting and staircase below only visualize the "
        "test-bench protocol; they do not change solver inputs or the loss."
    )

    # Editable dwell for Polarization / HFR — **affects the preview plot
    # only**. The objective evaluates the experimental I_LOAD values directly,
    # so the dwell value does not change the solver or loss.
    dwell = None
    if cfg["dataset"] == "Polarization":
        cfg["dwell_pola"] = float(st.number_input(
            "Dwell per point (s)",
            value=float(cfg.get("dwell_pola", DWELL_POLA_S)),
            min_value=0.1, step=5.0, format="%.1f",
            key="calib_dwell_pola",
            help="Test-bench hold time shown on the preview's time axis. "
                 "It does not change the calibration solver or loss.",
        ))
        dwell = cfg["dwell_pola"]
    elif cfg["dataset"] == "HFR":
        cfg["dwell_hfr"] = float(st.number_input(
            "Dwell per point (s)",
            value=float(cfg.get("dwell_hfr", DWELL_HFR_S)),
            min_value=0.1, step=5.0, format="%.1f",
            key="calib_dwell_hfr",
            help="Test-bench hold time shown on the preview's time axis. "
                 "It does not change the calibration solver or loss.",
        ))
        dwell = cfg["dwell_hfr"]
    else:
        st.caption(
            "EIS uses the actual `Temps (ms)` column shipped with the data."
        )

    fig = _build_load_profile_figure(
        cfg["dataset"], data, cfg["conditions"], dwell=dwell,
    )
    st.pyplot(fig, clear_figure=True)


def _dataset_key(label):
    return {"Polarization": "pola", "HFR": "hfr", "EIS": "eis"}[label]


def _build_data_figure(dataset, data, conditions):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = _style.PALETTE
    if dataset == "Polarization":
        for i, key in enumerate(conditions):
            df = data[key].dropna(subset=["I_LOAD", "VFC"]).sort_values("I_LOAD")
            ax.plot(df["I_LOAD"], df["VFC"], marker="o", markersize=4,
                    linewidth=1.4, label=key, color=palette[i % len(palette)])
        ax.set_xlabel("Load current $I_{LOAD}$ (A)")
        ax.set_ylabel("Stack voltage $V_{FC}$ (V)")
        ax.set_title("Polarization curves")
    elif dataset == "HFR":
        for i, key in enumerate(conditions):
            df = data[key].copy()
            df["R_real"] = df["R"].apply(_parse_hfr_R)
            df = df.dropna(subset=["I_LOAD", "R_real"]).sort_values("I_LOAD")
            ax.plot(df["I_LOAD"], df["R_real"], marker="s", markersize=4,
                    linewidth=1.4, label=key, color=palette[i % len(palette)])
        ax.set_xlabel("Load current $I_{LOAD}$ (A)")
        ax.set_ylabel("HFR (m$\\Omega$)")
        ax.set_title("High-frequency resistance")
    else:  # EIS — Nyquist plot
        # EIS column names are in French with accented characters; pick them
        # by prefix so any encoding quirk in the source file still matches.
        for i, key in enumerate(conditions):
            df = data[key]
            re_col = _find_col(df, "Partie r", exclude="stack")
            im_col = _find_col(df, "Partie i", exclude="stack")
            if re_col is None or im_col is None:
                continue
            sub = df.dropna(subset=[re_col, im_col])
            ax.plot(sub[re_col], -sub[im_col],
                    marker=".", markersize=3, linewidth=1.0,
                    label=key, color=palette[i % len(palette)])
        ax.set_xlabel("Re(Z) (m$\\Omega$)")
        ax.set_ylabel("-Im(Z) (m$\\Omega$)")
        ax.set_title("Nyquist spectra")
        ax.set_aspect("equal", adjustable="datalim")
    if len(conditions) <= 8:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# Experimental dwell-time convention. Polarization / HFR test-bench
# protocols hold each set-point for ~60 s to reach steady state before
# recording (Win/Wout updates between points are smooth). The static
# calibration solver is algebraic so the value only affects how the
# preview *looks*, not the loss landscape.
DWELL_POLA_S = 60.0
DWELL_HFR_S  = 30.0


def _build_load_profile_figure(dataset, data, conditions, *, dwell=None):
    """Plot the per-condition load profile as a staircase over time.

    * **Polarization / HFR** — there is no time column in the data, so we
      build a staircase using the dwell value passed in (falls back to
      the module-level default when ``dwell`` is ``None``). The static
      calibration solver is algebraic, so the time axis here represents
      the experimental protocol, not the simulator's internal time.
    * **EIS** — the data ships a ``Temps (ms)`` column with the actual
      measurement timestamps; we use it directly and ignore ``dwell``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    palette = _style.PALETTE
    if dataset in ("Polarization", "HFR"):
        if dwell is None:
            dwell = DWELL_POLA_S if dataset == "Polarization" else DWELL_HFR_S
        for i, key in enumerate(conditions):
            df = data[key]
            if "I_LOAD" not in df.columns:
                continue
            seq = np.sort(df["I_LOAD"].dropna().to_numpy(dtype=float))
            if seq.size == 0:
                continue
            # Staircase: hold seq[k] over [k*dwell, (k+1)*dwell].
            n = seq.size
            t = np.repeat(np.arange(n + 1) * dwell, 2)[1:-1]
            y = np.repeat(seq, 2)
            ax.plot(t, y, linewidth=1.5,
                    label=key, color=palette[i % len(palette)])
            ax.plot(np.arange(n) * dwell + dwell / 2, seq,
                    marker="o", markersize=3, linestyle="",
                    color=palette[i % len(palette)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$I_{LOAD}$ (A)")
        ax.set_title(
            f"Load profile  ·  dwell = {dwell:.0f} s/point  "
            f"(test-bench protocol; static solve is algebraic)"
        )
    else:  # EIS — actual experimental timestamps
        for i, key in enumerate(conditions):
            df = data[key]
            f_col = _find_col(df, "Fr")   # "Fréquence (Hz)"
            t_col = _find_col(df, "Temp") # "Temps (ms)"
            if f_col is None:
                continue
            freq = df[f_col].dropna().to_numpy(dtype=float)
            if t_col is not None:
                t_s = df[t_col].dropna().to_numpy(dtype=float) / 1000.0
                # Some sweeps drop NaNs differently between columns — pair
                # by min length to be safe.
                n = min(len(t_s), len(freq))
                t_s, freq = t_s[:n], freq[:n]
            else:
                t_s = np.arange(freq.size, dtype=float)
            if freq.size == 0:
                continue
            ax.plot(t_s, freq, marker=".", markersize=3, linewidth=1.0,
                    label=key, color=palette[i % len(palette)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.set_title("EIS frequency sweep over time")
    if 1 <= len(conditions) <= 6:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Column 2 — Optimizer settings + Start button
# ----------------------------------------------------------------------------
def _render_optimizer_panel(state):
    st.markdown("#### § Optimizer")

    cfg = state["calib"]
    cfg["target"] = st.selectbox(
        "Calibration target",
        options=DATASETS,
        index=DATASETS.index(cfg.get("target", "Polarization")),
        key="calib_target",
        help="Which experimental dataset the simulation is fitted against.",
    )
    mc1, mc2 = st.columns(2)
    cfg["model"] = mc1.selectbox(
        "Model variant",
        options=MODEL_VARIANTS,
        index=MODEL_VARIANTS.index(
            cfg.get("model", DEFAULT_MODEL_VARIANT)
        ),
        key="calib_model",
        help=("Static = PEMFC_stat algebraic solver (~50 ms / point, "
              "recommended).\n"
              "Dual-scale / Dynamic = transient PEMFC / PEMFC_dyn settled "
              "to steady state at every measurement current "
              "(~seconds per point — see the per-trial estimate below)."),
    )
    aux_disabled = (cfg["model"] == "Static")
    aux_index = 0 if cfg.get("aux_system", False) else 1
    aux_choice = mc2.selectbox(
        "Auxiliary system",
        options=AUX_CHOICES,
        index=aux_index,
        key="calib_aux_system",
        disabled=aux_disabled,
        help=("With: include the compressor / balance-of-plant (PEMFC_dyn).\n"
              "Without: skip BoP — the cell sees an ideal supply (PEMFC).\n"
              "Static is algebraic and has no BoP concept, so this is "
              "disabled when Static is selected."),
    )
    cfg["aux_system"] = (aux_choice == AUX_CHOICES[0]) and not aux_disabled

    # PDE solver — only meaningful for transient calibration (Static is
    # algebraic and has no integrator). Locked-in for the whole run.
    solver_disabled = (cfg["model"] == "Static")
    cfg["method"] = st.selectbox(
        "PDE solver",
        options=PDE_SOLVERS,
        index=PDE_SOLVERS.index(cfg.get("method", "BDF")),
        key="calib_method",
        disabled=solver_disabled,
        help=("ODE integrator passed to `scipy.integrate.solve_ivp`. "
              "The chosen solver is used for **every** trial in the "
              "calibration — the backend never silently falls back to a "
              "different one, so a trial that fails with this solver stays "
              "failed. Disabled for Static (algebraic, no integrator)."),
    )

    # Honest cost estimate so the user is not surprised by a multi-hour run.
    n_conds  = max(1, len(cfg.get("conditions") or []))
    n_pts_est = 8  # ~ 7-8 polarization points per condition
    if cfg["model"] == "Static":
        per_trial_s = 0.05 * n_pts_est * n_conds
    else:
        # Resolve transient class to set the right settle time.
        kind = calib_backend.resolve_transient_model(cfg["model"], cfg["aux_system"])
        settle = (calib_backend.SETTLE_S_PEMFC if kind == "PEMFC"
                  else calib_backend.SETTLE_S_PEMFC_DYN)
        # Wall-time per integration is roughly the settle time at default mesh.
        per_trial_s = settle * n_pts_est * n_conds
    total_s = per_trial_s * max(1, int(cfg.get("n_trials", 50)))
    if cfg["model"] == "Static":
        st.caption(
            f"ℹ️ Static evaluates `PEMFC_stat` (algebraic). "
            f"Estimated wall time: ~{per_trial_s:.1f} s/trial, "
            f"~{total_s:.0f} s total."
        )
    else:
        kind = calib_backend.resolve_transient_model(cfg["model"], cfg["aux_system"])
        st.warning(
            f"Transient calibration: each trial integrates **{kind}** for "
            f"{n_pts_est} currents × {n_conds} condition(s). "
            f"Estimated wall time: ~{per_trial_s/60:.1f} min/trial, "
            f"~{total_s/60:.0f} min ({total_s/3600:.1f} h) total. "
            f"Consider reducing **Trials** or the number of conditions, or "
            f"use Static for full sweeps and Transient for a final refinement.",
            icon="⏱️",
        )

    st.markdown("**Optimizer settings**")
    # TPE is the recommended default — see the OPTIMIZERS docstring above.
    # Re-probe optuna on every rerun (cheap; uses importlib.util.find_spec)
    # so a mid-session install is picked up on the next page refresh.
    optuna_ok = _is_optuna_available()
    def _opt_label(name):
        if name == "TPE (Optuna)" and not optuna_ok:
            return "TPE (Optuna — not installed)"
        return name
    cfg["optimizer"] = st.selectbox(
        "Optimizer",
        options=OPTIMIZERS,
        index=OPTIMIZERS.index(cfg.get("optimizer", OPTIMIZER_DEFAULT)),
        format_func=_opt_label,
        key="calib_optimizer",
        help=("TPE (Optuna) — recommended; sample-efficient Bayesian search.\n"
              "Genetic algorithm — population-based, robust but slower to converge.\n"
              "Grid search — exhaustive; only practical for ≤3 parameters."),
    )
    if cfg["optimizer"] == "TPE (Optuna)" and not optuna_ok:
        import sys
        st.warning(
            f"`optuna` is not importable in **this** Python environment:\n\n"
            f"`{sys.executable}`\n\n"
            f"Install it into that exact interpreter:\n\n"
            f"```\n\"{sys.executable}\" -m pip install optuna\n```\n\n"
            f"Then **stop and relaunch Streamlit** (a browser refresh is not "
            f"enough because Python only imports modules once per process). "
            f"GA and Grid will work without any re-install.",
            icon="📦",
        )
    c1, c2 = st.columns(2)
    cfg["n_trials"] = int(c1.number_input(
        "Trials", value=int(cfg.get("n_trials", 50)),
        min_value=1, step=10, key="calib_n_trials",
    ))
    cfg["seed"] = int(c2.number_input(
        "Random seed", value=int(cfg.get("seed", 42)),
        min_value=0, step=1, key="calib_seed",
        help=("Reproducibility seed for the sampler / population. "
              "Same seed + same bounds + same data = identical search trajectory. "
              "Change it if you want an independent run, or keep it fixed to "
              "share a result that someone else can reproduce exactly."),
    ))

    # ---- Parameters to fit + per-parameter bounds editor ----------------
    st.markdown("**Parameters to fit**")
    cfg["params"] = st.multiselect(
        "Vary",
        options=list(CALIB_PARAMS.keys()),
        default=cfg.get("params", list(DEFAULT_PARAMS)),
        key="calib_params",
        help="Each selected parameter gets a (low, high) bound below.",
    )
    if cfg["params"]:
        with st.expander("Edit bounds", expanded=False):
            for k in cfg["params"]:
                spec = CALIB_PARAMS[k]
                unit = spec[4]
                if _is_discrete(k):
                    # Discrete — multiselect over the allowed values.
                    choices = list(spec[5])
                    current = cfg["bounds"].get(k, choices)
                    if not isinstance(current, list):
                        current = list(choices)
                    selected = st.multiselect(
                        f"{k} candidates ({unit})",
                        options=choices,
                        default=[c for c in current if c in choices] or choices,
                        key=f"calib_disc_{k}",
                        help=f"{k} samples from this set (default = all "
                             f"{len(choices)} values).",
                    )
                    # Don't let the user empty the set — fall back to all choices.
                    cfg["bounds"][k] = selected if selected else list(choices)
                else:
                    # Continuous — (low, high) number inputs.
                    _, low_default, high_default, log_scale, *_ = spec
                    current = cfg["bounds"].get(k, (low_default, high_default))
                    if not (isinstance(current, tuple) and len(current) == 2):
                        current = (low_default, high_default)
                    lo, hi = current
                    bc1, bc2 = st.columns(2)
                    fmt = "%.3e" if log_scale else "%.4g"
                    lo_new = bc1.number_input(
                        f"{k} low ({unit})", value=float(lo),
                        format=fmt, key=f"calib_lo_{k}",
                    )
                    hi_new = bc2.number_input(
                        f"{k} high ({unit})", value=float(hi),
                        format=fmt, key=f"calib_hi_{k}",
                    )
                    cfg["bounds"][k] = (lo_new, hi_new)

    st.divider()
    if st.button(
        "▶ Start calibration",
        type="primary", use_container_width=True,
        key="calib_start_button",
        disabled=not cfg.get("params") or not cfg.get("conditions"),
        help=("Pick at least one parameter and one condition before "
              "starting." if not cfg.get("params") or not cfg.get("conditions")
              else None),
    ):
        state["calib_request"] = {
            "target":     cfg["target"],
            "model":      cfg["model"],
            "aux_system": cfg["aux_system"],
            "method":     cfg["method"],
            "n_trials":   cfg["n_trials"],
            "optimizer":  cfg["optimizer"],
            "seed":       cfg["seed"],
            "params":     list(cfg["params"]),
            "bounds":     {k: cfg["bounds"][k] for k in cfg["params"]},
            "conditions": list(cfg.get("conditions", [])),
        }
        _run_calibration(state)


# ----------------------------------------------------------------------------
# Backend driver (synchronous, with a live progress bar)
# ----------------------------------------------------------------------------
def _run_calibration(state):
    """Run the calibration synchronously, updating Streamlit widgets live.

    Stores the result in ``state["calib_result"]``. The result panel
    renders from that on the next rerun.
    """
    req = state["calib_request"]
    target = req["target"]
    if target == "EIS":
        st.error("EIS calibration is not implemented yet — pick Polarization or HFR.")
        return

    # Defensive: reuse the baseline params + the cached experimental data.
    from copy import deepcopy
    from config.initialize import parameters as _PARAMS_DEFAULT
    baseline = deepcopy(_PARAMS_DEFAULT)
    # Apply any user-edited bounds for parameters not in the search set?
    # No — only the variable parameters get sampled; the rest stay at default.

    try:
        data = _load(_dataset_key(target))
    except Exception as exc:
        st.error(f"Could not load {target} data: {exc}")
        return

    progress = st.progress(0.0, text="Initialising optimizer...")
    n_total = req["n_trials"]

    def _on_progress(done, total, best_loss):
        # Streamlit caps progress at 1.0; we may overshoot for some
        # drivers (DE expands beyond n_trials), so clamp.
        frac = min(1.0, done / max(1, total or n_total))
        progress.progress(
            frac,
            text=(f"{req['optimizer']}  ·  {done}/{total or n_total} evals  ·  "
                  f"best loss = {best_loss:.4g}"),
        )

    try:
        with st.spinner(f"Running {req['optimizer']}..."):
            result = calib_backend.run_calibration(
                req, baseline_params=baseline, data=data,
                on_progress=_on_progress,
            )
    except Exception as exc:
        st.error(f"Calibration failed: {type(exc).__name__}: {exc}")
        return
    finally:
        try:
            progress.empty()
        except Exception:
            pass

    state["calib_result"] = result
    st.success(
        f"Done in {result['elapsed_s']:.1f} s — best loss "
        f"{result['best_loss']:.4g} after {result['n_evals']} evals."
    )


# ----------------------------------------------------------------------------
# Column 3 — Calibration results
# ----------------------------------------------------------------------------
def _render_results_panel(state):
    st.markdown("#### § Calibration result")

    res = state.get("calib_result")
    req = state.get("calib_request")
    if res is None:
        st.info(
            "Configure the optimizer in the middle column, then click "
            "**▶ Start calibration**. Convergence, best parameters, and the "
            "best-fit overlay will appear here."
        )
        return

    # ---- Header strip ----------------------------------------------------
    model_label = res.get("resolved_class") or res.get("model_variant", "?")
    aux_tag = " (aux on)" if res.get("aux_system") else ""
    solver_tag = ("" if res.get("model_variant") == "Static"
                  else f"  ·  solver: **{res.get('method', 'BDF')}**")
    st.caption(
        f"optimizer: **{res['optimizer']}**  ·  target: **{res['target']}**  ·  "
        f"model: **{model_label}{aux_tag}**{solver_tag}  ·  "
        f"evals: **{res['n_evals']}**  ·  elapsed: **{res['elapsed_s']:.1f} s**  ·  "
        f"best loss: **{res['best_loss']:.4g}**"
    )

    # ---- Set-as-default button -----------------------------------------------
    # Copies the best-fit parameter values into the simulation page's
    # `state["params"]` dict so the next Run on the Simulation page uses
    # them directly. The button is disabled if the trial did not produce
    # a usable result (loss saturated at FAILURE_LOSS).
    disabled = res["best_loss"] >= 1e6 * 0.9   # near the sentinel
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        st.caption(
            "💡 Push the calibrated values into the Simulation page — "
            "the next Run there will use them as defaults."
        )
    with bc2:
        if st.button(
            "💾 Set as simulation defaults",
            key="calib_set_as_default_button",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            help=("Copy every entry in `best_params` into "
                  "`st.session_state[\"params\"]`, which is what the "
                  "Simulation page reads. Disabled when the calibration "
                  "did not produce a usable result."
                  if not disabled else
                  "Disabled: the best trial failed (loss at FAILURE_LOSS)."),
        ):
            # Ensure the sim-page params dict exists, then merge in the
            # calibrated values.  We update in place so the Simulation
            # page picks up the changes on its next rerun.
            sim_params = state.setdefault("params", {})
            updated = {}
            for k, v in res["best_params"].items():
                if k in sim_params or k in CALIB_PARAMS:
                    sim_params[k] = v
                    updated[k] = v
            state["params"] = sim_params
            st.success(
                f"Set {len(updated)} parameter{'s' if len(updated) != 1 else ''} "
                f"as simulation defaults: "
                f"{', '.join(sorted(updated.keys()))}."
            )
            st.toast(
                f"Updated {len(updated)} simulation parameters.",
                icon="✅",
            )

    # ---- Convergence curve ----------------------------------------------
    st.markdown("**Convergence**")
    history = res["history"]
    if history:
        trials = np.asarray([t for t, _ in history])
        losses = np.asarray([l for _, l in history])
        best_so_far = np.minimum.accumulate(losses)
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(trials, losses, marker=".", linestyle="",
                color=_style.PALETTE[4], alpha=0.55, label="trial loss")
        ax.plot(trials, best_so_far, linewidth=1.6,
                color=_style.PALETTE[0], label="best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Loss (MSE)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # ---- Best parameter table -------------------------------------------
    # Bounds can be either a (low, high) tuple (continuous params) or a
    # list of allowed values (discrete params like `e: [3, 4, 5]`). Handle
    # both so a discrete entry doesn't crash the unpack.
    st.markdown("**Best parameters**")
    bounds = (req or {}).get("bounds", {})
    rows = []
    for p, v in res["best_params"].items():
        b = bounds.get(p)
        if isinstance(b, (list,)):  # discrete
            lo_disp = min(b) if b else None
            hi_disp = max(b) if b else None
            search = f"{{{', '.join(str(c) for c in b)}}}"
        elif isinstance(b, tuple) and len(b) == 2:  # continuous
            lo_disp, hi_disp = b
            search = f"[{lo_disp:g}, {hi_disp:g}]" if all(
                isinstance(x, (int, float)) for x in b) else f"{b}"
        else:
            lo_disp = hi_disp = None
            search = "—"
        rows.append({"parameter": p, "search range": search, "best": v})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No parameters returned.")

    # ---- Best-fit overlay ------------------------------------------------
    st.markdown("**Best-fit overlay vs experimental data**")
    curves = res.get("best_curves") or {}
    if not curves:
        st.caption("No best-fit curves available.")
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    palette = _style.PALETTE
    for i, (cond_key, c) in enumerate(curves.items()):
        col = palette[i % len(palette)]
        ax.plot(c["i_meas"], c["y_meas"], marker="o", markersize=4,
                linewidth=0, label=f"{cond_key} (exp)", color=col)
        ax.plot(c["i_meas"], c["y_pred"], linestyle="--", linewidth=1.4,
                label=f"{cond_key} (fit)", color=col)
    ax.set_xlabel("Load current $I_{LOAD}$ (A)")
    if res["target"] == "Polarization":
        ax.set_ylabel("Per-cell voltage $U_{cell}$ (V)")
        ax.set_title("Polarization — fit vs measurement")
    else:
        ax.set_ylabel("HFR (m$\\Omega$)")
        ax.set_title("HFR — fit vs measurement")
    if len(curves) <= 6:
        ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.caption(res["message"])
