"""Section 2: simulator options.

Renders model-variant and test-profile selectors plus their sub-forms,
the time span, solver settings, and mesh resolution. Also previews the
resulting current-density profile so the user sees what they're about
to feed the model.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from gui import profiles
from gui.runner import MODEL_VARIANTS


SOLVERS = ("BDF", "Radau", "LSODA", "RK45")


def render(state):
    st.markdown("#### § 2 Options")

    _render_run_error(state)

    state["model_variant"] = st.radio(
        "Model variant",
        options=MODEL_VARIANTS,
        index=MODEL_VARIANTS.index(state.get("model_variant", "Dual-scale")),
        horizontal=False,
        key="opt_model_variant",
    )

    profile_kind = st.radio(
        "Test profile",
        options=profiles.PROFILE_KINDS,
        index=profiles.PROFILE_KINDS.index(state.get("profile_kind", "Constant")),
        key="opt_profile_kind",
    )
    state["profile_kind"] = profile_kind

    pcfg = state.setdefault("profile_cfg", {})

    if profile_kind == "Constant":
        pcfg["i_const"] = st.number_input(
            "Current density (A/m²)",
            value=float(pcfg.get("i_const", 4000.0)),
            min_value=0.0, step=500.0, format="%.4g",
            key="opt_i_const",
        )

    elif profile_kind == "Step":
        st.caption(
            "Periodic tanh-smoothed square load — defaults match "
            "`simulation/control/square load.ipynb`."
        )
        c1, c2 = st.columns(2)
        pcfg["step_tstart"] = c1.number_input(
            "Period start tstart (s)", value=float(pcfg.get("step_tstart", 0.0)),
            step=0.5, format="%.2f", key="opt_step_tstart",
        )
        pcfg["step_tend"] = c2.number_input(
            "Period end tend (s)", value=float(pcfg.get("step_tend", 6.0)),
            min_value=0.1, step=0.5, format="%.2f", key="opt_step_tend",
        )
        c3, c4 = st.columns(2)
        pcfg["i_low"] = c3.number_input(
            "i_low (A/m²)", value=float(pcfg.get("i_low", 20.0)),
            min_value=0.0, step=10.0, format="%.4g", key="opt_i_low",
        )
        pcfg["i_high"] = c4.number_input(
            "i_high (A/m²)", value=float(pcfg.get("i_high", 12000.0)),
            min_value=0.0, step=500.0, format="%.4g", key="opt_i_high",
        )
        c5, c6 = st.columns(2)
        pcfg["tau_switch"] = c5.number_input(
            "Ramp begin tau_switch (s)", value=float(pcfg.get("tau_switch", 1.0)),
            min_value=0.0, step=0.1, format="%.2f", key="opt_tau_switch",
        )
        pcfg["t_switch"] = c6.number_input(
            "Ramp duration t_switch (s)", value=float(pcfg.get("t_switch", 3.0)),
            min_value=0.05, step=0.1, format="%.2f", key="opt_t_switch",
        )

    elif profile_kind == "Polarization":
        c1, c2 = st.columns(2)
        pcfg["i_max"] = c1.number_input(
            "i_max (A/m²)", value=float(pcfg.get("i_max", 16500.0)),
            min_value=50.0, step=500.0, format="%.4g", key="opt_i_max",
        )
        pcfg["n_steps"] = c2.number_input(
            "Number of points", value=int(pcfg.get("n_steps", 30)),
            min_value=2, step=1, key="opt_n_steps",
        )
        pcfg["t_per_step"] = st.number_input(
            "Hold time per step (s)",
            value=float(pcfg.get("t_per_step", 60.0)),
            min_value=1.0, step=5.0, format="%.1f", key="opt_t_per_step",
        )

    elif profile_kind == "EIS":
        c1, c2 = st.columns(2)
        pcfg["i_dc"] = c1.number_input(
            "i_DC (A/m²)", value=float(pcfg.get("i_dc", 10000.0)),
            min_value=50.0, step=500.0, format="%.4g", key="opt_i_dc",
        )
        pcfg["ratio"] = c2.number_input(
            "AC ratio", value=float(pcfg.get("ratio", 0.05)),
            min_value=0.0, step=0.01, format="%.3f", key="opt_ratio",
        )
        pcfg["frequency"] = st.number_input(
            "Frequency (Hz)", value=float(pcfg.get("frequency", 1.0)),
            min_value=1e-3, step=0.1, format="%.4g", key="opt_freq",
        )

    elif profile_kind == "AST cycling":
        c1, c2 = st.columns(2)
        pcfg["period"] = c1.number_input(
            "Cycle period (s)", value=float(pcfg.get("period", 60.0)),
            min_value=1.0, step=1.0, format="%.1f", key="opt_period",
        )
        pcfg["smoothing"] = c2.number_input(
            "Smoothing", value=float(pcfg.get("smoothing", 4.0)),
            min_value=0.1, step=0.1, format="%.2f", key="opt_ast_smooth",
        )
        c3, c4 = st.columns(2)
        pcfg["I_low"] = c3.number_input(
            "I_low (A)", value=float(pcfg.get("I_low", 1.0)),
            min_value=0.0, step=0.5, format="%.2f", key="opt_I_low",
        )
        pcfg["I_high"] = c4.number_input(
            "I_high (A)", value=float(pcfg.get("I_high", 25.8)),
            min_value=0.0, step=0.5, format="%.2f", key="opt_I_high",
        )

    state["profile_cfg"] = pcfg

    st.markdown("**Time span & solver**")
    c1, c2 = st.columns(2)
    state["t_start"] = c1.number_input(
        "t_start (s)", value=float(state.get("t_start", 0.0)),
        step=1.0, format="%.2f", key="opt_t_start",
    )
    state["t_end"] = c2.number_input(
        "t_end (s)", value=float(state.get("t_end", 30.0)),
        step=1.0, format="%.2f", key="opt_t_end",
    )
    c3, c4 = st.columns(2)
    state["max_step"] = c3.number_input(
        "max_step (s)", value=float(state.get("max_step", 0.1)),
        min_value=1e-4, step=0.05, format="%.4g", key="opt_max_step",
    )
    state["method"] = c4.selectbox(
        "Method", options=SOLVERS,
        index=SOLVERS.index(state.get("method", "BDF")),
        key="opt_method",
    )

    st.markdown("**Mesh**")
    mc1, mc2, mc3 = st.columns(3)
    state["params"]["n_gdl"] = int(mc1.number_input(
        "n_gdl", value=int(state["params"].get("n_gdl", 10)),
        min_value=2, step=1, key="opt_n_gdl",
    ))
    state["params"]["n_mem"] = int(mc2.number_input(
        "n_mem", value=int(state["params"].get("n_mem", 10)),
        min_value=2, step=1, key="opt_n_mem",
    ))
    state["params"]["n_group_pt"] = int(mc3.number_input(
        "n_group_pt", value=int(state["params"].get("n_group_pt", 10)),
        min_value=2, step=1, key="opt_n_group_pt",
    ))

    profile_func = build_profile_func(state)
    state["profile_func"] = profile_func
    _draw_profile_preview(profile_func, (state["t_start"], state["t_end"]))

    return state


def _render_run_error(state):
    res = state.get("last_result")
    if not res:
        return
    status = res.get("status", {})
    if status.get("success"):
        return
    msg = status.get("message") or "Simulation did not complete successfully."
    variant = status.get("model_variant", "?")
    st.error(f"⚠ Last run failed ({variant}):\n\n{msg}")


def build_profile_func(state):
    pk = state["profile_kind"]
    pcfg = state["profile_cfg"]
    Aact = state["params"].get("Aact", 31e-4)

    if pk == "Constant":
        return profiles.constant(pcfg.get("i_const", 4000.0))
    if pk == "Step":
        return profiles.step(
            pcfg.get("step_tstart", 0.0), pcfg.get("step_tend", 6.0),
            pcfg.get("i_low", 20.0), pcfg.get("i_high", 12000.0),
            pcfg.get("tau_switch", 1.0), pcfg.get("t_switch", 3.0),
        )
    if pk == "Polarization":
        return profiles.polarization_ramp(
            pcfg.get("i_max", 16500.0),
            int(pcfg.get("n_steps", 30)),
            pcfg.get("t_per_step", 60.0),
        )
    if pk == "EIS":
        return profiles.eis(
            pcfg.get("i_dc", 10000.0), pcfg.get("ratio", 0.05),
            pcfg.get("frequency", 1.0),
        )
    if pk == "AST cycling":
        return profiles.ast_cycling(
            pcfg.get("period", 60.0),
            pcfg.get("I_low", 1.0),
            pcfg.get("I_high", 25.8),
            pcfg.get("smoothing", 4.0),
            Aact,
        )
    return profiles.constant(4000.0)


def _draw_profile_preview(profile_func, t_span):
    if t_span[1] <= t_span[0]:
        return
    ts, ys = profiles.sample(profile_func, t_span, n=300)
    fig, ax = plt.subplots(figsize=(4.5, 1.6))
    ax.plot(ts, np.asarray(ys), linewidth=1.4)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("i (A/m$^2$)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Current-density preview", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
