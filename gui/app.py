"""PEMFC simulator GUI — three-section compact Streamlit layout.

Run from the project root:

    streamlit run gui/app.py
"""

import os
import sys
from copy import deepcopy

# Make the project root importable when streamlit is launched from anywhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from gui import style as _style
# Matplotlib rcParams can be set at import time — no Streamlit interaction.
# Streamlit CSS injection (apply_streamlit) MUST happen after set_page_config,
# so it's called from inside main() below.
_style.apply_matplotlib()

from config.initialize import parameters as _PARAMS_DEFAULT
from config.initialize import operating_inputs as _OP_DEFAULT
from gui import parameters as panel_params
from gui import options as panel_options
from gui import results as panel_results
from gui import save as panel_save
from gui.runner import run as run_simulation


def _ensure_state():
    if "params" not in st.session_state:
        st.session_state["params"] = deepcopy(_PARAMS_DEFAULT)
    if "op_inputs" not in st.session_state:
        op = deepcopy({k: v for k, v in _OP_DEFAULT.items() if k != "current_density"})
        st.session_state["op_inputs"] = op
    st.session_state.setdefault("model_variant", "Dual-scale")
    st.session_state.setdefault("aux_system", True)
    st.session_state.setdefault("profile_kind", "Constant")
    st.session_state.setdefault("profile_cfg", {})
    st.session_state.setdefault("t_start", 0.0)
    st.session_state.setdefault("t_end", 20.0)
    st.session_state.setdefault("max_step", 0.1)
    st.session_state.setdefault("method", "BDF")
    st.session_state.setdefault("visible_groups", panel_params.DEFAULT_VISIBLE)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("running", False)

    # One-time migration: profile_cfg used to store currents in A/cm^2;
    # values < 100 are almost certainly stale A/cm^2 entries that need to
    # be rescaled to A/m^2 (the current convention).
    if not st.session_state.get("_units_A_per_m2"):
        pcfg = st.session_state["profile_cfg"]
        for key in ("i_const", "i_low", "i_high", "i_max", "i_dc"):
            if key in pcfg and pcfg[key] < 100:
                pcfg[key] = float(pcfg[key]) * 1e4
        st.session_state["_units_A_per_m2"] = True


def _trigger_run():
    s = st.session_state
    profile_func = panel_options.build_profile_func(s)

    polar_sweep = None
    if s["model_variant"] == "Static":
        pcfg = s["profile_cfg"]
        polar_sweep = {
            "i_max_A_cm2": pcfg.get("i_max", 1.65),
            "n_points": int(pcfg.get("n_steps", 30)),
        }

    try:
        with st.spinner("Solving..."):
            model, sol_or_polar, status = run_simulation(
                params=s["params"],
                op_inputs=s["op_inputs"],
                model_variant=s["model_variant"],
                profile_func=profile_func,
                t_span=(s["t_start"], s["t_end"]),
                max_step=s["max_step"],
                method=s["method"],
                polar_sweep=polar_sweep,
                aux_system=s.get("aux_system", True),
            )
    except Exception as exc:
        st.session_state["last_result"] = {
            "model": None, "solution": None, "polar": None,
            "status": {"runtime_s": 0.0, "n_steps": 0, "n_states": 0,
                       "success": False, "message": str(exc),
                       "model_variant": s["model_variant"],
                       "kind": "polar" if s["model_variant"] == "Static" else "transient"},
        }
        return

    if status["kind"] == "polar":
        st.session_state["last_result"] = {
            "model": model, "solution": None, "polar": sol_or_polar, "status": status,
        }
    else:
        st.session_state["last_result"] = {
            "model": model, "solution": sol_or_polar, "polar": None, "status": status,
        }


def main():
    st.set_page_config(page_title="PEMFC Simulator", layout="wide")
    # CSS injection — must come AFTER set_page_config (which has to be the
    # very first Streamlit call) but before any visible widgets are rendered.
    _style.apply_streamlit()
    _ensure_state()

    # ---- Hero title: centered Playfair Display, italic navy accent on
    #      "Simulator", uppercase tracked subtitle below.
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Playfair+Display:ital,wght@0,600;0,700;1,600;1,700'
        '&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
        '<div class="pemfc-hero">'
        '  <div class="title">PEMFC <span class="accent">Simulator</span></div>'
        '  <div class="rule"></div>'
        '  <div class="subtitle">'
        '    Proton Exchange Membrane Fuel Cell · 1D Dual-Scale Dynamic Model'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Three columns. The result column stacks two cards (results on top,
    # save/download below) so all three bottom edges share the same baseline.
    SECTION_HEIGHT  = 820
    SAVE_HEIGHT     = 270
    GAP_PX          = 16   # vertical gap Streamlit inserts between containers
    RESULTS_HEIGHT  = SECTION_HEIGHT - SAVE_HEIGHT - GAP_PX  # = 534

    col_p, col_o, col_r = st.columns([0.85, 1.0, 1.65], gap="medium")
    with col_p:
        with st.container(height=SECTION_HEIGHT, border=True):
            panel_params.render(st.session_state)
    with col_o:
        with st.container(height=SECTION_HEIGHT, border=True):
            panel_options.render(st.session_state)
    with col_r:
        with st.container(height=RESULTS_HEIGHT, border=True):
            panel_results.render(st.session_state, on_run=_trigger_run)
        with st.container(height=SAVE_HEIGHT, border=True):
            panel_save.render(st.session_state)


if __name__ == "__main__":
    main()
else:
    main()
