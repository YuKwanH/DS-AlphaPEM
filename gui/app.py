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
_style.apply()

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
    st.session_state.setdefault("profile_kind", "Constant")
    st.session_state.setdefault("profile_cfg", {})
    st.session_state.setdefault("t_start", 0.0)
    st.session_state.setdefault("t_end", 30.0)
    st.session_state.setdefault("max_step", 0.1)
    st.session_state.setdefault("method", "BDF")
    st.session_state.setdefault("visible_groups", panel_params.DEFAULT_VISIBLE)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("running", False)


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
    _ensure_state()

    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown("### PEMFC Simulator")
        st.caption("Configure parameters · pick simulator options · run · inspect results.")
    with header_right:
        st.write("")
        st.button("▶ Run", type="primary", use_container_width=True,
                  on_click=_trigger_run, key="run_button")

    st.divider()

    SECTION_HEIGHT = 820

    col_p, col_o, col_r = st.columns([0.85, 1.0, 1.65], gap="medium")
    with col_p:
        with st.container(height=SECTION_HEIGHT, border=True):
            panel_params.render(st.session_state)
    with col_o:
        with st.container(height=SECTION_HEIGHT, border=True):
            panel_options.render(st.session_state)
    with col_r:
        with st.container(height=SECTION_HEIGHT, border=True):
            panel_results.render(st.session_state)

    # Save / download box, aligned under the result section.
    _spacer_left, col_save = st.columns([0.85 + 1.0, 1.65], gap="medium")
    with col_save:
        with st.container(height=260, border=True):
            panel_save.render(st.session_state)


if __name__ == "__main__":
    main()
else:
    main()
