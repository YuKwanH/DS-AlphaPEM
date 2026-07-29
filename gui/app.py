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
from gui import calibration as panel_calibration
from gui.runner import run as run_simulation
from gui.runner import run_0d_companion
from config import user_defaults as _user_defaults


def _ensure_state():
    # Load user-saved overrides (from config/user_defaults.json) exactly once
    # per fresh session; the "Save as default" buttons in each panel write to
    # that file so tweaks persist across GUI restarts.
    _overrides = _user_defaults.load()

    if "params" not in st.session_state:
        p = deepcopy(_PARAMS_DEFAULT)
        p.update(_overrides.get("parameters", {}))
        st.session_state["params"] = p
    if "op_inputs" not in st.session_state:
        op = deepcopy({k: v for k, v in _OP_DEFAULT.items() if k != "current_density"})
        op.update(_overrides.get("op_inputs", {}))
        st.session_state["op_inputs"] = op
    if "kinetic_consts" not in st.session_state:
        # Pt-surface reaction rate constants ("Micro-scale CL" group).
        # Defaults come from the module globals in model/coefficients.py;
        # user-saved overrides win. The runner pushes these into the
        # consuming modules before each simulation.
        import model.coefficients as _coeffs
        kin = {k: float(getattr(_coeffs, k))
               for k in ("k1", "k1_ref", "k2", "k2_ref", "k3",
                         "krdp", "k4", "k5", "kdet_ref")}
        kin.update(_overrides.get("kinetic_consts", {}))
        st.session_state["kinetic_consts"] = kin

    _opts = _overrides.get("options", {})
    st.session_state.setdefault("model_variant", _opts.get("model_variant", "Dual-scale"))
    st.session_state.setdefault("aux_system",    _opts.get("aux_system",    False))
    st.session_state.setdefault("profile_kind",  _opts.get("profile_kind",  "Step"))
    st.session_state.setdefault("profile_cfg",   _opts.get("profile_cfg", {
        "step_tstart": 0.0,
        "step_tend": 6.0,
        "i_low": 1000.0,
        "i_high": 14500.0,
        "tau_switch": 1.5,
        "t_switch": 1.0,
    }))
    st.session_state.setdefault("t_start",       _opts.get("t_start",       0.0))
    st.session_state.setdefault("t_end",         _opts.get("t_end",         30000.0))
    st.session_state.setdefault("max_step",      _opts.get("max_step",      0.1))
    st.session_state.setdefault("method",        _opts.get("method",        "BDF"))
    st.session_state.setdefault("visible_groups", panel_params.DEFAULT_VISIBLE)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("benchmark_0d", False)

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
                aux_system=s.get("aux_system", False),
                kinetic_consts=s.get("kinetic_consts"),
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

    # 0D benchmark companion: if the user ticked "Compare with 0D model",
    # also run the lumped-parameter 0D model with the same params, profile
    # and time span (or polar sweep) and stash the result alongside.
    st.session_state["last_result"]["benchmark_0d"] = None
    if s.get("benchmark_0d"):
        try:
            with st.spinner("Running 0D benchmark..."):
                bench = run_0d_companion(
                    params=s["params"],
                    op_inputs=s["op_inputs"],
                    profile_func=profile_func,
                    t_span=(s["t_start"], s["t_end"]),
                    max_step=s["max_step"],
                    method=s["method"],
                    polar_sweep=polar_sweep,
                )
            st.session_state["last_result"]["benchmark_0d"] = bench
        except Exception as exc:
            st.session_state["last_result"]["benchmark_0d"] = {
                "variables": {}, "echem_traj": {}, "polar": None,
                "status": {"success": False, "message": str(exc),
                           "kind": "polar" if polar_sweep else "transient",
                           "runtime_s": 0.0, "n_steps": 0, "n_points": 0},
            }


# Vertical-rhythm constants shared by both pages.
SECTION_HEIGHT  = 820
SAVE_HEIGHT     = 270
GAP_PX          = 16   # vertical gap Streamlit inserts between containers
RESULTS_HEIGHT  = SECTION_HEIGHT - SAVE_HEIGHT - GAP_PX  # = 534


def _render_nav_rail():
    """Slim two-button page selector on the left side of the app.

    Each button fills roughly half the viewport height; the active page is
    drawn with `type="primary"` (navy fill, white text) and the inactive
    one with the default outlined style.
    """
    page = st.session_state.get("page", "simulation")
    if st.button(
        "⚡\n\n**Simulation**",
        key="nav_sim",
        use_container_width=True,
        type=("primary" if page == "simulation" else "secondary"),
        help="Run a PEMFC simulation with the parameter / options panels.",
    ):
        st.session_state["page"] = "simulation"
        st.rerun()
    if st.button(
        "📈\n\n**Calibration**",
        key="nav_cal",
        use_container_width=True,
        type=("primary" if page == "calibration" else "secondary"),
        help="Calibrate model parameters against experimental data.",
    ):
        st.session_state["page"] = "calibration"
        st.rerun()


def _render_simulation_page():
    """The original 3-column simulation layout: parameters · options · results."""
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


def _render_calibration_page():
    """Three-column calibration workflow (data viewer · optimizer · result)."""
    panel_calibration.render(st.session_state, section_height=SECTION_HEIGHT)


def main():
    st.set_page_config(page_title="PEMFC Simulator", layout="wide")
    # CSS injection — must come AFTER set_page_config (which has to be the
    # very first Streamlit call) but before any visible widgets are rendered.
    _style.apply_streamlit()
    _ensure_state()
    # Page routing: "simulation" (the original 3-column workflow) or
    # "calibration" (the new placeholder). The nav rail toggles between them.
    st.session_state.setdefault("page", "simulation")

    # Load Inter for body UI (no hero title — removed for a cleaner header).
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@400;500;600&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    # Outermost split: slim left nav rail + main content area.
    nav_col, main_col = st.columns([0.06, 0.94], gap="small")
    with nav_col:
        _render_nav_rail()
    with main_col:
        if st.session_state["page"] == "simulation":
            _render_simulation_page()
        else:
            _render_calibration_page()


if __name__ == "__main__":
    main()
else:
    main()
