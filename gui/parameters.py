"""Section 1: physics-parameter and control-input widgets.

Renders editable widgets for every entry in `parameters` and `operating_inputs`,
grouped by physical region (GC / GDL / CL / MEM / ...) so the user can hide
groups they're not currently tweaking. The `current_density` callable is
deliberately NOT rendered here -- it lives in section 2's test-profile form.
"""

import streamlit as st


PARAM_GROUPS = {
    "Operating": [
        ("op", "Tfc",        "Cell temperature",       "K",      "%.2f"),
        ("op", "Pa_des",     "Anode supply pressure",  "Pa",     "%.4g"),
        ("op", "Pc_des",     "Cathode supply pressure","Pa",     "%.4g"),
        ("op", "Phi_a_des",  "Anode RH",               "-",      "%.3f"),
        ("op", "Phi_c_des",  "Cathode RH",             "-",      "%.3f"),
        ("op", "Sa",         "Anode stoichiometry",    "-",      "%.3f"),
        ("op", "Sc",         "Cathode stoichiometry",  "-",      "%.3f"),
        ("op", "Imin_aux",   "Min auxiliary current",  "A",      "%.2f"),
    ],
    "GC (gas channel)": [
        ("p",  "Hgc",        "Channel height",         "m",      "%.4g"),
        ("p",  "Wgc",        "Channel width",          "m",      "%.4g"),
        ("p",  "Lgc",        "Channel length",         "m",      "%.4g"),
    ],
    "GDL": [
        ("p",  "Hgdl",        "GDL thickness",         "m",      "%.4g"),
        ("p",  "epsilon_gdl", "GDL porosity",          "-",      "%.3f"),
        ("p",  "tau",         "Tortuosity exponent",   "-",      "%.3f"),
        ("p",  "e",           "Bruggeman exponent",    "-",      "%d"),
    ],
    "CL (catalyst layer)": [
        # Electrochemistry (grouped together at the top so cathode-reaction
        # tuning knobs sit next to each other).
        ("p",  "OCV",         "Open-circuit voltage",  "V",      "%.4f"),
        ("p",  "i0_c_ref",    "Cathode i0 ref",        "A/m^2",  "%.4g"),
        ("p",  "kappa_c",     "O2 reaction order",     "-",      "%.3f"),
        ("p",  "C_scl",       "CL capacitance",        "F/m^2",  "%.4g"),
        # Geometry & porosity (Aact is the CL's active area — same face
        # area for anode and cathode CL, so it lives here rather than in
        # the gas channel section it used to be in).
        ("p",  "Aact",        "Active area",           "m^2",    "%.4g"),
        ("p",  "Hcl",         "CL thickness",          "m",      "%.4g"),
        ("p",  "epsilon_cl",  "CL porosity",           "-",      "%.3f"),
        ("p",  "epsilon_c",   "CL ionomer fraction",   "-",      "%.3f"),
        ("p",  "epsilon_mc",  "Micro-scale porosity",  "-",      "%.3f"),
    ],
    "MEM (membrane)": [
        ("p",  "Hmem",        "Membrane thickness",    "m",      "%.4g"),
        ("p",  "kappa_co",    "Conductivity constant", "-",      "%.3f"),
        ("p",  "Re",          "Electronic resistance", "Ohm",    "%.4g"),
    ],
    "Saturation transitions": [
        ("p",  "a_slim",      "a_slim",                "-",      "%.3f"),
        ("p",  "b_slim",      "b_slim",                "-",      "%.3f"),
        ("p",  "a_switch",    "a_switch",              "-",      "%.3f"),
    ],
    "Numerics": [
        ("p",  "max_step",    "ODE max step",          "s",      "%.4g"),
        ("p",  "n_gdl",       "GDL nodes",             "-",      "%d"),
        ("p",  "n_mem",       "Membrane nodes",        "-",      "%d"),
        ("p",  "n_group_pt",  "Pt-particle bins",      "-",      "%d"),
    ],
    # Chemical rate constants for the Pt surface reactions (micro-scale
    # catalyst-layer degradation). Defined as module globals in
    # model/coefficients.py; the GUI runner pushes edited values into every
    # consuming module (kinetic_eq / state_eq / model) before each run.
    "Micro-scale CL": [
        ("kin", "k1",        "Pt dissolution fwd rate",           "-", "%.6g"),
        ("kin", "k1_ref",    "Pt dissolution bwd rate",           "-", "%.6g"),
        ("kin", "k2",        "Pt oxidation fwd rate",             "-", "%.6g"),
        ("kin", "k2_ref",    "Pt oxidation bwd rate",             "-", "%.6g"),
        ("kin", "k3",        "Pt-oxide chemical dissolution",     "-", "%.6g"),
        ("kin", "krdp",      "Pt redeposition rate",              "-", "%.6g"),
        ("kin", "k4",        "Reaction 4 rate (reserved)",        "-", "%.6g"),
        ("kin", "k5",        "Reaction 5 rate (reserved)",        "-", "%.6g"),
        ("kin", "kdet_ref",  "Pt detachment ref rate",            "-", "%.6g"),
    ],
}

DEFAULT_VISIBLE = ["Operating", "GC (gas channel)", "GDL", "CL (catalyst layer)", "MEM (membrane)"]


def render(state):
    from config import user_defaults as _user_defaults
    # Compact "Save" pill in the top-right of the section header.
    # Column ratio 4:1 gives the pill ~80 px, and `use_container_width`
    # is left False so the pill sizes to its content (see the
    # `.st-key-param_save_default` block in gui/style.py).
    title_col, save_col = st.columns([4, 1], gap="small")
    title_col.markdown("#### § 1 Parameters")
    if save_col.button("Save", key="param_save_default",
                       help="Save the current parameter values as the "
                            "default. They will be pre-loaded next time "
                            "the GUI opens."):
        _user_defaults.save_parameters(state["params"], state["op_inputs"],
                                       state.get("kinetic_consts", {}))
        st.toast("Parameters saved as default.", icon="💾")

    visible = st.multiselect(
        "Show region",
        options=list(PARAM_GROUPS.keys()),
        default=state.get("visible_groups", DEFAULT_VISIBLE),
        key="param_visible_groups",
        label_visibility="collapsed",
    )
    state["visible_groups"] = visible

    params = state["params"]
    op = state["op_inputs"]
    kin = state.setdefault("kinetic_consts", {})

    _stores = {"op": op, "p": params, "kin": kin}

    for group_name in visible:
        with st.expander(group_name, expanded=(group_name == "Operating")):
            items = [
                it for it in PARAM_GROUPS[group_name]
                if it[1] in _stores.get(it[0], {})
            ]
            for row_start in range(0, len(items), 2):
                cols = st.columns(2, gap="small")
                for col_idx, item in enumerate(items[row_start:row_start + 2]):
                    store, key, label, unit, fmt = item
                    target = _stores[store]
                    current = target[key]
                    widget_label = f"{key} ({unit})" if unit not in ("", "-") else key
                    with cols[col_idx]:
                        if fmt == "%d":
                            target[key] = st.number_input(
                                widget_label, value=int(current), step=1,
                                help=label, key=f"w_{store}_{key}",
                            )
                        else:
                            target[key] = st.number_input(
                                widget_label, value=float(current), format=fmt,
                                help=label, key=f"w_{store}_{key}",
                            )

    return state
