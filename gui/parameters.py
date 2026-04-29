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
        ("p",  "Aact",       "Active area",            "m^2",    "%.4g"),
    ],
    "GDL": [
        ("p",  "Hgdl",        "GDL thickness",         "m",      "%.4g"),
        ("p",  "epsilon_gdl", "GDL porosity",          "-",      "%.3f"),
        ("p",  "tau",         "Tortuosity exponent",   "-",      "%.3f"),
    ],
    "CL (catalyst layer)": [
        ("p",  "Hcl",         "CL thickness",          "m",      "%.4g"),
        ("p",  "epsilon_cl",  "CL porosity",           "-",      "%.3f"),
        ("p",  "epsilon_c",   "CL ionomer fraction",   "-",      "%.3f"),
        ("p",  "epsilon_mc",  "Micro-scale porosity",  "-",      "%.3f"),
        ("p",  "i0_c_ref",    "Cathode i0 ref",        "A/m^2",  "%.4g"),
        ("p",  "kappa_c",     "O2 reaction order",     "-",      "%.3f"),
        ("p",  "C_scl",       "CL capacitance",        "F/m^2",  "%.4g"),
    ],
    "MEM (membrane)": [
        ("p",  "Hmem",        "Membrane thickness",    "m",      "%.4g"),
        ("p",  "kappa_co",    "Conductivity constant", "-",      "%.3f"),
        ("p",  "Re",          "Electronic resistance", "Ohm",    "%.4g"),
        ("p",  "e",           "Exchange constant",     "-",      "%.3f"),
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
}

DEFAULT_VISIBLE = ["Operating", "GC (gas channel)", "GDL", "CL (catalyst layer)", "MEM (membrane)"]


def render(state):
    st.markdown("#### § 1 Parameters")

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

    for group_name in visible:
        with st.expander(group_name, expanded=(group_name == "Operating")):
            items = [
                it for it in PARAM_GROUPS[group_name]
                if (it[0] == "op" and it[1] in op) or (it[0] == "p" and it[1] in params)
            ]
            for row_start in range(0, len(items), 2):
                cols = st.columns(2, gap="small")
                for col_idx, item in enumerate(items[row_start:row_start + 2]):
                    store, key, label, unit, fmt = item
                    target = op if store == "op" else params
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
