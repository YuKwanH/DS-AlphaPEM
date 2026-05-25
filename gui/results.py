"""Section 3: result tabs.

Renders simulation output. Two top-level shapes are supported:
  * transient: a populated PEMFC/PEMFC_dyn model + scipy OdeResult.
  * polar    : a dict {"i_A_m2": array, "Ucell_V": array} from PEMFC_stat.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from modules.display import build_profile_figure
from gui import style as _style


# Variable name -> (descriptive label, unit). Used to populate axis labels
# and legends so plotted results are self-describing.
VAR_UNITS = {
    "t":         ("Time", "s"),
    "Ucell":     ("Cell voltage", "V"),
    "i_fc":      ("Current density", "A/m$^2$"),
    "Ueq":       ("Open-circuit voltage", "V"),
    "eta_act":   ("Activation overpotential", "V"),
    "eta_conc":  ("Concentration overpotential", "V"),
    "eta_c":     ("Cathode overpotential", "V"),
    "fdrop":     ("Liquid coverage factor", "-"),
    "S_N":       ("ECSA ratio", "-"),
    "Rmem":      ("Membrane resistance", r"$\Omega\,$m$^2$"),
    "Rccl":      ("Cathode CL resistance", r"$\Omega\,$m$^2$"),
    "Racl":      ("Anode CL resistance", r"$\Omega\,$m$^2$"),
    "Pasm":      ("Pressure", "Pa"),
    "Paem":      ("Pressure", "Pa"),
    "Pcsm":      ("Pressure", "Pa"),
    "Pcem":      ("Pressure", "Pa"),
    "Phi_asm":   ("Relative humidity", "-"),
    "Phi_aem":   ("Relative humidity", "-"),
    "Phi_csm":   ("Relative humidity", "-"),
    "Phi_cem":   ("Relative humidity", "-"),
    "Wcp":       ("Compressor flow", "kg/s"),
    "Wa_inj":    ("Anode injection", "kg/s"),
    "Wc_inj":    ("Cathode injection", "kg/s"),
    "Abp_a":     ("Anode BP valve area", "m$^2$"),
    "Abp_c":     ("Cathode BP valve area", "m$^2$"),
    "delta_mem": ("Membrane thickness", "m"),
    "C_N2":      ("N$_2$ concentration", "mol/m$^3$"),
    "C_Pt2_ccl": ("Pt$^{2+}$ in CCL", "mol/m$^3$"),
    "lambda_acl":("Water content $\\lambda$", "-"),
    "lambda_ccl":("Water content $\\lambda$", "-"),
}

# Prefix-keyed fallbacks for discretised variables (e.g. C_H2_agdl_3).
PREFIX_UNITS = [
    ("C_H2_",      "H$_2$ concentration", "mol/m$^3$"),
    ("C_O2_",      "O$_2$ concentration", "mol/m$^3$"),
    ("C_v_",       "Vapour concentration", "mol/m$^3$"),
    ("C_Pt2_mem_", "Pt$^{2+}$ in MEM", "mol/m$^3$"),
    ("s_",         "Liquid saturation", "-"),
    ("lambda_mem", "Water content $\\lambda$", "-"),
    ("Tagdl_",     "Anode GDL temperature", "K"),
    ("Tcgdl_",     "Cathode GDL temperature", "K"),
    ("Tmem_",      "Membrane temperature", "K"),
    ("Tacl",       "Anode CL temperature", "K"),
    ("Tccl",       "Cathode CL temperature", "K"),
    ("S_N_ccl_",   "ECSA / Pt bin", "-"),
    ("theta_ccl_", "Pt oxide coverage", "-"),
]


def lookup_unit(name):
    if name in VAR_UNITS:
        return VAR_UNITS[name]
    for prefix, label, unit in PREFIX_UNITS:
        if name.startswith(prefix):
            return label, unit
    return name, ""


def axis_label(name):
    label, unit = lookup_unit(name)
    return f"{label} ({unit})" if unit else label


def render(state):
    st.markdown("#### § 3 Results")

    result = state.get("last_result")
    if result is None:
        st.info("Configure parameters and options on the left, then click **▶ Run** above.")
        return state

    status = result["status"]

    # Status strip + Hold button on the same row. The button column needs
    # enough room for the "📌 Hold" text to stay on a single line; ratio
    # [3.5, 1] reserves ~22 % of the result column for the button.
    strip_col, hold_col = st.columns([3.5, 1])
    with strip_col:
        _status_strip(status)
    with hold_col:
        if status.get("success"):
            if st.button("📌 Hold", key="hold_button", use_container_width=True,
                         help="Pin this result; the NEXT simulation will be plotted "
                              "on top of it for direct comparison. Only one held "
                              "slot — pressing Hold again replaces the previous pin."):
                state["held_result"] = _snapshot_for_hold(result)

    # Held-result indicator with one-click release. Release is a compact
    # ✖ icon button to keep the caption readable.
    held = state.get("held_result")
    if held is not None:
        h_col, rel_col = st.columns([8, 1])
        h_col.caption(
            f"📌 **Held**: {held['label']}  ·  overlaid as dashed grey curves"
        )
        if rel_col.button("✖", key="release_button", use_container_width=True,
                          help="Release / forget the held result."):
            state["held_result"] = None
            held = None

    if not status.get("success"):
        return state

    if status["kind"] == "polar":
        _render_polar(result["polar"], held)
    else:
        _render_transient(result["model"], result["solution"], held)

    return state


def _status_strip(status):
    parts = [
        f"variant: **{status['model_variant']}**",
        f"runtime: **{status['runtime_s']:.2f} s**",
    ]
    if status["kind"] == "transient":
        parts.append(f"n_steps: **{status['n_steps']}**")
        parts.append(f"n_states: **{status['n_states']}**")
    else:
        parts.append(f"n_points: **{status['n_points']}**")
    parts.append("✓ ok" if status["success"] else "⚠ failed")
    st.caption(" · ".join(parts))
    if not status["success"] and status.get("message"):
        st.warning(status["message"])


# ---------------------------------------------------------------------------
# Hold / overlay helpers
# ---------------------------------------------------------------------------
def _snapshot_for_hold(result):
    """Take a lightweight deep copy of just the arrays needed to re-plot.

    We do NOT hold the live model object — its references can be huge or
    contain solver state. We pull only ``variables`` and ``echem_traj``
    for transient runs, and the i/Ucell arrays for polar runs.
    """
    status = result["status"]
    # Short label: strip parenthetical aux note, prepend run timestamp.
    raw = str(status.get("model_variant", "run"))
    base = raw.split(" (")[0]
    label = f"{base}"
    if "aux" in raw:
        label += f"  ({'aux on' if status.get('aux_system', True) else 'aux off'})"

    if status["kind"] == "polar":
        polar = result["polar"]
        return {
            "kind": "polar",
            "label": label,
            "polar": {
                "i_A_m2":  np.asarray(polar["i_A_m2"]).copy(),
                "Ucell_V": np.asarray(polar["Ucell_V"]).copy(),
            },
        }

    model = result["model"]
    return {
        "kind": "transient",
        "label": label,
        "variables":  {k: np.asarray(v).copy()
                       for k, v in getattr(model, "variables", {}).items()
                       if hasattr(v, "__len__") and not isinstance(v, (str, dict))},
        "echem_traj": {k: np.asarray(v).copy()
                       for k, v in getattr(model, "echem_traj", {}).items()
                       if hasattr(v, "__len__") and not isinstance(v, (str, dict))},
    }


# Visual style for the held overlay. Dashed grey so it's obviously a
# reference, not the current result. Same line for every plot so the user
# learns the convention.
_HELD_KW = dict(linestyle="--", linewidth=1.3, alpha=0.6, color="#475569")


def _overlay_held(ax, held, var_key, source="variables"):
    """Draw the held trace for ``var_key`` on ``ax`` if it exists.

    ``source`` is either ``"variables"`` (defaults) or ``"echem_traj"``
    (for derived electrochemistry like Ucell/i_fc).
    Returns True if anything was drawn (so the caller can add a legend).
    """
    if not held or held.get("kind") != "transient":
        return False
    t_h = np.asarray(held["variables"].get("t", []))
    y_h = np.asarray(held.get(source, {}).get(var_key, []))
    if t_h.size == 0 or y_h.size == 0:
        return False
    n = min(len(t_h), len(y_h))
    ax.plot(t_h[:n], y_h[:n], label=f"held: {held['label']}", **_HELD_KW)
    return True


def _render_polar(polar, held=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    i_A_cm2 = polar["i_A_m2"] / 1e4
    ax.plot(i_A_cm2, polar["Ucell_V"], marker="o", linewidth=1.5, markersize=4,
            label="current")
    if held and held.get("kind") == "polar":
        h = held["polar"]
        ax.plot(h["i_A_m2"] / 1e4, h["Ucell_V"], marker="o", markersize=3,
                label=f"held: {held['label']}", **_HELD_KW)
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Current density (A/cm$^2$)")
    ax.set_ylabel("Cell voltage (V)")
    ax.set_title("Polarization curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_transient(model, solution, held=None):
    tabs = st.tabs(["Cell performance", "Spatial profile", "Manifolds",
                    "Water content", "Degradation", "Custom"])

    with tabs[0]:
        _tab_cell_performance(model, held)
    with tabs[1]:
        _tab_spatial(model, solution)  # snapshot overlay not meaningful
    with tabs[2]:
        _tab_manifolds(model, held)
    with tabs[3]:
        _tab_water(model, held)
    with tabs[4]:
        _tab_degradation(model, held)
    with tabs[5]:
        _tab_custom(model, held)


def _tab_cell_performance(model, held=None):
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No time-domain output recorded.")
        return
    Ucell = np.asarray(model.echem_traj.get("Ucell", []))
    i_fc = np.asarray(model.echem_traj.get("i_fc", []))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    if i_fc.size:
        ax[0].plot(t[: len(i_fc)], i_fc, label="current")
    drew_h0 = _overlay_held(ax[0], held, "i_fc", source="echem_traj")
    if drew_h0:
        ax[0].legend(fontsize=7, loc="best")
    ax[0].set_xlabel(axis_label("t"))
    ax[0].set_ylabel(axis_label("i_fc"))
    ax[0].set_title("Load current")

    if Ucell.size:
        ax[1].plot(t[: len(Ucell)], Ucell, color=_style.PALETTE[1], label="current")
    drew_h1 = _overlay_held(ax[1], held, "Ucell", source="echem_traj")
    if drew_h1:
        ax[1].legend(fontsize=7, loc="best")
    ax[1].set_xlabel(axis_label("t"))
    ax[1].set_ylabel(axis_label("Ucell"))
    ax[1].set_title("Cell voltage")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _tab_spatial(model, solution):
    if solution is None or not hasattr(solution, "y") or solution.y.shape[1] == 0:
        st.info("No transient solution available.")
        return
    n_t = solution.y.shape[1]
    idx = st.slider("Time index", min_value=0, max_value=n_t - 1, value=n_t - 1, key="spatial_idx")
    t_val = float(solution.t[idx])
    st.caption(f"t = {t_val:.3f} s")
    try:
        fig = build_profile_figure(solution, model, t_index=idx)
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.error(f"Could not render spatial profile: {exc}")


def _tab_manifolds(model, held=None):
    keys = [
        ("Phi_asm", "Anode supply RH"),
        ("Pasm",    "Anode supply pressure"),
        ("Phi_csm", "Cathode supply RH"),
        ("Pcsm",    "Cathode supply pressure"),
        ("Phi_aem", "Anode exhaust RH"),
        ("Paem",    "Anode exhaust pressure"),
        ("Phi_cem", "Cathode exhaust RH"),
        ("Pcem",    "Cathode exhaust pressure"),
    ]
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No manifold data recorded.")
        return
    fig, axes = plt.subplots(4, 2, figsize=(10, 8))
    any_held = False
    for ax, (key, title) in zip(axes.flatten(), keys):
        y = np.asarray(model.variables.get(key, []))
        if y.size:
            ax.plot(t[: len(y)], y, linewidth=1.2, label="current")
        if _overlay_held(ax, held, key):
            any_held = True
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(axis_label("t"))
        ax.set_ylabel(axis_label(key))
        ax.grid(True, alpha=0.3)
    if any_held:
        axes.flatten()[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _tab_water(model, held=None):
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No water-content data recorded.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for name, ax_i in zip(("lambda_acl", "lambda_ccl"), ax):
        y = np.asarray(model.variables.get(name, []))
        if y.size:
            ax_i.plot(t[: len(y)], y, linewidth=1.4, label="current")
        if _overlay_held(ax_i, held, name):
            ax_i.legend(fontsize=7, loc="best")
        ax_i.set_title(name)
        ax_i.set_xlabel(axis_label("t"))
        ax_i.set_ylabel(axis_label(name))
        ax_i.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    mem_keys = sorted(
        (k for k in model.variables if k.startswith("lambda_mem_")),
        key=lambda s: int(s.rsplit("_", 1)[1]) if s.rsplit("_", 1)[1].isdigit() else 0,
    )
    if mem_keys:
        mat = np.array([model.variables[k] for k in mem_keys])
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        im = ax2.imshow(mat, aspect="auto", origin="lower",
                        extent=[float(t[0]), float(t[-1]), 1, len(mem_keys)])
        ax2.set_xlabel(axis_label("t"))
        ax2.set_ylabel("Membrane node")
        ax2.set_title("Membrane water content $\\lambda$ (-)")
        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label(r"$\lambda$ (-)")
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)


def _tab_degradation(model, held=None):
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No degradation data recorded.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    delta = np.asarray(model.variables.get("delta_mem", []))
    if delta.size:
        ax[0].plot(t[: len(delta)], delta, linewidth=1.4, label="current")
    if _overlay_held(ax[0], held, "delta_mem"):
        ax[0].legend(fontsize=7, loc="best")
    ax[0].set_xlabel(axis_label("t"))
    ax[0].set_ylabel(axis_label("delta_mem"))
    ax[0].set_title("Membrane thinning")
    ax[0].grid(True, alpha=0.3)

    s_n = np.asarray(model.echem_traj.get("S_N", []))
    if s_n.size:
        ax[1].plot(t[: len(s_n)], s_n, linewidth=1.4, label="current")
    if _overlay_held(ax[1], held, "S_N", source="echem_traj"):
        ax[1].legend(fontsize=7, loc="best")
    ax[1].set_xlabel(axis_label("t"))
    ax[1].set_ylabel(axis_label("S_N"))
    ax[1].set_title("Pt active surface")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _tab_custom(model, held=None):
    options = sorted(k for k in model.variables.keys() if k != "t")
    if not options:
        st.info("No variables recorded.")
        return
    picks = st.multiselect("Variables", options=options,
                           default=options[:1], key="custom_picks")
    if not picks:
        return
    t = np.asarray(model.variables["t"])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    units = set()
    drew_held = False
    for name in picks:
        y = np.asarray(model.variables[name])
        label, unit = lookup_unit(name)
        legend = f"{name} [{unit}]" if unit else name
        ax.plot(t[: len(y)], y, label=legend, linewidth=1.2)
        units.add(unit)
        # Overlay the held trace for the same variable if available.
        if _overlay_held(ax, held, name):
            drew_held = True
    ax.set_xlabel(axis_label("t"))
    if len(units) == 1:
        only_unit = next(iter(units))
        ax.set_ylabel(f"value ({only_unit})" if only_unit else "value")
    else:
        ax.set_ylabel("value (mixed units — see legend)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
