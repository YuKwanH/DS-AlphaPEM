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
    _status_strip(status)

    if not status.get("success"):
        return state

    if status["kind"] == "polar":
        _render_polar(result["polar"])
    else:
        _render_transient(result["model"], result["solution"])

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


def _render_polar(polar):
    fig, ax = plt.subplots(figsize=(6, 4))
    i_A_cm2 = polar["i_A_m2"] / 1e4
    ax.plot(i_A_cm2, polar["Ucell_V"], marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("Current density (A/cm$^2$)")
    ax.set_ylabel("Cell voltage (V)")
    ax.set_title("Polarization curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_transient(model, solution):
    tabs = st.tabs(["Cell performance", "Spatial profile", "Manifolds",
                    "Water content", "Degradation", "Custom"])

    with tabs[0]:
        _tab_cell_performance(model)
    with tabs[1]:
        _tab_spatial(model, solution)
    with tabs[2]:
        _tab_manifolds(model)
    with tabs[3]:
        _tab_water(model)
    with tabs[4]:
        _tab_degradation(model)
    with tabs[5]:
        _tab_custom(model)


def _tab_cell_performance(model):
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No time-domain output recorded.")
        return
    Ucell = np.asarray(model.echem_traj.get("Ucell", []))
    i_fc = np.asarray(model.echem_traj.get("i_fc", []))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    if i_fc.size:
        ax[0].plot(t[: len(i_fc)], i_fc)
    ax[0].set_xlabel(axis_label("t"))
    ax[0].set_ylabel(axis_label("i_fc"))
    ax[0].set_title("Load current")

    if Ucell.size:
        ax[1].plot(t[: len(Ucell)], Ucell, color=_style.PALETTE[1])
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


def _tab_manifolds(model):
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
    for ax, (key, title) in zip(axes.flatten(), keys):
        y = np.asarray(model.variables.get(key, []))
        if y.size:
            ax.plot(t[: len(y)], y, linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(axis_label("t"))
        ax.set_ylabel(axis_label(key))
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _tab_water(model):
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No water-content data recorded.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for name, ax_i in zip(("lambda_acl", "lambda_ccl"), ax):
        y = np.asarray(model.variables.get(name, []))
        if y.size:
            ax_i.plot(t[: len(y)], y, linewidth=1.4)
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


def _tab_degradation(model):
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No degradation data recorded.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    delta = np.asarray(model.variables.get("delta_mem", []))
    if delta.size:
        ax[0].plot(t[: len(delta)], delta, linewidth=1.4)
    ax[0].set_xlabel(axis_label("t"))
    ax[0].set_ylabel(axis_label("delta_mem"))
    ax[0].set_title("Membrane thinning")
    ax[0].grid(True, alpha=0.3)

    s_n = np.asarray(model.echem_traj.get("S_N", []))
    if s_n.size:
        ax[1].plot(t[: len(s_n)], s_n, linewidth=1.4)
    ax[1].set_xlabel(axis_label("t"))
    ax[1].set_ylabel(axis_label("S_N"))
    ax[1].set_title("Pt active surface")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _tab_custom(model):
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
    for name in picks:
        y = np.asarray(model.variables[name])
        label, unit = lookup_unit(name)
        legend = f"{name} [{unit}]" if unit else name
        ax.plot(t[: len(y)], y, label=legend, linewidth=1.2)
        units.add(unit)
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
