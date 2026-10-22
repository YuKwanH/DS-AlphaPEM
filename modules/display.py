import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from config.settings import (
    operating_inputs, parameters, borders, nodes,
    nodes_names_vp, nodes_names_H2, nodes_name_O2,
    nodes_names_s, nodes_lambda, nodes_T,
    expand_profile_on_nodes, get_plot_properties,
)
from model.coefficients import C_v_sat


PROFILE_PANELS = (
    ("v",          "Vapor concentration $(mol/m^3)$"),
    ("O2",         "Oxygen concentration $(mol/m^3)$"),
    ("H2",         "Hydrogen concentration $(mol/m^3)$"),
    ("saturation", "Liquid saturation $(-)$"),
    ("lambda",     r"Water content $\lambda$ $(-)$"),
    ("T",          "Temperature $(K)$"),
)

DYN_NAME_GROUPS = (
    ("v",          nodes_names_vp),
    ("O2",         nodes_name_O2),
    ("H2",         nodes_names_H2),
    ("saturation", nodes_names_s),
    ("lambda",     nodes_lambda),
    ("T",          nodes_T),
)

# Membrane-zoom insets for the crossover panels. The membrane is ~1/30 of a
# GDL thick, so its profile collapses to a vertical line at full scale.
# Each inset sits in the half of its panel that never carries data: H2 only
# exists on the anode side + membrane (right half empty), O2 only on the
# membrane + cathode side (left half empty). Bounds are axes fractions
# (x0, y0, width, height).
ZOOM_INSETS = {
    "H2": (0.55, 0.22, 0.42, 0.66),
    "O2": (0.05, 0.30, 0.42, 0.62),
}


def _zoom_window():
    """x-range of the membrane, padded slightly so the edge nodes are visible.

    The pad stays well under Hcl/2, so the catalyst-layer nodes (and their
    much larger gas-phase concentrations) fall outside the window and do
    not flatten the membrane gradient when the y-range is autoscaled.
    """
    x_lo, x_hi = borders[3], borders[4]
    pad = 0.04 * (x_hi - x_lo)
    return x_lo - pad, x_hi + pad


def _add_zoom_inset(ax, bounds):
    inset = ax.inset_axes(bounds)
    x_lo, x_hi = _zoom_window()
    inset.set_xlim(x_lo, x_hi)
    for x in (borders[3], borders[4]):
        inset.axvline(x=x, color="tab:gray", linestyle="--", linewidth=0.7, alpha=0.6)
    inset.set_facecolor("white")
    inset.grid(True, alpha=0.25)
    inset.set_title("Membrane (zoom)", fontsize=8, pad=3)
    inset.tick_params(labelsize=7)
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(4))
    inset.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    inset.xaxis.get_offset_text().set_fontsize(7)
    # Mark the zoomed band on the parent panel so the inset reads as a
    # magnification of that strip, not a separate plot.
    ax.axvspan(borders[3], borders[4], color="tab:blue", alpha=0.08, linewidth=0)
    ax._mfc_zoom = inset


def _rescale_zoom_y(inset):
    """Autoscale the inset's y-range to the data inside its x-window only."""
    x_lo, x_hi = inset.get_xlim()
    ys = []
    for line in inset.get_lines():
        if line.get_gid() != "mfc_profile":   # skip the axvline border markers
            continue
        x, y = (np.asarray(a, dtype=float) for a in line.get_data())
        mask = (x >= x_lo) & (x <= x_hi) & np.isfinite(y)
        ys.extend(y[mask].tolist())
    if not ys:
        return
    y_lo, y_hi = min(ys), max(ys)
    span = y_hi - y_lo
    scale = max(abs(y_lo), abs(y_hi))
    if span > 1e-9 * max(scale, 1.0):
        pad = 0.12 * span
    else:
        # Flat profile (e.g. crossover not yet developed): give it a readable
        # band instead of a ±1e-12 window with a meaningless offset label.
        pad = max(0.05 * scale, 1e-3)
    inset.set_ylim(y_lo - pad, y_hi + pad)


def _setup_axes(fig=None, axes=None, Tfc=None):
    """Create a 6-panel through-plane figure or reuse the one passed in.

    Layer borders, the saturation-vapor reference line, and titles are drawn
    on first creation only — subsequent calls keep them and just append data.
    """
    if fig is not None and axes is not None:
        return fig, axes

    n_panels = len(PROFILE_PANELS)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), sharex=False)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    Tfc_ref = Tfc if Tfc is not None else operating_inputs["Tfc"]
    for ax, (profile_key, title) in zip(axes, PROFILE_PANELS):
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.25)
        for x in borders:
            ax.axvline(x=x, color="tab:gray", linestyle="--", linewidth=0.7, alpha=0.4)
        if profile_key == "v":
            ax.hlines(C_v_sat(Tfc_ref), xmin=nodes[0], xmax=nodes[-1],
                      colors="tab:gray", linestyles=":", linewidth=1.0, alpha=0.7,
                      label=r"$C_{v,sat}(T_{fc})$")
        if profile_key == "lambda":
            ax.set_xlim(borders[2], borders[5])
        if profile_key in ZOOM_INSETS:
            _add_zoom_inset(ax, ZOOM_INSETS[profile_key])
    return fig, axes


def _resolve_style(cond_key, color, linestyle, marker):
    """Map a condition key to (color, linestyle, marker), letting explicit kwargs win."""
    if cond_key is None:
        return color, linestyle or "-", marker or "o"
    style = get_plot_properties(cond_key)
    return (color or style["color"],
            linestyle or style["linestyle"],
            marker or style["marker"])


def _plot_panels(axes, profile_1d, label, color, linestyle, marker, linewidth, markersize):
    for ax, (profile_key, _) in zip(axes, PROFILE_PANELS):
        values = profile_1d.get(profile_key)
        if values is None or len(values) == 0:
            continue
        y = expand_profile_on_nodes(profile_key, values)
        line, = ax.plot(nodes, y, label=label,
                        color=color, linestyle=linestyle, marker=marker,
                        linewidth=linewidth, markersize=markersize)
        inset = getattr(ax, "_mfc_zoom", None)
        if inset is not None:
            # Reuse the drawn line's colour so overlaid conditions keep the
            # same colour in the inset as in the main panel.
            inset.plot(nodes, y, color=line.get_color(), linestyle=linestyle,
                       marker=marker, linewidth=linewidth, markersize=markersize,
                       gid="mfc_profile")
            _rescale_zoom_y(inset)


def _read_dyn_value(source, name, t_index):
    """Pull one variable value out of a model, dict, or list-of-dicts source."""
    if hasattr(source, "variables") and name in source.variables:
        seq = source.variables[name]
        return seq[t_index] if hasattr(seq, "__len__") else seq
    if isinstance(source, dict):
        v = source[name]
        if hasattr(v, "__len__") and not isinstance(v, str):
            return v[t_index]
        return v
    raise TypeError(f"Cannot read '{name}' from source of type {type(source).__name__}")


def extract_profile_dyn(source, t_index=-1):
    """Build the 6-panel through-plane profile dict from a dynamic-model snapshot.

    `source` may be a `PEMFC_dyn` / `PEMFC` instance (after `_recovery`) or a
    dict mapping variable name -> scalar/list (e.g. `dyn_log_all[cond]["states"]`
    where each entry is a list indexed by current).
    """
    profile = {}
    for key, names in DYN_NAME_GROUPS:
        try:
            profile[key] = [_read_dyn_value(source, n, t_index) for n in names]
        except (KeyError, IndexError, TypeError):
            profile[key] = []
    return profile


def extract_profile_stat(sol):
    """Build the 6-panel through-plane profile dict from a `PEMFC_stat.solve()` result.

    Static GDL arrays run from the gas-channel side to the catalyst-layer side.
    To share node ordering with the dynamic model (`agdl_1` at AGC, `cgdl_1` at
    CCL, `mem_1` at ACL), the cathode and membrane arrays are reversed.
    Variables that do not exist in the static model (H2/O2 in the membrane,
    saturation in the catalyst layer, temperature) are filled with NaN or left
    empty so the corresponding panels are blank for static curves.
    """
    n_mem = parameters["n_mem"]
    return {
        "v": (
            [sol["C_v_agc"]]
            + list(sol["C_v_agdl"])
            + [sol["C_v_acl"], sol["C_v_ccl"]]
            + list(sol["C_v_cgdl"][::-1])
            + [sol["C_v_cgc"]]
        ),
        "O2": (
            [np.nan] * n_mem
            + [sol["C_O2_ccl"]]
            + list(sol["C_O2_cgdl"][::-1])
            + [sol["C_O2_cgc"]]
        ),
        "H2": (
            [sol["C_H2_agc"]]
            + list(sol["C_H2_agdl"])
            + [sol["C_H2_acl"]]
            + [np.nan] * n_mem
        ),
        "saturation": (
            list(sol["s_agdl"])
            + [np.nan, np.nan]
            + list(sol["s_cgdl"][::-1])
        ),
        "lambda": (
            [sol["lambda_acl"]]
            + list(sol["lambda_mem"][::-1])
            + [sol["lambda_ccl"]]
        ),
        "T": [],
    }


def build_profile_dyn(source, t_index=-1, *, cond_key=None, label=None,
                      fig=None, axes=None,
                      color=None, linestyle=None, marker=None,
                      linewidth=1.6, markersize=3, Tfc=None):
    """Plot the 1D through-plane profile of a dynamic-model snapshot.

    Pass `fig, axes` (returned by an earlier call) to overlay another condition
    on the same figure. Style follows `cond_key` via `get_plot_properties`
    unless `color`/`linestyle`/`marker` are passed explicitly.
    """
    if Tfc is None and hasattr(source, "operating_inputs"):
        Tfc = source.operating_inputs.get("Tfc")
    fig, axes = _setup_axes(fig, axes, Tfc=Tfc)
    color, linestyle, marker = _resolve_style(cond_key, color, linestyle, marker)
    _plot_panels(axes, extract_profile_dyn(source, t_index),
                 label or cond_key,
                 color, linestyle, marker, linewidth, markersize)
    return fig, axes


def build_profile_stat(sol, *, cond_key=None, label=None,
                       fig=None, axes=None,
                       color=None, linestyle=None, marker=None,
                       linewidth=1.6, markersize=3, Tfc=None):
    """Plot the 1D through-plane profile of a static-model `solve()` result.

    Same overlay/styling contract as `build_profile_dyn`.
    """
    fig, axes = _setup_axes(fig, axes, Tfc=Tfc)
    color, linestyle, marker = _resolve_style(cond_key, color, linestyle, marker)
    _plot_panels(axes, extract_profile_stat(sol),
                 label or cond_key,
                 color, linestyle, marker, linewidth, markersize)
    return fig, axes


def build_profile_figure(solution, model, t_index=-1):
    """Legacy wrapper used by the GUI's "Spatial profile" tab."""
    fig, axes = build_profile_dyn(model, t_index=t_index)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def display(solution, model):
    build_profile_figure(solution, model)
    plt.show()
