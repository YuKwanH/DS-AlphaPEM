"""Section 3: result tabs.

Renders simulation output. Two top-level shapes are supported:
  * transient: a populated PEMFC/PEMFC_dyn model + scipy OdeResult.
  * polar    : a dict {"i_A_m2": array, "Ucell_V": array} from PEMFC_stat.
"""

import ast as _ast

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from modules.display import build_profile_figure
from gui import style as _style
from data.export import export_experiment_data


# Number of cells in the LEV-200 stack (used to convert the experimental
# stack voltage VFC to per-cell voltage for the comparison overlay). Kept
# in sync with model.coefficients.n_cell.
N_CELL_STACK = 22


@st.cache_data(show_spinner=False)
def _exp_polarization():
    """Load Polar_curves.xlsx once per session (cached)."""
    return export_experiment_data("pola")


@st.cache_data(show_spinner=False)
def _exp_hfr():
    """Load HFR.xlsx once per session (cached)."""
    return export_experiment_data("hfr")


def _parse_hfr_cell(cell):
    """HFR.xlsx stores each row as the string ``(R_mOhm, I_load_A)`` — pull
    out the real resistance (first tuple element)."""
    if isinstance(cell, (int, float)):
        return float(cell)
    try:
        val = _ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return float("nan")
    return float(val[0]) if isinstance(val, tuple) else float(val)


def _get_xaxis_max(fallback=90.0):
    """Read the current display cap on the time axis from session state.

    Kept as a helper so every panel that draws vs time uses the same
    source of truth and picks up user edits without any threading of
    arguments.
    """
    try:
        v = float(st.session_state.get("xaxis_max_s", fallback))
    except (TypeError, ValueError):
        v = fallback
    return max(1.0, v)


def _apply_time_xlim(ax, t_data=None):
    """Clamp the x-axis of ``ax`` to (0, xaxis_max).

    Uses the user-picked cap from session state; if a data array is
    provided, the effective upper bound is ``min(cap, t_data[-1])`` so
    that a short sim isn't padded with empty space.
    """
    xmax = _get_xaxis_max()
    if t_data is not None:
        try:
            arr = np.asarray(t_data)
            if arr.size and np.isfinite(arr[-1]):
                xmax = min(xmax, float(arr[-1]))
        except Exception:
            pass
    if xmax > 0:
        ax.set_xlim(0.0, xmax)


def _apply_robust_ylim(ax, *_ignored, pad_frac=0.10, k_sigma=4.0):
    """Set y-limits robustly from the traces drawn on ``ax``.

    Call AFTER plotting and AFTER ``_apply_time_xlim`` -- the helper reads
    the axes' current x-window and only considers samples inside it.

    Why not matplotlib's default autoscale: unconverged solver samples
    (startup transients, or a diverging 0D-benchmark tail) reach values
    like Ucell = -2.5 V that stretch the axis and flatten the real signal.
    Why not plain percentiles: a diverging segment can easily exceed 1 %
    of the samples, so p1/p99 still include it.

    Approach -- per-trace sigma clipping:
      1. mask each Line2D's data to the visible x-window, finite only;
      2. estimate a robust center (median) and spread (half the p16-p84
         band, ~1 sigma for unimodal data; for square waves both levels
         stay inside ~1 sigma so nothing legitimate is clipped);
      3. reject samples farther than ``k_sigma`` * spread from the median
         (diverging tails, startup spikes);
      4. y-limits = union of every trace's surviving min/max, padded.

    Extra positional args are accepted and ignored for backwards
    compatibility with the previous (series-based) signature.
    """
    xmin, xmax = ax.get_xlim()
    lo, hi = np.inf, -np.inf

    for line in ax.get_lines():
        try:
            x = np.asarray(line.get_xdata(), dtype=float).ravel()
            y = np.asarray(line.get_ydata(), dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        n = min(x.size, y.size)
        if n == 0:
            continue
        x, y = x[:n], y[:n]
        m = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
        yw = y[m]
        if yw.size == 0:
            continue
        if yw.size < 8:
            lo = min(lo, float(yw.min())); hi = max(hi, float(yw.max()))
            continue
        med = float(np.median(yw))
        p16, p84 = np.percentile(yw, [16.0, 84.0])
        sigma = max((p84 - p16) / 2.0, abs(med) * 1e-6, 1e-12)
        keep = yw[np.abs(yw - med) <= k_sigma * sigma]
        if keep.size == 0:
            keep = yw
        lo = min(lo, float(keep.min())); hi = max(hi, float(keep.max()))

    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if hi <= lo:                       # flat signal -- pad around the value
        span = max(abs(hi), 1.0) * 0.05
        lo, hi = lo - span, hi + span
    pad = (hi - lo) * pad_frac
    ax.set_ylim(lo - pad, hi + pad)


def _rmem_total_series(model):
    """Return the total per-cell membrane resistance vs time (Ω·m²).

    ``model.echem_traj["Rmem"]`` stores a *per-node* array at each time
    step, so we sum along the node axis. Handles the ragged / cast-to-
    object case that can happen if the sim ended mid-integration.
    """
    raw = model.echem_traj.get("Rmem", None)
    if raw is None or len(raw) == 0:
        return None
    try:
        arr = np.asarray(raw, dtype=float)
    except Exception:
        # Ragged shapes — fall back to per-timestep sum
        return np.asarray([np.sum(np.asarray(x, dtype=float)) for x in raw])
    if arr.ndim == 2:
        return arr.sum(axis=1)
    return arr


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


def render(state, on_run=None):
    st.markdown("#### Simulation")

    # Run + Hold on the same row at the top of the section. Run is the
    # primary action (navy), Hold is secondary and only appears after a
    # successful run.
    result = state.get("last_result")
    has_result = result is not None and result["status"].get("success")

    # Three-column header: [Run] [x-axis window] [Hold]. Column ratios
    # give the Run button ~45 % of the header instead of the old ~78 %.
    # The middle column hosts a compact number-input that caps the time
    # axis on every panel that plots vs t.
    #
    # Layout note: the number_input's label is COLLAPSED so it does not push
    # the input widget down below the button baseline. The label text is
    # placed inline beside the input via an inner 2-column split, so the
    # user still sees "x-axis (s)" without breaking the row alignment.
    run_col, xaxis_col, hold_col = st.columns([1.8, 1.6, 1], gap="small")
    with run_col:
        if on_run is not None:
            st.button("▶ Run simulation", type="primary",
                      use_container_width=True,
                      on_click=on_run, key="run_button_section")
    with xaxis_col:
        lbl_col, num_col = st.columns([1, 2], gap="small")
        # Inline label that sits at the same vertical position as the
        # buttons (line-height matches the ~38 px button/input height).
        lbl_col.markdown(
            '<div class="xaxis-inline-label">x-axis (s)</div>',
            unsafe_allow_html=True,
        )
        # Default 90 s — long enough to show a few AST cycles / dwell steps
        # by default without overwhelming plots when a much longer sim runs.
        state["xaxis_max_s"] = float(num_col.number_input(
            "x-axis (s)",
            value=float(state.get("xaxis_max_s", 90.0)),
            min_value=1.0, step=10.0, format="%.1f",
            key="results_xaxis_max",
            label_visibility="collapsed",
            help="Upper limit on the time axis for every panel that plots vs "
                 "time (Cell performance, Manifolds, Water content, "
                 "Degradation, Custom). Only affects display — the underlying "
                 "trajectory is unchanged.",
        ))
    with hold_col:
        # Always render the Hold button so all three header controls sit in
        # a proper row. When there is no successful result yet, the button
        # renders disabled with an explanatory tooltip instead of being
        # hidden — hiding it broke the row alignment on a fresh session.
        hold_disabled = not has_result
        hold_help = (
            "Pin this result; the NEXT simulation will be plotted on top of "
            "it for direct comparison. Only one held slot — pressing Hold "
            "again replaces the previous pin."
            if not hold_disabled else
            "Run a simulation first — Hold can only pin a completed result."
        )
        if st.button("📌 Hold", key="hold_button", use_container_width=True,
                     disabled=hold_disabled, help=hold_help) and has_result:
            state["held_result"] = _snapshot_for_hold(result)

    # 0D benchmark toggle. When ticked, the NEXT Run will also execute the
    # lumped-parameter 0D model (PEMFC_0D) and overlay its polarisation
    # curve (in polar mode) or expose its state variables in the Custom
    # tab (in transient mode).
    state["benchmark_0d"] = st.checkbox(
        "Compare with 0D benchmark",
        value=bool(state.get("benchmark_0d", False)),
        key="benchmark_0d_checkbox",
        help="Runs the 0D lumped-parameter PEMFC model with the same "
             "parameters / load profile alongside your main simulation. "
             "Polar mode: the 0D polarisation curve is overlaid directly. "
             "Transient mode: 0D state variables appear in the Custom tab "
             "(prefixed '0D:').",
    )

    if result is None:
        st.info("Configure parameters and options on the left, then click **▶ Run simulation** above.")
        return state

    status = result["status"]
    _status_strip(status)
    # 0D benchmark status strip (a thin caption under the main status).
    bench = result.get("benchmark_0d")
    if bench is not None:
        bst = bench.get("status", {})
        if bst.get("success"):
            tag = (f"n_points: {bst.get('n_points', '–')}"
                   if bst.get("kind") == "polar"
                   else f"n_steps: {bst.get('n_steps', '–')}")
            st.caption(f"0D benchmark · runtime: **{bst.get('runtime_s', 0):.2f} s** · {tag} · ✓ ok")
        else:
            st.warning(f"0D benchmark failed: {bst.get('message', 'unknown error')}")

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

    bench_polar     = (bench["polar"]      if (bench and bench.get("polar")     is not None) else None)
    bench_variables = (bench["variables"]  if (bench and bench.get("variables")) else {})
    bench_echem     = (bench["echem_traj"] if (bench and bench.get("echem_traj")) else {})

    if status["kind"] == "polar":
        _render_polar(result["polar"], held, bench_polar=bench_polar)
    else:
        _render_transient(result["model"], result["solution"], held,
                          bench_variables=bench_variables,
                          bench_echem=bench_echem)

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
    # Prominent notice when the simulation runner auto-switched the ODE
    # method. Per the honesty rule ("silent switching is not allowed on
    # calibration; on simulation it must be user-visible"), we surface this
    # as an st.warning banner — not just as a suffix on the variant label.
    if status.get("fallback_used"):
        requested = status.get("method_requested", "requested")
        st.warning(
            f"⚠ **Solver auto-switched from `{requested}` → `LSODA`.**  "
            f"`{requested}` hit a transient NaN mid-integration; the runner "
            f"re-ran the ODE with LSODA (which tolerates the same intermediate "
            f"NaNs that older SciPy silently survived). The result above is from "
            f"LSODA, not `{requested}`. If you need a strict single-solver run, "
            f"pick a different method or use the calibration path (which never "
            f"switches).",
            icon="🔀",
        )


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

# Visual style for the 0D benchmark overlay. Dashed green — distinct from the
# held/grey overlay so users can tell a benchmark trace from a pinned one.
_BENCH_KW = dict(linestyle="--", linewidth=1.4, alpha=0.85, color="#009E73")


def _overlay_bench(ax, bench_t, bench_y, label="0D benchmark"):
    """Plot a 0D-benchmark trace on ``ax`` if both arrays are non-empty.

    Returns True if anything was drawn (so the caller can add a legend).
    """
    t = np.asarray(bench_t)
    y = np.asarray(bench_y)
    if t.size == 0 or y.size == 0:
        return False
    n = min(len(t), len(y))
    ax.plot(t[:n], y[:n], label=label, **_BENCH_KW)
    return True


def _bench_time(bench_variables, bench_echem):
    """Pick a time axis from the 0D bundle (echem first, then variables)."""
    if bench_echem and "t" in bench_echem:
        return bench_echem["t"]
    if bench_variables and "t" in bench_variables:
        return bench_variables["t"]
    return []


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


def _render_polar(polar, held=None, bench_polar=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    i_A_cm2 = polar["i_A_m2"] / 1e4
    ax.plot(i_A_cm2, polar["Ucell_V"], marker="o", linewidth=1.5, markersize=4,
            label="1D static")
    if bench_polar is not None and len(bench_polar.get("i_A_m2", [])):
        b_i = np.asarray(bench_polar["i_A_m2"]) / 1e4
        b_u = np.asarray(bench_polar["Ucell_V"])
        ax.plot(b_i, b_u, marker="s", linewidth=1.4, markersize=4,
                color=_style.PALETTE[2], label="0D benchmark")
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


def _render_transient(model, solution, held=None,
                      bench_variables=None, bench_echem=None):
    tabs = st.tabs(["Cell performance", "Spatial profile", "Manifolds",
                    "Water content", "Degradation", "Custom"])

    bench_variables = bench_variables or {}
    bench_echem     = bench_echem     or {}
    with tabs[0]:
        _tab_cell_performance(model, held,
                              bench_variables=bench_variables,
                              bench_echem=bench_echem)
    with tabs[1]:
        _tab_spatial(model, solution)  # snapshot overlay not meaningful
    with tabs[2]:
        _tab_manifolds(model, held)
    with tabs[3]:
        _tab_water(model, held)
    with tabs[4]:
        _tab_degradation(model, held,
                         bench_variables=bench_variables,
                         bench_echem=bench_echem)
    with tabs[5]:
        # The Custom tab also exposes every 0D variable individually, prefixed
        # "0D:", for plotting any quantity not covered by the named tabs.
        _tab_custom(model, held,
                    bench_variables=bench_variables,
                    bench_echem=bench_echem)


def _tab_cell_performance(model, held=None,
                          bench_variables=None, bench_echem=None):
    """Cell performance tab.

    Row 1 (always): [Load current i(t)] [Cell voltage Ucell(t)] [Membrane
    resistance R_mem(t)] — all vs time.

    Below: an optional "Compare with experimental data" checkbox that
    reveals a condition selector and two extra panels rendering the sim
    curves against selected Polar_curves / HFR data at a matching
    operating condition.
    """
    bench_variables = bench_variables or {}
    bench_echem     = bench_echem     or {}
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No time-domain output recorded.")
        return
    Ucell = np.asarray(model.echem_traj.get("Ucell", []))
    i_fc  = np.asarray(model.echem_traj.get("i_fc",  []))
    Rmem  = _rmem_total_series(model)
    bt    = _bench_time(bench_variables, bench_echem)

    # ---- Row 1: three time-series panels ---------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))

    if i_fc.size:
        ax[0].plot(t[: len(i_fc)], i_fc, label="1D")
    drew_h0 = _overlay_held(ax[0], held, "i_fc", source="echem_traj")
    drew_b0 = _overlay_bench(ax[0], bt, bench_echem.get("i_fc", []))
    if drew_h0 or drew_b0:
        ax[0].legend(fontsize=7, loc="best")
    ax[0].set_xlabel(axis_label("t"))
    ax[0].set_ylabel(axis_label("i_fc"))
    ax[0].set_title("Load current")
    ax[0].grid(True, alpha=0.3)

    if Ucell.size:
        ax[1].plot(t[: len(Ucell)], Ucell, color=_style.PALETTE[1], label="1D")
    drew_h1 = _overlay_held(ax[1], held, "Ucell", source="echem_traj")
    drew_b1 = _overlay_bench(ax[1], bt, bench_echem.get("Ucell", []))
    if drew_h1 or drew_b1:
        ax[1].legend(fontsize=7, loc="best")
    ax[1].set_xlabel(axis_label("t"))
    ax[1].set_ylabel(axis_label("Ucell"))
    ax[1].set_title("Cell voltage")
    ax[1].grid(True, alpha=0.3)

    if Rmem is not None and len(Rmem):
        ax[2].plot(t[: len(Rmem)], np.asarray(Rmem),
                   color=_style.PALETTE[2], label="1D")
        ax[2].legend(fontsize=7, loc="best")
    ax[2].set_xlabel(axis_label("t"))
    ax[2].set_ylabel(r"$R_{mem}$  ($\Omega \cdot m^2$)")
    ax[2].set_title("Membrane resistance")
    ax[2].grid(True, alpha=0.3)

    # Apply the shared x-axis-length cap to every time-series panel.
    for _ax in ax:
        _apply_time_xlim(_ax, t)

    # Robust y-limits: ignore startup-transient outliers (e.g. an
    # unconverged Ucell dip in the first solver steps) that otherwise
    # stretch the autoscale and flatten the real signal.
    _apply_robust_ylim(ax[0], i_fc, bench_echem.get("i_fc"))
    _apply_robust_ylim(ax[1], Ucell, bench_echem.get("Ucell"))
    _apply_robust_ylim(ax[2], Rmem)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # ---- Optional overlay panels ----------------------------------------
    _render_experimental_overlay(model, i_fc, Ucell, Rmem)


def _render_experimental_overlay(model, i_fc, Ucell, Rmem):
    """Below the time-series row: opt-in comparison against Polar_curves and
    HFR at a user-picked experimental condition."""
    st.divider()
    compare = st.checkbox(
        "Compare with experimental data",
        value=st.session_state.get("cellperf_compare", False),
        key="cellperf_compare",
        help="Overlay Polar_curves.xlsx on the cell-voltage panel and "
             "HFR.xlsx on the membrane-resistance panel, at the operating "
             "condition selected below.",
    )
    if not compare:
        return

    # Load both datasets (cached; no reload after the first click).
    try:
        pola = _exp_polarization()
        hfr  = _exp_hfr()
    except Exception as exc:
        st.error(f"Could not load experimental data: {exc}")
        return

    # Available conditions are the union of pola + hfr keys (both files use
    # the same T{deg}_P{P}_HRC{RH} scheme). The auto-pick tries to match
    # the current simulation's operating conditions.
    conds = sorted(set(pola.keys()) | set(hfr.keys()))
    if not conds:
        st.warning("No experimental conditions available.")
        return

    default_cond = _autopick_condition(conds, model)
    sel_col_1, sel_col_2 = st.columns(2)
    ucell_cond = sel_col_1.selectbox(
        "Condition — Polar_curves overlay (Cell voltage panel)",
        options=conds,
        index=conds.index(default_cond) if default_cond in conds else 0,
        key="cellperf_pola_cond",
        help="Which measured operating point to overlay on the Ucell vs i "
             "comparison. Auto-selected to match the sim's Tfc / Pc / Phi_c "
             "where possible.",
    )
    hfr_conds = sorted(hfr.keys())
    rmem_default = default_cond if default_cond in hfr_conds else (hfr_conds[0] if hfr_conds else None)
    rmem_cond = sel_col_2.selectbox(
        "Condition — HFR overlay (Membrane resistance panel)",
        options=hfr_conds,
        index=hfr_conds.index(rmem_default) if rmem_default in hfr_conds else 0,
        key="cellperf_hfr_cond",
        help="Which measured HFR point to overlay on the Rmem vs i "
             "comparison.",
    ) if hfr_conds else None

    # Extract simulation curves in the (i, y) parametric form.
    Aact = float(model.parameters.get("Aact", 31e-4))
    n = min(len(i_fc), len(Ucell)) if (i_fc.size and Ucell.size) else 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.3))

    # -- Cell voltage vs current density with polar overlay ----------------
    ax_u = axes[0]
    if n:
        # Sort by current density so the parametric curve is monotonic
        # in the sweep sense (helps a lot with AST / cycling profiles).
        order = np.argsort(i_fc[:n])
        ax_u.plot(i_fc[:n][order], Ucell[:n][order],
                  color=_style.PALETTE[1], linewidth=1.5, label="Simulation")
    if ucell_cond and ucell_cond in pola:
        df = pola[ucell_cond].dropna(subset=["I_LOAD", "VFC"]).sort_values("I_LOAD")
        i_exp = df["I_LOAD"].to_numpy(dtype=float) / Aact         # A -> A/m^2
        v_exp = df["VFC"].to_numpy(dtype=float) / N_CELL_STACK    # stack V -> per cell
        ax_u.plot(i_exp, v_exp, marker="o", markersize=5, linewidth=0,
                  color=_style.PALETTE[3], label=f"Exp — {ucell_cond}")
    ax_u.set_xlabel("Current density (A/m$^2$)")
    ax_u.set_ylabel("Cell voltage (V)")
    ax_u.set_title("Cell voltage — sim vs Polar_curves")
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=7, loc="best")

    # -- Membrane resistance vs current density with HFR overlay ------------
    ax_r = axes[1]
    if Rmem is not None and len(Rmem) and i_fc.size:
        m = min(len(i_fc), len(Rmem))
        order = np.argsort(i_fc[:m])
        ax_r.plot(i_fc[:m][order], np.asarray(Rmem)[:m][order],
                  color=_style.PALETTE[2], linewidth=1.5, label="Simulation")
    if rmem_cond and rmem_cond in hfr:
        df = hfr[rmem_cond].copy()
        df["R_mOhm"] = df["R"].apply(_parse_hfr_cell)
        df = df.dropna(subset=["I_LOAD", "R_mOhm"]).sort_values("I_LOAD")
        i_exp = df["I_LOAD"].to_numpy(dtype=float) / Aact          # A -> A/m^2
        # Experimental HFR is a stack-level absolute resistance in mΩ.
        # Convert to a per-cell area-specific resistance in Ω·m² so it
        # sits on the same axis as sum(Rmem):
        #   R_absolute_Ω = R_mOhm × 1e-3
        #   R_per_cell_Ω  = R_absolute_Ω / N_CELL_STACK
        #   R_area_Ω·m²  = R_per_cell_Ω × Aact
        r_area = df["R_mOhm"].to_numpy(dtype=float) * 1e-3 / N_CELL_STACK * Aact
        ax_r.plot(i_exp, r_area, marker="s", markersize=5, linewidth=0,
                  color=_style.PALETTE[3], label=f"Exp — {rmem_cond}")
    ax_r.set_xlabel("Current density (A/m$^2$)")
    ax_r.set_ylabel(r"$R_{mem}$  ($\Omega \cdot m^2$)")
    ax_r.set_title("Membrane resistance — sim vs HFR")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=7, loc="best")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Experimental HFR is in mΩ (stack) and has been converted to "
        f"per-cell area-specific resistance using N_CELL={N_CELL_STACK} and "
        "Aact from the sim parameters. It reflects the *total* HF-observable "
        "ohmic loss, so it is generally larger than the model's `sum(Rmem)` "
        "alone (which excludes catalyst-layer + contact contributions)."
    )


def _autopick_condition(conds, model):
    """Pick the experimental condition closest to the sim's operating point.

    Falls back to the first condition if the sim's op inputs are missing.
    Scoring is a simple weighted L1 distance on (T_C, P_bar, RHC_pct).
    """
    op = getattr(model, "operating_inputs", {}) or {}
    try:
        T_C   = float(op.get("Tfc", 353.15)) - 273.15
        P_bar = float(op.get("Pc_des", 1.8e5)) / 1e5
        RHC   = float(op.get("Phi_c_des", 0.5)) * 100.0
    except Exception:
        return conds[0]
    import re
    rx = re.compile(r"^T(\d+)_P(\d+)_HRC(\d+)$")
    best, best_score = conds[0], float("inf")
    for c in conds:
        m = rx.match(c)
        if not m:
            continue
        Tc, Pc, RHc = (int(g) for g in m.groups())
        # Pc encoded as (bar × 100 - 1000) — reverse: bar = (Pc + 1000) / 100
        Pc_bar = (Pc + 1000) / 100.0
        score = abs(Tc - T_C) + 5 * abs(Pc_bar - P_bar) + 0.2 * abs(RHc - RHC)
        if score < best_score:
            best, best_score = c, score
    return best


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
    for _ax in axes.flatten():
        _apply_time_xlim(_ax, t)
        _apply_robust_ylim(_ax)
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
        _apply_time_xlim(ax_i, t)
        _apply_robust_ylim(ax_i, y)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    mem_keys = sorted(
        (k for k in model.variables if k.startswith("lambda_mem_")),
        key=lambda s: int(s.rsplit("_", 1)[1]) if s.rsplit("_", 1)[1].isdigit() else 0,
    )
    if mem_keys:
        # Clip the heatmap's time axis so its extent lines up with the
        # user-picked x-axis window (extent still shows every column of
        # data — the axis limit is what controls the visible range).
        mat = np.array([model.variables[k] for k in mem_keys])
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        im = ax2.imshow(mat, aspect="auto", origin="lower",
                        extent=[float(t[0]), float(t[-1]), 1, len(mem_keys)])
        ax2.set_xlabel(axis_label("t"))
        ax2.set_ylabel("Membrane node")
        ax2.set_title("Membrane water content $\\lambda$ (-)")
        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label(r"$\lambda$ (-)")
        _apply_time_xlim(ax2, t)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)


def _tab_degradation(model, held=None,
                     bench_variables=None, bench_echem=None):
    bench_variables = bench_variables or {}
    bench_echem     = bench_echem     or {}
    t = np.asarray(model.variables.get("t", []))
    if t.size == 0:
        st.info("No degradation data recorded.")
        return
    bt = _bench_time(bench_variables, bench_echem)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    delta = np.asarray(model.variables.get("delta_mem", []))
    if delta.size:
        ax[0].plot(t[: len(delta)], delta, linewidth=1.4, label="1D")
    drew_h_d = _overlay_held(ax[0], held, "delta_mem")
    # 0D tracks total membrane thickness as ``Hmem`` — same physical quantity
    # as 1D ``delta_mem`` (both are absolute thickness in metres), so overlay
    # them on the same axes for a direct comparison.
    drew_b_d = _overlay_bench(ax[0], bt, bench_variables.get("Hmem", []))
    if drew_h_d or drew_b_d:
        ax[0].legend(fontsize=7, loc="best")
    ax[0].set_xlabel(axis_label("t"))
    ax[0].set_ylabel(axis_label("delta_mem"))
    ax[0].set_title("Membrane thinning")
    ax[0].grid(True, alpha=0.3)

    s_n = np.asarray(model.echem_traj.get("S_N", []))
    if s_n.size:
        ax[1].plot(t[: len(s_n)], s_n, linewidth=1.4, label="1D")
    drew_h_s = _overlay_held(ax[1], held, "S_N", source="echem_traj")
    drew_b_s = _overlay_bench(ax[1], bt, bench_echem.get("S_N", []))
    if drew_h_s or drew_b_s:
        ax[1].legend(fontsize=7, loc="best")
    ax[1].set_xlabel(axis_label("t"))
    ax[1].set_ylabel(axis_label("S_N"))
    ax[1].set_title("Pt active surface")
    ax[1].grid(True, alpha=0.3)
    for _ax in ax:
        _apply_time_xlim(_ax, t)
    # Robust y-limits so a stray outlier sample doesn't stretch the scale.
    _apply_robust_ylim(ax[0], delta, bench_variables.get("Hmem"))
    _apply_robust_ylim(ax[1], s_n, bench_echem.get("S_N"))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _tab_custom(model, held=None, bench_variables=None, bench_echem=None):
    bench_variables = bench_variables or {}
    bench_echem     = bench_echem or {}

    # Build the picker options: 1D state variables first, then 0D ones
    # (prefixed "0D:") so they sort cleanly together at the bottom.
    options_1d = sorted(k for k in model.variables.keys() if k != "t")
    options_0d_vars  = sorted(f"0D: {k}" for k in bench_variables.keys() if k != "t")
    options_0d_echem = sorted(f"0D: {k}" for k in bench_echem.keys()    if k != "t")
    options = options_1d + options_0d_vars + options_0d_echem
    if not options:
        st.info("No variables recorded.")
        return
    if options_0d_vars or options_0d_echem:
        st.caption("0D-prefixed variables come from the lumped-parameter benchmark.")

    picks = st.multiselect("Variables", options=options,
                           default=options[:1], key="custom_picks")
    if not picks:
        return

    t_1d = np.asarray(model.variables.get("t", []))
    t_0d = np.asarray(bench_variables.get("t", []))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    units = set()
    for name in picks:
        if name.startswith("0D: "):
            key = name[4:]
            # Look in 0D variables first, then echem_traj.
            y = np.asarray(bench_variables.get(key,
                            bench_echem.get(key, [])))
            t_use = t_0d
            label_unit, unit = lookup_unit(key)
        else:
            y = np.asarray(model.variables[name])
            t_use = t_1d
            label_unit, unit = lookup_unit(name)
        legend = f"{name} [{unit}]" if unit else name
        if y.size and t_use.size:
            n = min(len(t_use), len(y))
            ax.plot(t_use[:n], y[:n], label=legend, linewidth=1.2)
        units.add(unit)
        # Hold overlay only for 1D vars (the held snapshot stores 1D data).
        if not name.startswith("0D: "):
            _overlay_held(ax, held, name)

    ax.set_xlabel(axis_label("t"))
    if len(units) == 1:
        only_unit = next(iter(units))
        ax.set_ylabel(f"value ({only_unit})" if only_unit else "value")
    else:
        ax.set_ylabel("value (mixed units — see legend)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    # The Custom tab's x axis is always time; clamp it to the shared window.
    _apply_time_xlim(ax, t_1d)
    _apply_robust_ylim(ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
