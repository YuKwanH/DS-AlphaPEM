"""End-to-end verification of the GUI's default simulation.

Reproduces the GUI's defaults (Dual-scale + aux=off + Step profile + BDF),
runs the simulation, then exercises every tab in the result panel and the
Save & download panel. Any tab that errors or returns empty figures is
flagged. Renders are written to ``.claude/verify_artifacts/`` so you can
spot-check the plots visually.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from config.initialize import parameters as _PARAMS_DEFAULT
from config.initialize import operating_inputs as _OP_DEFAULT
from gui import profiles, style as _style
from gui.runner import run as run_simulation
from gui.results import (
    _tab_cell_performance, _tab_spatial, _tab_manifolds,
    _tab_water, _tab_degradation, _tab_custom,
    axis_label, lookup_unit, _HELD_KW,
)
from gui.save import _serialize as save_serialize
from modules.display import build_profile_figure

_style.apply_matplotlib()

OUTDIR = os.path.join(os.path.dirname(__file__), "verify_artifacts")
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Run the GUI's default simulation
# ----------------------------------------------------------------------------
params    = deepcopy(_PARAMS_DEFAULT)
op_inputs = deepcopy({k: v for k, v in _OP_DEFAULT.items()
                      if k != "current_density"})
profile_func = profiles.step(
    tstart=0.0, tend=6.0,
    i_low_A_m2=20.0, i_high_A_m2=12000.0,
    tau_switch=1.0, t_switch=3.0,
)

print("=== Default simulation ===")
print(f"  variant=Dual-scale  aux=off  profile=Step  t_span=(0, 20)")
print(f"  i_low=20, i_high=12000 A/m^2, period 0-6 s, tau_switch=1.0, t_switch=3.0")
print()

t0 = time.perf_counter()
model, sol, status = run_simulation(
    params=params, op_inputs=op_inputs,
    model_variant="Dual-scale", profile_func=profile_func,
    t_span=(0.0, 20.0), max_step=0.1, method="BDF",
    polar_sweep=None, aux_system=False,
)
wall = time.perf_counter() - t0

print(f"  resolved class : {type(model).__name__}  (from {type(model).__module__})")
print(f"  variant label  : {status['model_variant']}")
print(f"  success        : {status['success']}    n_states={status['n_states']} "
      f"n_steps={status['n_steps']}    wall={wall:.2f}s")
if status.get("message"):
    print(f"  message        : {status['message']}")
print()
assert status["success"], "Default sim failed — aborting tab verification"


# ----------------------------------------------------------------------------
# 2. Probe data each result tab expects
# ----------------------------------------------------------------------------
variables  = getattr(model, "variables",  {}) or {}
echem_traj = getattr(model, "echem_traj", {}) or {}
fluxes     = getattr(model, "fluxes",     {}) or {}

print("=== Data availability ===")
print(f"  model.variables  : {len(variables):3d} keys")
print(f"  model.echem_traj : {len(echem_traj):3d} keys")
print(f"  model.fluxes     : {len(fluxes):3d} keys")
print(f"  solution.y shape : {sol.y.shape if sol is not None else '—'}")
print(f"  solution.t shape : {sol.t.shape if sol is not None else '—'}")
print()


def _check(label, has_data, expected, *, missing_ok=False):
    tag = "ok  " if has_data else ("ok* " if missing_ok else "MISS")
    print(f"  [{tag}] {label:<32s}  expects: {expected}")
    if not has_data and not missing_ok:
        print(f"          ^^ missing data!")


print("=== Tab-by-tab data probe ===")
_check("Cell performance: Ucell",
       "Ucell" in echem_traj and len(echem_traj["Ucell"]) > 0,
       "echem_traj['Ucell']")
_check("Cell performance: i_fc",
       "i_fc" in echem_traj and len(echem_traj["i_fc"]) > 0,
       "echem_traj['i_fc']")
_check("Cell performance: t",
       "t" in variables and len(variables["t"]) > 0,
       "variables['t']")
_check("Spatial profile: solution",
       sol is not None and getattr(sol, "y", np.zeros((0,0))).shape[1] > 0,
       "solution.y")
manif_keys = ["Phi_asm","Pasm","Phi_csm","Pcsm","Phi_aem","Paem","Phi_cem","Pcem"]
present_manif = [k for k in manif_keys if k in variables and len(variables[k]) > 0]
_check("Manifolds (8 BoP vars)",
       len(present_manif) == 8,
       "Phi_*sm / P*sm  (8 keys — BoP only)",
       missing_ok=True)
print(f"          present: {present_manif or '(none — expected for aux=off)'}")
_check("Water: lambda_acl",
       "lambda_acl" in variables and len(variables["lambda_acl"]) > 0,
       "variables['lambda_acl']")
_check("Water: lambda_ccl",
       "lambda_ccl" in variables and len(variables["lambda_ccl"]) > 0,
       "variables['lambda_ccl']")
mem_keys = sorted([k for k in variables if k.startswith("lambda_mem_")])
_check(f"Water: lambda_mem_* (n={len(mem_keys)})",
       len(mem_keys) >= 5, "lambda_mem_1..N (membrane nodes)")
_check("Degradation: delta_mem",
       "delta_mem" in variables and len(variables["delta_mem"]) > 0,
       "variables['delta_mem']")
_check("Degradation: S_N",
       "S_N" in echem_traj and len(echem_traj["S_N"]) > 0,
       "echem_traj['S_N']")
_check(f"Custom: pickable vars (n={len(variables) - 1})",
       len(variables) >= 5,
       "all variables for the multiselect")
print()


# ----------------------------------------------------------------------------
# 3. Render every tab's figure using the same code paths the GUI calls.
#    We monkey-patch streamlit.pyplot to capture the figure into a PNG instead.
# ----------------------------------------------------------------------------
import streamlit as st

# Use a lightweight in-memory shim so we can call the tab functions outside
# the Streamlit runtime. Anything other than st.pyplot just no-ops.
_captured = {}

class _StreamlitShim:
    def __init__(self, name): self._name = name
    def __call__(self, *a, **k):
        if self._name == "pyplot":
            fig = a[0] if a else k.get("fig")
            path = os.path.join(OUTDIR, _captured["current_tab"])
            fig.savefig(path, dpi=110, bbox_inches="tight")
            _captured.setdefault("saved", []).append(path)
            plt.close(fig)
        return None
    def __getattr__(self, item): return _StreamlitShim(item)
    def __enter__(self): return self
    def __exit__(self, *a): pass

# Replace only what the tab functions touch.
_real_pyplot = st.pyplot
_real_info   = st.info
_real_error  = st.error
_real_caption= st.caption
_real_slider = st.slider
_real_multi  = st.multiselect
st.pyplot     = _StreamlitShim("pyplot")
st.info       = lambda *a, **k: print(f"          [tab info  ] {a}")
st.error      = lambda *a, **k: print(f"          [tab error ] {a}")
st.caption    = lambda *a, **k: None
st.slider     = lambda label, **k: k.get("value", k.get("max_value", 0))
st.multiselect= lambda label, options, **k: list(k.get("default", options[:1]))

print("=== Render each results tab ===")
def _render(label, fname, fn, *args):
    _captured["current_tab"] = fname
    print(f"  [{label}] -> {fname}", end="  ")
    n_before = len(_captured.get("saved", []))
    try:
        fn(*args)
        n_after = len(_captured.get("saved", []))
        nfigs = n_after - n_before
        print(f"({nfigs} figure{'s' if nfigs != 1 else ''})")
    except Exception as exc:
        print(f"FAILED  ({type(exc).__name__}: {exc})")

_render("Cell performance", "tab1_cell_performance.png",
        _tab_cell_performance, model, None)
_render("Spatial profile ", "tab2_spatial.png",
        _tab_spatial,           model, sol)
_render("Manifolds       ", "tab3_manifolds.png",
        _tab_manifolds,         model, None)
_render("Water content   ", "tab4_water.png",
        _tab_water,             model, None)
_render("Degradation     ", "tab5_degradation.png",
        _tab_degradation,       model, None)

# Custom tab needs us to pre-set the multiselect default. Use Ucell-ish picks.
st.multiselect = lambda label, options, **k: ["delta_mem", "lambda_acl"] if "delta_mem" in options else options[:2]
_render("Custom          ", "tab6_custom.png",
        _tab_custom,            model, None, {}, {})

# Restore real streamlit funcs (not strictly needed in this script).
st.pyplot     = _real_pyplot
st.info       = _real_info
st.error      = _real_error
st.caption    = _real_caption
st.slider     = _real_slider
st.multiselect= _real_multi
print()


# ----------------------------------------------------------------------------
# 4. Save panel: serialize to all three formats
# ----------------------------------------------------------------------------
print("=== Save & download panel ===")
res_bundle = {"model": model, "solution": sol, "polar": None, "status": status}
for fmt in ("CSV", "NumPy (.npz)", "Excel (.xlsx)"):
    try:
        payload = save_serialize(res_bundle, fmt)
        ext = {"CSV": "csv", "NumPy (.npz)": "npz", "Excel (.xlsx)": "xlsx"}[fmt]
        path = os.path.join(OUTDIR, f"save_default.{ext}")
        with open(path, "wb") as f:
            f.write(payload)
        print(f"  [ok  ] {fmt:<15s} -> {os.path.basename(path)}  ({len(payload)/1024:.1f} kB)")
    except Exception as exc:
        print(f"  [FAIL] {fmt:<15s} ({type(exc).__name__}: {exc})")
print()

print(f"Artifacts written to: {OUTDIR}")
print("[verify] DONE.")
