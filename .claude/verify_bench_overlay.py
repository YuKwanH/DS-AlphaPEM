"""Confirm the 0D benchmark overlay reaches Cell performance + Degradation tabs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from copy import deepcopy
import numpy as np

from config.initialize import parameters as P, operating_inputs as O
from gui import profiles, style as _style
from gui.runner import run as run_simulation, run_0d_companion
from gui.results import _tab_cell_performance, _tab_degradation, _tab_water

_style.apply_matplotlib()
OUTDIR = os.path.join(os.path.dirname(__file__), "verify_artifacts")
os.makedirs(OUTDIR, exist_ok=True)

# Run 1D (no aux) + 0D companion with the new default profile (AST, 6 s period).
op = deepcopy({k: v for k, v in O.items() if k != "current_density"})
profile_func = profiles.ast_cycling(6.0, 1.0, 25.8, 4.0, P["Aact"])

model, sol, status = run_simulation(
    params=deepcopy(P), op_inputs=op, model_variant="Dual-scale",
    profile_func=profile_func, t_span=(0.0, 18.0),
    max_step=0.1, method="BDF", polar_sweep=None, aux_system=False,
)
U1d = np.asarray(model.echem_traj['Ucell'])
print(f"1D: success={status['success']}, Ucell range [{U1d.min():.3f}, {U1d.max():.3f}] V")

bench = run_0d_companion(deepcopy(P), op, profile_func, (0.0, 18.0))
print(f"0D: success={bench['status']['success']}, n_steps={bench['status']['n_steps']}")
bv = bench["variables"]; be = bench["echem_traj"]

# Capture each tab's figure.
_saved = {}
def _pyplot_capture(name):
    def _f(fig, *a, **k):
        fig.savefig(os.path.join(OUTDIR, name), dpi=110, bbox_inches="tight")
        _saved[name] = True
        plt.close(fig)
    return _f

st.info = lambda *a, **k: None
st.error = lambda *a, **k: None

for name, fn, args in [
    ("bench_cell_performance.png", _tab_cell_performance,
        (model, None, bv, be)),
    ("bench_degradation.png", _tab_degradation,
        (model, None, bv, be)),
]:
    st.pyplot = _pyplot_capture(name)
    fn(*args)

print()
print(f"Saved -> {sorted(_saved.keys())}")
