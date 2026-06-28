"""Release-readiness verification.

Simulates the experience of a user who just cloned the repository and
ran `pip install -r requirements.txt`. Exercises every entry point and
every code path that matters for a fresh install. Prints a single
GO / NO-GO summary at the end.
"""
import os, sys, time, json, importlib, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

REPORT = []
def _check(label, ok, detail=""):
    REPORT.append((label, ok, detail))
    tag = "PASS" if ok else "FAIL"
    bar = " " * max(0, 50 - len(label))
    print(f"  [{tag}] {label}{bar}{detail}")


# ---------------------------------------------------------------------------
# Block 1 — dependencies declared and installed
# ---------------------------------------------------------------------------
print("=" * 72)
print(" BLOCK 1 — Dependencies declared and installed")
print("=" * 72)

# Read requirements.txt
with open("requirements.txt") as f:
    declared = [ln.strip().split(">=")[0].split("==")[0]
                for ln in f if ln.strip() and not ln.strip().startswith("#")]

REQUIRED_RUNTIME = ["streamlit", "numpy", "scipy", "pandas", "matplotlib",
                    "openpyxl", "optuna"]
for pkg in REQUIRED_RUNTIME:
    _check(f"{pkg} declared in requirements.txt", pkg in declared)

for pkg in REQUIRED_RUNTIME:
    try:
        importlib.import_module(pkg)
        ok = True
    except ImportError as exc:
        ok = False
    _check(f"{pkg} importable", ok)


# ---------------------------------------------------------------------------
# Block 2 — Every production .py file imports cleanly
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 2 — Every production module imports cleanly")
print("=" * 72)

import_targets = [
    "config.initialize", "config.settings",
    "data.export",
    "model.model", "model.coefficients", "model.inst_values",
    "model.kinetic_eq", "model.state_eq", "model.static",
    "modules.signals", "modules.tests", "modules.display",
    "modules.nan_tracker",
    "gui.app", "gui.calibration", "gui.calib_backend",
    "gui.runner", "gui.parameters", "gui.options", "gui.results",
    "gui.save", "gui.style", "gui.profiles",
]
for mod in import_targets:
    try:
        importlib.import_module(mod)
        ok = True; detail = ""
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    _check(f"import {mod}", ok, detail)


# ---------------------------------------------------------------------------
# Block 3 — Legacy import paths are GONE (no model.dualscale, no model.dynamic)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 3 — Legacy import paths cleaned up")
print("=" * 72)
for stale in ("model.dualscale", "model.dynamic"):
    try:
        importlib.import_module(stale)
        ok = False; detail = "module still importable — was supposed to be removed"
    except ImportError:
        ok = True; detail = ""
    _check(f"`import {stale}` correctly absent", ok, detail)


# ---------------------------------------------------------------------------
# Block 4 — Data files referenced from code actually exist
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 4 — Required data files present")
print("=" * 72)
for f in ("data/Polar_curves.xlsx", "data/HFR.xlsx", "data/eis.xlsx",
          "data/export.py"):
    _check(f"{f} exists", os.path.exists(f), f"({os.path.getsize(f):,} bytes)" if os.path.exists(f) else "MISSING")


# ---------------------------------------------------------------------------
# Block 5 — Data loaders return non-empty datasets
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 5 — Experimental data loaders")
print("=" * 72)
from data.export import export_experiment_data
for key, expected_min in [("pola", 5), ("hfr", 5), ("eis", 10)]:
    try:
        d = export_experiment_data(key)
        ok = isinstance(d, dict) and len(d) >= expected_min
        _check(f"export_experiment_data({key!r:6s})", ok, f"{len(d)} conditions")
    except Exception as exc:
        _check(f"export_experiment_data({key!r:6s})", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Block 6 — main.py runs end-to-end
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 6 — main.py runs end-to-end (short transient)")
print("=" * 72)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
    # Patch shorten the simulation so the audit is fast.
    import main
    main.T_SPAN = (0.0, 3.0)
    main.MAX_STEP = 0.2
    main.SHOW_PLOTS = False
    t0 = time.perf_counter()
    main.main()
    _check("main.py runs to completion", True, f"({time.perf_counter()-t0:.1f}s)")
except SystemExit:
    _check("main.py runs to completion", True, "(SystemExit)")
except Exception as exc:
    traceback.print_exc()
    _check("main.py runs to completion", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Block 7 — calibration.py runs end-to-end with each optimizer
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 7 — calibration.py runs with each optimizer (4 trials)")
print("=" * 72)
for opt in ("TPE (Optuna)", "Genetic algorithm", "Grid search"):
    try:
        importlib.invalidate_caches()
        # Re-import calibration to pick up the latest baseline state.
        import calibration as calib
        importlib.reload(calib)
        calib.OPTIMIZER  = opt
        calib.N_TRIALS   = 4
        calib.CONDITIONS = ["T50_P300_HRC0"]
        calib.PARAMS_TO_FIT = {"i0_c_ref": (1e-3, 10.0), "e": [3, 4, 5]}
        calib.SHOW_PLOTS = False
        calib.SAVE_RESULT = False
        # Mute the progress bar — it floods the log.
        calib._print_progress = lambda *a, **k: None
        t0 = time.perf_counter()
        calib.main()
        _check(f"calibration.py · {opt}", True, f"({time.perf_counter()-t0:.1f}s)")
    except Exception as exc:
        traceback.print_exc()
        _check(f"calibration.py · {opt}", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Block 8 — GUI panels render without exceptions (mocked Streamlit)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 8 — GUI panels render (mocked Streamlit)")
print("=" * 72)

import streamlit as st
class _C:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __getattr__(self, n): return getattr(st, n)
st.container   = lambda *a, **k: _C()
st.columns     = lambda spec, **k: [_C() for _ in range(len(spec) if hasattr(spec, '__len__') else int(spec))]
st.expander    = lambda *a, **k: _C()
st.tabs        = lambda labels, **k: [_C() for _ in labels]
st.selectbox   = lambda label, options, index=0, **k: list(options)[index]
st.multiselect = lambda label, options, default=None, **k: list(default) if default else []
st.number_input = lambda label, value=0, **k: value
st.text_input  = lambda label, value="", **k: value
st.button      = lambda label, **k: False
st.checkbox    = lambda label, value=False, **k: value
st.pyplot      = lambda *a, **k: None
st.dataframe   = lambda *a, **k: None
st.slider      = lambda label, **k: k.get("value", k.get("max_value", 0))
st.spinner     = lambda *a, **k: _C()
st.progress    = lambda *a, **k: _C()
st.toast       = lambda *a, **k: None
for n in ('markdown', 'divider', 'caption', 'info', 'error', 'success',
          'warning', 'json', 'subheader', 'header', 'title', 'metric',
          'image', 'code', 'rerun'):
    setattr(st, n, lambda *a, **k: None)
st.set_page_config = lambda *a, **k: None
st.session_state = {}

try:
    from gui import calibration as panel_calibration
    panel_calibration.render(st.session_state, section_height=820)
    _check("Calibration page renders", True,
           f"({len(st.session_state.get('calib', {}).get('params', []))} params loaded)")
except Exception as exc:
    traceback.print_exc()
    _check("Calibration page renders", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Block 9 — Notebooks recovered earlier are valid JSON
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 9 — Recovered notebooks parse as valid JSON")
print("=" * 72)
for nb in [
    "simulation/parameter calibration/dynamic model/pointwise.ipynb",
    "simulation/parameter calibration/dynamic model/polar.ipynb",
    "simulation/parameter calibration/dynamic model/manual_polar.ipynb",
]:
    try:
        with open(nb, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = json.dumps(data)
        ok = (
            "cells" in data
            and "<<<<<<<" not in text
            and ">>>>>>>" not in text
            and "model.dualscale" not in text
            and "model.dynamic" not in text
        )
        _check(f"{os.path.basename(nb)}", ok,
               f"{len(data.get('cells', []))} cells")
    except json.JSONDecodeError as exc:
        _check(f"{os.path.basename(nb)}", False, f"invalid JSON: {exc}")
    except Exception as exc:
        _check(f"{os.path.basename(nb)}", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Block 10 — No stale `model.dualscale` / `model.dynamic` references anywhere
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" BLOCK 10 — No stale legacy-import strings in tracked files")
print("=" * 72)
stale_hits = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in
               ("__pycache__", ".claude", ".git", ".venv", "node_modules",
                ".ipynb_checkpoints", "worktrees")]
    for f in files:
        if not f.endswith((".py", ".ipynb", ".md")):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except UnicodeDecodeError:
            continue
        # Skip release-verification scripts that mention the legacy names.
        if "verify_release" in path:
            continue
        for token in ("from model.dualscale", "from model.dynamic",
                      "import model.dualscale", "import model.dynamic",
                      "model.dualscale.PEMFC", "model.dynamic.PEMFC_dyn"):
            if token in text:
                stale_hits.append((path, token))
_check("no stale 'model.dualscale' / 'model.dynamic' imports",
       not stale_hits,
       f"{len(stale_hits)} hits" if stale_hits else "clean")
for p, tok in stale_hits[:10]:
    print(f"    -> {p}: {tok!r}")


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print()
print("=" * 72)
n_pass = sum(1 for _, ok, _ in REPORT if ok)
n_fail = sum(1 for _, ok, _ in REPORT if not ok)
print(f" SUMMARY: {n_pass}/{len(REPORT)} passed, {n_fail} failed")
print("=" * 72)
if n_fail:
    print("\nFAILING CHECKS:")
    for label, ok, detail in REPORT:
        if not ok:
            print(f"  • {label}  ({detail})")
    sys.exit(1)
else:
    print("\n  *** ALL RELEASE CHECKS PASSED ***\n")
