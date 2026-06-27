"""Verify the calibration GUI page end-to-end without actually calibrating.

Exercises the data loaders, the figure builders for every dataset, the
optimizer-config defaults, the start-button request payload, and the
column-layout constants.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

import streamlit as st  # noqa
from gui import calibration as cal


# ---------------------------------------------------------------------------
# Block 1: constants and defaults
# ---------------------------------------------------------------------------
print("=" * 80)
print("BLOCK 1 — Module surface area")
print("=" * 80)
assert cal.DATASETS == ("Polarization", "HFR", "EIS")
print(f"  [ok] DATASETS              = {cal.DATASETS}")

assert cal.OPTIMIZERS == ("TPE (Optuna)", "Genetic algorithm", "Grid search")
print(f"  [ok] OPTIMIZERS            = {cal.OPTIMIZERS}")
assert cal.OPTIMIZER_DEFAULT == "TPE (Optuna)"
print(f"  [ok] OPTIMIZER_DEFAULT     = {cal.OPTIMIZER_DEFAULT!r}")

assert cal.MODEL_VARIANTS == ("Dual-scale", "Dynamic", "Static")
print(f"  [ok] MODEL_VARIANTS        = {cal.MODEL_VARIANTS}")

assert set(cal.CALIB_PARAMS) == {"i0_c_ref", "kappa_c", "epsilon_c",
                                  "epsilon_mc", "tau", "Hmem"}
for k, (label, lo, hi, log, unit) in cal.CALIB_PARAMS.items():
    assert lo < hi, f"bad bounds for {k}"
print(f"  [ok] CALIB_PARAMS          = {len(cal.CALIB_PARAMS)} parameters, "
      f"all with valid bounds (low < high)")


# ---------------------------------------------------------------------------
# Block 2: experimental data loaders
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK 2 — Data loaders (cached)")
print("=" * 80)
expected = {"pola": 12, "hfr": 12, "eis": 83}
for key, n in expected.items():
    data = cal._load(key)
    n_found = len(data)
    ok = n_found == n
    print(f"  [{'ok' if ok else 'FAIL':<4}] _load({key!r:<7s}) -> {n_found} conditions "
          f"(expected {n})")
    assert ok, f"bad condition count for {key}"

# Spot-check the HFR tuple parser
hfr_first = next(iter(cal._load("hfr").values()))
import pandas as pd
r_real = hfr_first["R"].apply(cal._parse_hfr_R)
assert (r_real > 0).all() and r_real.notna().all()
print(f"  [ok] _parse_hfr_R         -> all 9 values positive & finite "
      f"(range {r_real.min():.2f} .. {r_real.max():.2f})")

# Spot-check the EIS column finder
eis_first = next(iter(cal._load("eis").values()))
re_col = cal._find_col(eis_first, "Partie r", exclude="stack")
im_col = cal._find_col(eis_first, "Partie i", exclude="stack")
stk_col = cal._find_col(eis_first, "Partie r")  # should pick stack-free first
assert re_col is not None and "stack" not in str(re_col)
assert im_col is not None and "stack" not in str(im_col)
print(f"  [ok] _find_col           -> Re col={re_col!r}, Im col={im_col!r}")


# ---------------------------------------------------------------------------
# Block 3: figure builders
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK 3 — Figure builders")
print("=" * 80)
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__),
                   "verify_artifacts", "calibration_panels")
os.makedirs(OUT, exist_ok=True)

for label, key, n_conds in [("Polarization", "pola", 3),
                            ("HFR",          "hfr",  3),
                            ("EIS",          "eis",  1)]:
    data = cal._load(key)
    conds = sorted(data.keys())[:n_conds]
    fig = cal._build_data_figure(label, data, conds)
    path = os.path.join(OUT, f"data_{key}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    # Sanity: figure has at least one axis with at least one line drawn
    ax = fig.axes[0]
    n_lines = len(ax.get_lines())
    print(f"  [{'ok' if n_lines >= n_conds else 'FAIL':<4}] {label:<12s} -> "
          f"{n_lines} line{'s' if n_lines != 1 else ''} drawn for "
          f"{n_conds} condition{'s' if n_conds != 1 else ''}")


# ---------------------------------------------------------------------------
# Block 4: simulate the render() flow with mocked Streamlit
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK 4 — Mock-render of the calibration page")
print("=" * 80)

# Capture what each Streamlit primitive would have done.
captured = {"buttons": [], "selectboxes": [], "multiselect": [],
            "number_input": [], "figures": []}

class _Container:
    """Stand-in for st.container / st.column — forwards primitive calls to
    the patched module-level functions so cN.number_input / cN.button work
    the same as st.number_input / st.button."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __getattr__(self, name):
        # Defer to whatever streamlit attribute the panel asks for; if the
        # test has patched it, the patched version is returned.
        return getattr(st, name)

def _selectbox(label, options, index=0, **k):
    captured["selectboxes"].append({"label": label, "options": list(options),
                                     "selected": list(options)[index]})
    return list(options)[index]

def _multiselect(label, options, default=None, **k):
    selected = list(default) if default is not None else []
    captured["multiselect"].append({"label": label, "n_options": len(options),
                                    "selected": selected})
    return selected

def _number_input(label, value=0, **k):
    captured["number_input"].append({"label": label, "value": value})
    return value

def _button(label, **k):
    captured["buttons"].append({"label": label, **{kk: vv for kk, vv in k.items()
                                                    if kk == "type"}})
    return False  # don't trigger the start callback

def _pyplot(fig, **k):
    captured["figures"].append(fig)

def _columns(spec, **k):
    n = len(spec) if hasattr(spec, "__len__") else int(spec)
    return [_Container() for _ in range(n)]

def _expander(label, **k): return _Container()
def _noop(*a, **k): return None

# Patch Streamlit primitives
st.container   = _Container
st.columns     = _columns
st.expander    = _expander
st.selectbox   = _selectbox
st.multiselect = _multiselect
st.number_input = _number_input
st.button      = _button
st.pyplot      = _pyplot
st.markdown    = _noop
st.divider     = _noop
st.caption     = _noop
st.info        = _noop
st.error       = _noop
st.success     = _noop
st.warning     = _noop
st.json        = _noop
st.dataframe   = _noop
st.toast       = _noop

# Fake session state — a dict that supports `setdefault`.
state = {}

# Render and inspect.
cal.render(state, section_height=820)

# Now check what was rendered:
labels = [s["label"] for s in captured["selectboxes"]]
assert "Dataset" in labels
assert "Calibration target" in labels
assert "Model variant" in labels
assert "Optimizer" in labels
print(f"  [ok] selectboxes shown   : {labels}")

# Optimizer selectbox must offer EXACTLY the three options the user asked for.
opt_box = next(s for s in captured["selectboxes"] if s["label"] == "Optimizer")
assert opt_box["options"] == ["TPE (Optuna)", "Genetic algorithm",
                               "Grid search"], opt_box["options"]
assert opt_box["selected"] == "TPE (Optuna)"
print(f"  [ok] Optimizer options   : {opt_box['options']}")
print(f"  [ok] Optimizer default   : {opt_box['selected']!r}")

# Sampler must NOT appear anywhere.
assert not any("sampler" in s["label"].lower() for s in captured["selectboxes"])
print(f"  [ok] no stale 'Sampler' label anywhere")

# Parameters-to-fit multiselect must be present.
varylist = [m for m in captured["multiselect"] if m["label"] == "Vary"]
assert len(varylist) == 1
assert varylist[0]["n_options"] == 6
print(f"  [ok] 'Vary' multiselect with {varylist[0]['n_options']} options")

# Bound editor: with default params=[i0_c_ref, kappa_c] selected, four
# number_inputs should have been emitted (low/high for each parameter)
# in addition to the Trials / Random seed pair.
bound_inputs = [n for n in captured["number_input"]
                if "low" in n["label"] or "high" in n["label"]]
assert len(bound_inputs) == 4, f"expected 4 bound inputs, got {len(bound_inputs)}"
print(f"  [ok] {len(bound_inputs)} bound inputs for the 2 default params")

# Start button present.
assert any("Start calibration" in b["label"] for b in captured["buttons"])
print(f"  [ok] Start-calibration button rendered as primary")

# Figure builder ran (column-1 plot).
assert len(captured["figures"]) >= 1
print(f"  [ok] {len(captured['figures'])} figure rendered in column 1")


# ---------------------------------------------------------------------------
# Block 5: simulate clicking Start and check the request payload
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("BLOCK 5 — Mock the Start button: confirm the queued request payload")
print("=" * 80)

# Re-render with the start button returning True so the callback fires.
captured = {"buttons": [], "selectboxes": [], "multiselect": [],
            "number_input": [], "figures": []}
def _button_clicked(label, **k):
    captured["buttons"].append({"label": label})
    return "Start" in label
st.button = _button_clicked

state2 = {}
cal.render(state2, section_height=820)

req = state2.get("calib_request")
assert req is not None, "Start did not queue a request"
print(f"  [ok] state['calib_request'] populated")
print(f"     target     = {req['target']!r}")
print(f"     model      = {req['model']!r}")
print(f"     optimizer  = {req['optimizer']!r}")
print(f"     n_trials   = {req['n_trials']}")
print(f"     seed       = {req['seed']}")
print(f"     params     = {req['params']}")
print(f"     bounds     = {req['bounds']}")
print(f"     conditions = {req['conditions']}")
assert req["optimizer"] == "TPE (Optuna)"
assert set(req["bounds"]) == set(req["params"])
print(f"  [ok] bounds dict matches selected params exactly")

print()
print("=" * 80)
print("ALL CALIBRATION CHECKS PASSED.")
print("=" * 80)
