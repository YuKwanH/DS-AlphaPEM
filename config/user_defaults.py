"""User-saved defaults for the Streamlit GUI.

The two "Save as default" buttons in the Simulation page (one next to
"§ 1 Parameters", one next to "§ 2 Options") write the current session
values to ``config/user_defaults.json``. On the next launch, the GUI
loads that file and pre-populates its widgets, so parameter/option
tweaks persist across sessions without touching source code.

The file is safe to delete -- the GUI falls back to the values in
``config/initialize.py`` when no overrides exist.
"""

from __future__ import annotations

import json
from pathlib import Path


_PATH = Path(__file__).parent / "user_defaults.json"


# --- Which keys go into each of the two saved buckets --------------------
_OPTION_KEYS = (
    "model_variant", "aux_system", "profile_kind", "profile_cfg",
    "t_start", "t_end", "max_step", "method",
)


_EMPTY = {"parameters": {}, "op_inputs": {}, "options": {}, "kinetic_consts": {}}


def load() -> dict:
    """Return the user-defaults dict, or an empty structure if none saved."""
    if not _PATH.exists():
        return {k: dict(v) for k, v in _EMPTY.items()}
    try:
        with _PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {k: dict(v) for k, v in _EMPTY.items()}
    return {k: dict(data.get(k, {})) for k in _EMPTY}


def _dump(payload: dict) -> None:
    """Write the merged payload to disk, preserving any keys we don't touch."""
    existing = load()
    for bucket in _EMPTY:
        existing[bucket].update(payload.get(bucket, {}))
    with _PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _jsonable(v):
    """Only save JSON-serialisable values (skip lambdas, arrays, etc.)."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return None       # numpy arrays, callables (current_density lambda), etc.


def save_parameters(params: dict, op_inputs: dict,
                    kinetic_consts: dict | None = None) -> None:
    """Save the current physics parameters, operating inputs, and Pt-surface
    rate constants ("Micro-scale CL" group) as defaults.
    ``current_density`` (a lambda) is skipped -- it's a session-only value."""
    params_ser = {k: _jsonable(v) for k, v in params.items()
                  if _jsonable(v) is not None and not callable(v)}
    op_ser = {k: _jsonable(v) for k, v in op_inputs.items()
              if k != "current_density" and _jsonable(v) is not None
              and not callable(v)}
    kin_ser = {k: _jsonable(v) for k, v in (kinetic_consts or {}).items()
               if _jsonable(v) is not None}
    _dump({"parameters": params_ser, "op_inputs": op_ser,
           "kinetic_consts": kin_ser})


def save_options(state) -> None:
    """Save the current §2 Options selections as defaults."""
    options_ser = {}
    for key in _OPTION_KEYS:
        if key in state:
            v = _jsonable(state[key])
            if v is not None:
                options_ser[key] = v
    _dump({"options": options_ser})


def path_str() -> str:
    """Human-readable path for the toast message."""
    return str(_PATH)
