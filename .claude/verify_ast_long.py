"""Long-term AST cycling verification — incrementally checkpointed.

Two runs at default mesh:
  * PEMFC AST 600 s sim (100 cycles, ~9 min wall expected)
  * PEMFC_dyn AST 300 s sim (50 cycles, ~10 min wall expected)

Each result is written to verify_ast_long.json immediately after the run
finishes, so a partial completion still preserves what did succeed.
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from copy import deepcopy
from config.initialize import parameters as P, operating_inputs as O
from gui import profiles
from gui.runner import run as run_simulation

p = deepcopy(P); o = deepcopy({k: v for k, v in O.items() if k != "current_density"})
Aact = p["Aact"]
JSON_PATH = os.path.join(os.path.dirname(__file__), "verify_ast_long.json")
records = []


def _save():
    with open(JSON_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def _stats(model):
    u = np.asarray(getattr(model, "echem_traj", {}).get("Ucell", []))
    sn = np.asarray(getattr(model, "echem_traj", {}).get("S_N", []))
    dm = np.asarray(getattr(model, "variables", {}).get("delta_mem", []))
    out = dict(ucell_min=float(u.min()) if u.size else None,
               ucell_max=float(u.max()) if u.size else None,
               ucell_finite=int(np.isfinite(u).sum()))
    if sn.size:
        out["sn_first"] = float(sn[0]); out["sn_last"] = float(sn[-1])
    if dm.size:
        out["delta_mem_first"] = float(dm[0]); out["delta_mem_last"] = float(dm[-1])
    return out


def _do(label, *, variant, aux, t_end):
    pp = deepcopy(P)
    oo = deepcopy({k: v for k, v in O.items() if k != "current_density"})
    profile_func = profiles.ast_cycling(6.0, 1.0, 25.8, 4.0, pp["Aact"])

    print(f"[{time.strftime('%H:%M:%S')}] START  {label}  (t_end={t_end} s)")
    t0 = time.perf_counter()
    try:
        m, s, st = run_simulation(
            params=pp, op_inputs=oo, model_variant=variant,
            profile_func=profile_func, t_span=(0.0, float(t_end)),
            max_step=0.5, method="BDF", polar_sweep=None, aux_system=aux,
        )
        wall = time.perf_counter() - t0
        rec = dict(label=label, success=st["success"], wall_s=wall,
                   runtime_s=st["runtime_s"], n_steps=st["n_steps"],
                   n_states=st["n_states"], message=st["message"],
                   resolved_class=type(m).__name__,
                   variant_label=st["model_variant"])
        rec.update(_stats(m))
    except Exception as exc:
        wall = time.perf_counter() - t0
        rec = dict(label=label, success=False, wall_s=wall,
                   message=f"EXCEPTION {type(exc).__name__}: {exc}")
    records.append(rec); _save()
    tag = "PASS" if rec.get("success") else "FAIL"
    print(f"[{time.strftime('%H:%M:%S')}] {tag}   {label}  wall={wall:.1f}s")
    return rec


_do("PEMFC + AST cycling, sim=600 s (100 cycles)",
    variant="Dual-scale", aux=False, t_end=600)

_do("PEMFC_dyn + AST cycling, sim=300 s (50 cycles)",
    variant="Dynamic", aux=True, t_end=300)

print(f"\n[{time.strftime('%H:%M:%S')}] Wrote {JSON_PATH}")
print(f"PASS {sum(r.get('success') for r in records)}/{len(records)}")
