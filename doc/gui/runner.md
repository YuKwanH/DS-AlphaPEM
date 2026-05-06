# `gui/runner.py`

**Purpose.** Dispatch layer between the GUI and the three model variants. Handles defensive copying, the `solve_ivp` call, and an automatic LSODA fallback when newer scipy raises a hard error on a transient NaN that older scipy silently tolerates.

## Public API

```python
run(params, op_inputs, model_variant, profile_func, t_span,
    max_step=0.1, method="BDF", polar_sweep=None)
    -> (model, sol_or_polar, status)
```

`status` is a dict with `runtime_s`, `n_states`, `n_steps`, `success`, `message`, `model_variant`, `kind` (`"transient"` or `"polar"`).

## Pipeline

1. Defensive copy: `params = dict(params)`, `op_inputs = dict(op_inputs)`.
2. `op_inputs["current_density"] = profile_func`.
3. Build the model (`PEMFC` / `PEMFC_dyn` / `PEMFC_stat`).
4. Integrate via `_solve_with_fallback`: try the requested method, on `ValueError` mentioning `nan`/`inf` retry with LSODA. Status reports `"<variant> → LSODA fallback"` so the user sees what happened.
5. `model._recovery(sol)` for transient runs; sweep `i` for polar runs.

## Related

- [`gui/options.md`](options.md) — produces the `profile_func` and `polar_sweep` payloads.
- [`gui/results.md`](results.md) — consumes the `(model, sol_or_polar, status)` tuple.
- [`modules/tests.md`](../modules/tests.md) — the equivalent runner outside the GUI (no fallback, no status dict).
