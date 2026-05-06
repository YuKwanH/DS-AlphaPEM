# `gui/app.py`

**Purpose.** Streamlit entry point. Composes the three-section dashboard (parameters · options · results), the save/download box, and the global **▶ Run** button into a single page.

## Run

```bash
streamlit run gui/app.py
```

## Layout

- Header with title and the **▶ Run** button.
- Three columns of equal-height containers (`SECTION_HEIGHT = 820`, `border=True`):
  - § 1 — [`gui/parameters.md`](parameters.md)
  - § 2 — [`gui/options.md`](options.md)
  - § 3 — [`gui/results.md`](results.md)
- Save/download box ([`gui/save.md`](save.md)) below the result section.

## Session state

Initialised by `_ensure_state()`:
- `params`, `op_inputs` — deep copies of the project defaults that the panels mutate.
- `model_variant`, `profile_kind`, `profile_cfg`, `t_start`, `t_end`, `max_step`, `method`.
- `last_result` — the dict produced by `gui.runner.run`.
- `_units_A_per_m2` — one-time migration flag (rescales any old A/cm² values stored in earlier sessions).

## Trigger

`_trigger_run()` is the `on_click` for the Run button. It builds the profile via `panel_options.build_profile_func`, calls `gui.runner.run`, and stores the result under `last_result` for the panels to pick up on the next rerun.

## Related

- [`gui/style.md`](style.md) is applied first so every figure inherits the research-paper look.
- [`gui/runner.md`](runner.md) does the actual simulation.
