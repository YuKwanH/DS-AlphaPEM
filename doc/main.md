# `main.py`

**Purpose.** Minimal smoke-test entry point. Builds a `PEMFC` with the project default `parameters` / `operating_inputs` and prepares the initial state vector.

**Key calls**
- `PEMFC(param=parameters, operating_inputs=operating_inputs, variable_names=…, flux_names=…)` — build the model.
- `init_x(operating_inputs, parameters)` — assemble the 181-element state vector.

**Related**
- For a full simulation pipeline, prefer the protocol runners in [`modules/tests.md`](modules/tests.md) or the GUI ([`gui/app.md`](gui/app.md)).
