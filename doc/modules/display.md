# `modules/display.py`

**Purpose.** Six-panel spatial-profile plot of the cell at a given time index. Used by both the legacy notebooks (`display`) and the GUI's "Spatial profile" tab (`build_profile_figure`).

## API

- **`build_profile_figure(solution, model, t_index=-1) -> matplotlib.figure.Figure`**  
  Returns a 3 × 2 figure showing vapour pressure, O₂ concentration, H₂ concentration, liquid saturation, water content `λ`, and temperature across the AGC → AGDL → ACL → MEM → CCL → CGDL → CGC profile at `solution.t[t_index]`.

- **`display(solution, model)`** — thin convenience wrapper that builds the figure and calls `plt.show()`. Used in the older notebooks; new code (and the GUI) should call `build_profile_figure` directly.

## Use

```python
from modules.display import build_profile_figure
fig = build_profile_figure(sol, model, t_index=-1)
fig.savefig("profile_t_end.png")
```

## Related

- [`config/settings.md`](../config/settings.md) — supplies `nodes`, `borders`, `nodes_*` lookup helpers that this plot depends on.
- [`gui/results.md`](../gui/results.md) — calls `build_profile_figure` from the "Spatial profile" tab.
