# `gui/style.py`

**Purpose.** Single research-paper matplotlib `rcParams` profile. Imported once by `gui/app.py` so every figure (in any panel, any tab, the live preview, the spatial profile) inherits a consistent look without per-figure styling code.

## What `apply()` sets

- **Categorical palette** — Okabe-Ito 8-colour cycle (colour-blind safe; commonly used in Nature / IEEE).
- **Sequential colormap** — `cividis` (perceptually uniform, prints well in greyscale).
- **Typography** — serif font stack (Times, then DejaVu Serif, then STIX), STIX math fonts, bold titles.
- **Axes** — thin spines, top + right hidden, dark-grey edges.
- **Grid** — light grey at 25 % alpha.
- **Ticks** — outward, uniform sizes (Nature/IEEE convention).
- **DPI** — 110 on-screen, 200 on save.

## Constants

- **`PALETTE`** — list of the 8 hex colours, exported so other modules can pick a specific colour by index when they want to keep it the same across plots.
