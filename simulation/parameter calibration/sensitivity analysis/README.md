# Sensitivity analysis — undetermined parameters vs. cell voltage

One-factor-at-a-time (OAT) studies of how each undetermined physical parameter affects the
cell voltage, and how that sensitivity changes along the polarization curve (low → high
current).

Normalized sensitivity coefficient used throughout:

    chi = (1/4) * sum_k | (dR_k / R) / (dA_k / A) |,   dA/A in {-20%, -10%, +10%, +20%}

A parameter is "sensitive" when chi > 0.05. The signed version (no absolute value) gives the
direction of influence, and the cosine similarity of the per-parameter chi(i) profiles shows
which parameters are interchangeable (not separately identifiable) in a fit.

## Notebooks

- **`OAT sensitivity (dynamic).ipynb`** — the dual-scale **dynamic** model `PEMFC`
  (`model/model.py`) run **with the auxiliary system on** (`aux_system=True`: compressor,
  humidifier injection, manifolds). Condition: T = 333.15 K, RH_c = 0.5, 1.5 bar. Voltage is
  swept with a slow current staircase so the balance-of-plant tracks the load. ~22 min to run.
- **`OAT sensitivity.ipynb`** — the algebraic **static** model `PEMFC_stat` (`model/static.py`).
  Fast (~15 s). Kept for comparison.

## Headline results — dynamic model (T = 333.15 K, RH_c = 0.5, 1.5 bar)

Sensitive to the cell voltage somewhere on the curve: **OCV, tau, epsilon_mc, Hcl** (all
across the curve), **Re** (high current only), **Hgdl** (low current only), and marginally
epsilon_gdl.

- At high current the ohmic/water cluster dominates: chi(tau) ≈ 0.97, chi(epsilon_mc) ≈ 0.83,
  chi(Hcl) ≈ 0.79 at 1.45 A/cm². OCV is the offset (chi ≈ 1.2–1.7 everywhere).
- **Degeneracy**: epsilon_mc ↔ tau are perfectly collinear (product epsilon_mc^tau in Rccl);
  at high current the whole set {epsilon_mc, tau, Re, Hcl} is mutually collinear (|cos| > 0.98)
  → only one combined "ohmic slope" is identifiable from the polarization curve. Break it with
  HFR/EIS.
- **Freeze** (chi ≈ 0 on the voltage): i0_c_ref, kappa_c (own eta_act, but eta_act is small),
  a_slim/b_slim/a_switch (own eta_conc, negligible here), epsilon_c, epsilon_cl, kappa_co.
- **`e` (capillary exponent) is unmeasurable here**: stable at nominal 3 but every ±perturbation
  crashes the integrator (fragile `s**e` liquid-flux term). Fix from literature.

Contrast with the static study: there only OCV crossed the threshold and the ohmic parameters
were near ~0.05; the dynamic + aux model resolves water/O2 transport and exposes far more
identifiable structure, mostly on the high-current branch.

See each notebook's Conclusions cell for the full tables and the calibration recipe.

## Notes on the parameter values

The `config/initialize.py` nominals starve the cathode above ~0.5 A/cm² in the dynamic model
with aux at this condition (thick, compressed GDL). The dynamic notebook therefore uses a
physically reasonable, opened-up set (ε_gdl = 0.70, ε_c = 0.15, Hgdl = 200 µm, i0_c_ref = 0.5,
κ_c = 1.0, OCV = 0.95, …) that yields a full, stable curve — see the parameter table in the
notebook. `C_scl` is excluded (the dual-scale model evaluates eta_c algebraically, so the
double-layer capacitance does not affect the steady voltage).

## How to run

Open a notebook and run all cells. The project root is auto-detected by walking up until
`config/initialize.py` and `model/model.py` are found. Operating point, current grid and
parameter ranges are set in the first code cells.

## Possible extensions

- Break the high-current ohmic degeneracy by adding the HFR/EIS outputs as extra objectives.
- Global methods (Morris, Sobol) to capture parameter interactions the OAT misses.
- Repeat at other (T, P, RH) points to check whether the sensitivity ranking is condition-dependent.
