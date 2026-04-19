# PEMFC 1D Model — Baltic 300W Stack

**Creator:** Qian HE  
**Institution:** FCLAB, FEMTO-ST, Belfort, FR

A 1D PEMFC simulation framework for the 300W Baltic stack, providing both **static** (algebraic) and **dynamic** (ODE-based) models.

---

## Model Overview

### Static Model (`PEMFC_stat`)

Solves the steady-state water/gas distribution across the MEA for a given current density $i$.

- **Solver**: Fixed-point iteration on net membrane water flux $J_{net}$ (algebraic, no time dimension)
- **Water balance**: Iterates until $|J_{net} - J_{mem}| \le 10^{-4}$
- **GDL regimes**: Classifies into Vapor (V), Mixed (M), or Liquid (L) with analytical profiles
- **Voltage**: Nernst equilibrium + Tafel cathode overpotential + ohmic resistance
- **Usage**: `model.solve(i)` → single result dict per current point

### Dynamic Model (`PEMFC_dyn`)

Time-evolves ~180+ state variables via `scipy.integrate.solve_ivp` (BDF stiff solver).

- **Solver**: ODE integration with adaptive time stepping
- **State variables**: Gas concentrations, saturations, membrane water content, temperatures, pressures, actuator states
- **Thermal dynamics**: Spatially-resolved temperatures ($T_{agdl}$, $T_{acl}$, $T_{mem}$, $T_{ccl}$, $T_{cgdl}$)
- **Balance of Plant**: Supply/exhaust manifolds, compressor, humidifier, back-pressure valves
- **Phase change**: Explicit condensation/evaporation source terms, membrane sorption/desorption
- **Usage**: `solve_ivp(model.dxdt, (0, T), x0, method='BDF')` → time-series solution

---

## Static vs. Dynamic — Key Differences

| Aspect | Static | Dynamic |
|---|---|---|
| **Approach** | Algebraic fixed-point | ODE system (BDF) |
| **State variables** | None (steady-state) | ~180+ (time-evolving) |
| **Temperature** | Fixed $T_{fc}$ | Spatially-resolved thermal dynamics |
| **Water balance** | Converged algebraically per current | Time-marched sorption, condensation, transport |
| **GDL saturation** | Analytical regime classification | Dynamic front tracking with conditional ODEs |
| **Cathode $\eta_c$** | Algebraic Tafel | ODE: $\frac{d\eta_c}{dt} = \frac{1}{C_{scl} H_{cl}}(i_{fc} - i_0 e^{f \alpha_c F \eta_c / RT})$ |
| **Gas crossover** | Not modeled | $O_2$/$H_2$ permeation through membrane |
| **$N_2$ tracking** | Not tracked | Accumulation/purge dynamics |
| **Auxiliary systems** | None | Manifolds, compressor, humidifier, valves |

---

## State Equations — Static vs. Dynamic

### Gas Channel

| Variable | **Static** (algebraic) | **Dynamic** (ODE) |
|---|---|---|
| $C_{v,agc}$ | $C_{v,agc} = \frac{J_w L_{gc}/H_{gc} + C_{v,in} W_{in}}{W_{out}}$ | $\frac{dC_{v,agc}}{dt} = \frac{J_{v,a,in} - J_{v,a,out}}{L_{gc}} - \frac{J_{v,agc \to agdl}}{H_{gc}}$ |
| $C_{H_2,agc}$ | $C_{H_2,agc} = \frac{P_a}{RT_{fc}} - C_{v,agc}$ | $\frac{dC_{H_2,agc}}{dt} = \frac{J_{H_2,in} - J_{H_2,out}}{L_{gc}} - \frac{J_{H_2,agc \to agdl}}{H_{gc}}$ |
| $C_{v,cgc}$ | Same algebraic balance as anode | $\frac{dC_{v,cgc}}{dt} = \frac{J_{v,c,in} - J_{v,c,out}}{L_{gc}} + \frac{J_{v,cgdl \to cgc}}{H_{gc}}$ |
| $C_{O_2,cgc}$ | $C_{O_2,cgc} = 0.21\left(\frac{P_c}{RT_{fc}} - C_{v,cgc}\right)$ | $\frac{dC_{O_2,cgc}}{dt} = \frac{J_{O_2,in} - J_{O_2,out}}{L_{gc}} + \frac{J_{O_2,cgdl \to cgc}}{H_{gc}}$ |
| $C_{N_2}$ | Not modeled | $\frac{dC_{N_2}}{dt} = \frac{J_{N_2,in} - J_{N_2,out}}{L_{gc}}$ |

### GDL (Anode & Cathode, per node $i$)

| Variable | **Static** (algebraic) | **Dynamic** (ODE) |
|---|---|---|
| $C_{v,gdl}(x)$ | Fick's law: $C_v(x) = C_{v,inter} + \frac{x}{D_{eff}} J_w$ | $\frac{dC_{v,gdl,i}}{dt} = \frac{1}{\varepsilon(1-s_i)}\left(\frac{J_{v,i-1}-J_{v,i}}{\Delta x} + S_{v,i}\right)$ |
| $C_{gas,gdl}(x)$ | $C_{gas}(x) = C_{gas,inter} + \frac{x}{D_{eff}(s)} J_{gas}$ | $\frac{dC_{gas,gdl,i}}{dt} = \frac{1}{\varepsilon(1-s_i)}\frac{J_{gas,i-1}-J_{gas,i}}{\Delta x}$ |
| $s_{gdl}$ | Leverett J-function inversion | $\frac{ds_{gdl,i}}{dt} = \frac{1}{\rho_{H_2O}\varepsilon}\left(\frac{J_{l,i-1}-J_{l,i}}{\Delta x} + M_{H_2O}S_{l,i}\right)$ |
| $x_{front}$ | Analytical: $x_f = \frac{(C_{v,sat} - C_{v,inter})D_{eff}}{J_w}$ | Saturation front tracking algorithm |

### Catalyst Layer

| Variable | **Static** (algebraic) | **Dynamic** (ODE) |
|---|---|---|
| $C_{v,acl}$ | Lumped into GDL endpoint | $\frac{dC_{v,acl}}{dt} = \frac{1}{\varepsilon_{cl}}\left(\frac{J_{v,agdl \to acl}}{\Delta x} - S_{sorp,acl} + S_{v,acl}\right)$ |
| $C_{H_2,acl}$ | $C_{H_2,acl} = C_{H_2,agdl}(H_{gdl}) + \frac{H_{cl}}{D_{a,eff}} J_{H_2}$ | $\frac{dC_{H_2,acl}}{dt} = \frac{1}{\varepsilon_{cl}}\left(\frac{J_{H_2,agdl \to acl} - J_{H_2,acl \to mem}}{\Delta x} - \frac{i}{2FH_{cl}}\right)$ |
| $C_{O_2,ccl}$ | $C_{O_2,ccl} = C_{O_2,cgdl}(H_{gdl}) + \frac{H_{cl}}{D_{c,eff}} J_{O_2}$ | $\frac{dC_{O_2,ccl}}{dt} = \frac{1}{\varepsilon_{cl}(1-s_{ccl})}\left(\frac{J_{O_2,mem \to ccl} - J_{O_2,ccl \to cgdl}}{\Delta x} - \frac{i}{4FH_{cl}}\right)$ |
| $s_{acl},\; s_{ccl}$ | Not modeled | $\frac{ds_{cl}}{dt} = \frac{1}{\rho_{H_2O}\varepsilon_{cl}}\left(\frac{J_{l,in}-J_{l,out}}{\Delta x} + M_{H_2O}S_{l,cl}\right)$ |
| $\eta_c$ | Inverse Tafel: $\eta_c = \frac{RT}{\alpha_c F \, f_{drop}}\ln\!\left(\frac{i}{i_{0,c}}\left(\frac{C_{O_2,ref}}{C_{O_2,ccl}}\right)^{\!\kappa_c}\right)$ | $\frac{d\eta_c}{dt} = \frac{1}{C_{scl}H_{cl}}\left(i_{fc} - i_{0,c}\exp\!\left(\frac{f_{drop}\,\alpha_c F}{RT_{ccl}}\eta_c\right)\right)$ |

### Membrane

| Variable | **Static** (algebraic) | **Dynamic** (ODE) |
|---|---|---|
| $\lambda_{acl},\;\lambda_{ccl}$ | $\lambda = \min\!\left[\lambda_{eq}(C_v,s,T) + \frac{J_w M_{eq}}{\varepsilon_{cl}H_{cl}\cdot 1.3\rho_{mem}},\;14\right]$ | $\frac{d\lambda_{cl}}{dt} = \frac{M_{eq}}{\rho_{mem}\varepsilon_{mc}}\left(\frac{J_{\lambda,in}-J_{\lambda,out}}{\Delta x} + S_{sorp} + S_p\right)$ |
| $\lambda_{mem}(x)$ | Exponential profile: $\lambda(x) = \frac{1-e^{-x/K_\lambda}}{1-e^{-H_{mem}/K_\lambda}}(\lambda_{acl}-\lambda_{ccl}e^{-H_{mem}/K_\lambda}) + \lambda_{ccl}e^{-x/K_\lambda}$ | $\frac{d\lambda_{mem,i}}{dt} = \frac{M_{eq}}{\rho_{mem}}\frac{J_{\lambda,i-1} - J_{\lambda,i}}{\Delta x_{mem}}$ |
| $J_{mem}$ | $J_{mem} = -\frac{2.5}{22}\frac{i}{F}\frac{\lambda_{ccl}e^{-H_{mem}/K_\lambda}-\lambda_{acl}}{e^{-H_{mem}/K_\lambda}-1}$ | $J_{\lambda,i} = \frac{2.5}{22}\frac{i}{F}\lambda_i - \frac{\rho_{mem}}{M_{eq}}D_w(\lambda_i,T)\frac{\lambda_{i+1}-\lambda_i}{\Delta x}$ |
| $C_{H_2,mem},\;C_{O_2,mem}$ | Not modeled | $\frac{dC_{gas,mem,i}}{dt} = \frac{J_{gas,i-1} - J_{gas,i}}{\Delta x_{mem}}$ |

### Thermal

| Variable | **Static** | **Dynamic** (ODE) |
|---|---|---|
| $T_{agdl,i},\;T_{cgdl,i}$ | Isothermal ($T_{fc}$ = const) | $\frac{dT_{gdl,i}}{dt} = \frac{1}{C_{p,gdl}\rho_{gdl}}\frac{J_{T,i-1}-J_{T,i}}{\Delta x}$ |
| $T_{acl},\;T_{ccl}$ | Isothermal | $\frac{dT_{cl}}{dt} = \frac{1}{C_{p,cl}\rho_{cl}}\left(\frac{J_{T,in}-J_{T,out}}{H_{cl}} + S_r + S_{re}\right)$ |
| $T_{mem,i}$ | Isothermal | $\frac{dT_{mem,i}}{dt} = \frac{1}{C_{p,mem}\rho_{mem}}\left(\frac{J_{T,i-1}-J_{T,i}}{\Delta x} + R_{mem,i}\cdot i^2\right)$ |

### Cell Voltage

| Variable | **Static** | **Dynamic** |
|---|---|---|
| $U_{eq}$ | $U_{eq} = E_0 - 8.5\!\times\!10^{-4}(T-298.15) + \frac{RT}{2F}\!\left[\ln\frac{RTC_{H_2}}{P_{ref}} + \frac{1}{2}\ln\frac{RTC_{O_2}}{P_{ref}}\right]$ | Identical |
| $R_{ohm}$ | $R_{ohm} = \sum_{i=1}^{n_{mem}} \frac{H_{mem}/n_{mem}}{\sigma_p(\lambda_i,T)} + R_e$ | Same formula, evaluated at $\lambda_i(t)$ and $T_i(t)$ |
| $U_{cell}$ | $U_{cell} = U_{eq} - i \cdot R_{ohm} - \eta_c$ | Same, but $\eta_c$ from ODE state |

### Balance of Plant (Dynamic only)

| Variable | **Dynamic** (ODE) |
|---|---|
| $P_{asm},\;P_{csm}$ | $\frac{dP_{sm}}{dt} = \frac{(W_{sm,in} - n_{cell}W_{sm,out})RT_{fc}}{V_{sm}M_{sm}}$ |
| $P_{aem},\;P_{cem}$ | $\frac{dP_{em}}{dt} = \frac{(n_{cell}W_{em,in} - W_{em,out})RT_{fc}}{V_{em}M_{em}}$ |
| $\Phi_{asm},\;\Phi_{aem},\;\Phi_{csm},\;\Phi_{cem}$ | $\frac{d\Phi}{dt} = \frac{(W_{v,in} - W_{v,out})RT_{fc}}{V \cdot P_{sat}}$ |
| $W_{cp}$ | $\frac{dW_{cp}}{dt} = \frac{W_{cp,des} - W_{cp}}{\tau_{cp}}$ |
| $W_{a,inj},\;W_{c,inj}$ | $\frac{dW_{inj}}{dt} = \frac{W_{inj,des} - W_{inj}}{\tau_{hum}}$ |
| $A_{bp,a},\;A_{bp,c}$ | $\frac{dA_{bp}}{dt} = -K_p(P_{des}-P_{gc})$ |

---

## Shared Components

Both models share the same physical foundation:

- **`configuration/settings.py`** — Physical constants, model parameters
- **`configuration/initialize.py`** — Default operating inputs, initial state vectors
- **`model/coefficients.py`** — Thermodynamic/transport functions ($P_{sat}$, $D_{eff}$, $\lambda_{eq}$, $\gamma_{sorp}$, etc.)
- **`model/states.py`** — Voltage calculations, auxiliary state helpers

---

## Project Structure

```
configuration/     Settings, parameters, initialization
model/
  static.py        Steady-state algebraic model
  dynamic.py       Time-dependent ODE model
  coefficients.py  Shared transport/thermodynamic functions
  states.py        Voltage and state helper functions
dynamic/
  gradients.py     Gradient functions per region (dynamic model)
  reaction.py      Pt kinetics (dynamic model)
  control.py       Controller placeholders (dynamic model)
simulation/
  polar_test/      Polarization curve validation (static & dynamic)
  model/           Sub-model analysis (humidifier, BV kinetics, etc.)
```