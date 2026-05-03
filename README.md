# PEMFC 1D Model — Baltic 300W Stack

**Creator:** Qian HE  
**Institution:** FCLAB, FEMTO-ST, Belfort, FR

A 1D PEMFC simulation framework for the 300W Baltic stack, providing both **static** (algebraic) and **dynamic** (ODE-based) models.

---

## Quick Start — Streamlit GUI

A three-section dashboard (parameters · simulator options · results + save/download) wraps the models. These instructions work on **Windows, macOS, and Linux** and assume **no prior Python experience**. You will use a *terminal* (the black/white text window where you type commands) — see below if you've never opened one.

### Prerequisites (install once)

You need three tools installed on your computer:

1. **Python 3.10 or newer** — download from <https://www.python.org/downloads/>. During the Windows installer, **tick "Add Python to PATH"**, otherwise the terminal will not find it.
2. **Git** — download from <https://git-scm.com/downloads>. Used to fetch the project from GitHub.
3. *(Optional)* **Anaconda / Miniconda** — only if you prefer `conda` over `venv` (Option B below). Get it from <https://www.anaconda.com/download>.

To verify they work, open a terminal (see "Opening a terminal" below) and run:

```bash
python --version       # should print Python 3.10.x or higher
git --version          # should print git version 2.x
```

If either command says *"not recognised / not found"*, the tool is not installed correctly — re-install and make sure the *"Add to PATH"* option is selected.

### Step 1 — Clone the project

Pick a folder on your computer where you want the project to live (e.g. `C:\Users\<you>\Documents` on Windows, or `~/projects` on macOS/Linux). Open a terminal **in that folder** and run:

```bash
git clone <this-repo-url> MFC2024
```

This creates a new folder called **`MFC2024`** containing the entire project. **That folder is the project root** — every command below assumes you are inside it.

### Step 2 — Find the project root and open a terminal there

The "project root" is the `MFC2024/` folder you just created. Every command in the following steps must be run from inside that folder. There are two ways to get a terminal into the right place:

#### Option A — Open the folder, then open a terminal inside it
- **Windows (File Explorer):** double-click into the `MFC2024` folder so it is open. Click the address bar at the top of the window, type `cmd` and press *Enter*. A Command Prompt opens already located in the project root.
- **macOS (Finder):** right-click the `MFC2024` folder in Finder → *"New Terminal at Folder"*. (If you don't see that option, enable it in *System Settings → Keyboard → Keyboard Shortcuts → Services → Files & Folders*.)
- **Linux (most file managers):** right-click inside the `MFC2024` folder → *"Open in Terminal"*.

#### Option B — Use `cd` from any open terminal
First find the absolute path of the `MFC2024` folder, then `cd` into it.
- **Windows:** in File Explorer, click the `MFC2024` folder once, then look at the address bar. Click the address bar and copy the text shown (e.g. `C:\Users\you\Documents\MFC2024`). In the terminal type:
  ```cmd
  cd "C:\Users\you\Documents\MFC2024"
  ```
- **macOS:** in Finder, right-click `MFC2024` while holding *Option*, choose *"Copy <name> as Pathname"*. Then in Terminal:
  ```bash
  cd "/Users/you/Documents/MFC2024"
  ```
- **Linux:** in your file manager, right-click → *Properties* to read the path, or run `realpath MFC2024` from the parent folder. Then:
  ```bash
  cd /home/you/projects/MFC2024
  ```

After `cd`, type `pwd` (macOS/Linux) or `cd` with no arguments (Windows) to confirm you are in the project root. You should see the `MFC2024` path printed.

> **Tip — what does the project root look like?** The right place will contain `gui/`, `model/`, `config/`, `simulation/`, `requirements.txt`, and this `README.md`. If your terminal's listing (`dir` on Windows, `ls` on macOS/Linux) doesn't show those, you are not in the project root yet.

### Step 3 — Create a Python environment and install dependencies

A "Python environment" is an isolated place to install the project's libraries so they don't clash with anything else on your system. Pick **one** of the two options below.

**Option A — `venv` (built into Python, simplest):**
```bash
# 1. create the environment in a folder called .venv inside the project
python -m venv .venv

# 2. activate it — pick the line that matches your terminal:
.venv\Scripts\activate.bat        # Windows  (cmd)
.venv\Scripts\Activate.ps1        # Windows  (PowerShell)
source .venv/bin/activate         # macOS / Linux

# 3. install the project's libraries
pip install -r requirements.txt
```

After activation your terminal prompt should start with `(.venv)` — that's how you know the environment is active. You only need to run steps 1 and 3 once; in future sessions just activate (step 2) and run.

**Option B — `conda` (recommended if you also use the Jupyter notebooks):**
```bash
conda create -n mfc python=3.11
conda activate mfc
pip install -r requirements.txt
```

The prompt will start with `(mfc)` once the environment is active.

### Step 4 — Run the GUI

Still inside the project root, with the environment active, run:

```bash
streamlit run gui/app.py
```

A browser tab opens at `http://localhost:8501` showing the dashboard. Stop the server with **Ctrl + C** in the terminal when you're done.

The next time you want to use the GUI, you only need three steps: open a terminal in the project root → activate the environment → run `streamlit run gui/app.py`.

### Layout

- **§1 Parameters** — every entry from `config/initialize.py`, grouped by region (Operating · GC · GDL · CL · MEM · Saturation · Numerics) with a region filter.
- **§2 Options** — model variant (Static / Dynamic / Dual-scale), test profile (Constant · Step · Polarization · EIS · AST cycling), time span, solver, and mesh.
- **§3 Results** — six tabs (Cell performance · Spatial profile · Manifolds · Water content · Degradation · Custom variable picker), plus a save/download box for CSV / Excel / NumPy export.

### Common errors

| Error message | What it means | Fix |
|---|---|---|
| `'python' is not recognized` / `command not found` | Python is not installed or not on your PATH. | Re-install Python and tick *Add to PATH*. |
| `'streamlit' is not recognized` | The environment is not active, or `pip install` was skipped. | Run the activate command from Step 3 first, then `pip install -r requirements.txt`. |
| `No module named 'config'` | You ran the command from outside the project root. | `cd` into the `MFC2024` folder before running `streamlit`. |
| Browser tab doesn't open | Streamlit is running but can't auto-open your browser. | Open <http://localhost:8501> manually. |

### Note on SciPy versions

The dual-scale model produces a transient NaN during ramp transitions that older `scipy` (< 1.15) silently tolerates, while newer `scipy` raises a hard error during the BDF Jacobian factorisation. The GUI handles this automatically: if a BDF run fails with the NaN error, it retries with LSODA and the status strip in §3 reads `<variant> → LSODA fallback`. To avoid the fallback entirely, pin `scipy<1.15` in your environment (the notebooks were validated against `scipy 1.13` / `numpy 1.26`).

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