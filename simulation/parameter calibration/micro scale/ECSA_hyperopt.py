# =============================================================================
# Hyperopt parameter search for the PEMFC catalyst-layer ECSA degradation model
#
# Fits k1, k2, k3, krdp to the experimental ECSA decay reproduced in the last
# figure of ECSA.ipynb.  Drop each `# %% ----` block into a new notebook cell.
# Run AFTER cells 0–11 of the original notebook (constants, kinetic functions,
# globals r_m / Tfc / n_group_ptParticle, initPRD, ucell_tw, the `ccl` class,
# and `exp_data` must already exist in the namespace).
# =============================================================================



# %% ---------- Cell A: imports + experimental reference ---------------------
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

# Experimental ECSA decay (from cell 11 of the notebook)
exp_data  = np.array([1.0, 0.71, 0.69, 0.595, 0.61, 0.58, 0.39])
cycles_x  = np.array([5000 * i for i in range(7)])             # x-axis: cycles
# Global variables
n_group_ptParticle = 50
rmin = 1e-8
rmax = 1e-6
dr = 1e-6 / n_group_ptParticle
r_m = (np.linspace(rmin, rmax, n_group_ptParticle + 1) + dr / 2)[1:]
Tfc = 273.15+80
# Reference ECSA denominator (initial particle radius distribution)
def initPRD(resolution=100, rmin=1e-8, rmax=1e-6, std=0.549, mu=0.538):
    
    radius = np.linspace(rmin, rmax, resolution)
    pdf = 1/(std*np.sqrt(4*np.pi)) * np.exp(-(np.log(radius*1e7)-mu)**2/(std**2*2))
    return pdf
_prd0     = initPRD(resolution=n_group_ptParticle)
ECSA_REF  = np.trapz(y=(r_m ** 2) * _prd0, x=r_m)

# Number of outer iterations to match the last figure exactly:
#   plotted indices are sol_hist[1] .. sol_hist[6]  (6 points)  +  the trivial 1.0
# So the optimiser needs 6 inner solve_ivp calls per evaluation.
N_OUTER = 6


# %% ---------- Cell B: simulation wrapper + hyperopt objective --------------
def simulate_ecsa(k1_v, k2_v, k3_v, krdp_v, n_outer=N_OUTER):
    """Run the ECSA AST simulation with the given kinetic constants and return
    the ECSA trajectory [1.0, e1, e2, ..., e_n_outer] (length n_outer+1)."""
    # Patch module-level globals consumed by the kinetic functions and dxdt.
    global k1, k2, k3, krdp
    k1, k2, k3, krdp = k1_v, k2_v, k3_v, krdp_v

    model  = ccl()                                 # default Ucell = ucell_tw (AST)
    y_last = np.array([0.0] + [0.0] * len(r_m) + model.prd0.tolist())

    ecsa_hist = [1.0]
    for _ in range(n_outer):
        sol = solve_ivp(fun=model.dxdt, y0=y_last,
                        t_span=(0, 5000 * 6), method='BDF')
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        y_last   = sol.y[:, -1]
        prd_last = y_last[-len(r_m):]
        ecsa_hist.append(np.trapz(y=(r_m ** 2) * prd_last, x=r_m) / ECSA_REF)
    return np.array(ecsa_hist)


_eval_log = []   # one entry per evaluation, kept for plotting / diagnostics

def objective(params):
    t0 = time.time()
    try:
        ecsa_sim = simulate_ecsa(params['k1'], params['k2'],
                                 params['k3'], params['krdp'])
        if np.any(np.isnan(ecsa_sim)) or np.any(ecsa_sim < 0) or np.any(ecsa_sim > 1.5):
            raise ValueError("non-physical ECSA values")
        # Drop the trivial point 0 (always 1.0 by construction) from the loss.
        loss   = float(np.mean((ecsa_sim[1:] - exp_data[1:]) ** 2))
        status = STATUS_OK
    except Exception as e:
        ecsa_sim = None
        loss     = 1.0           # large but finite -> TPE will avoid this region
        status   = STATUS_FAIL

    dt = time.time() - t0
    _eval_log.append({'params': params, 'loss': loss, 'status': status,
                      'ecsa_sim': None if ecsa_sim is None else ecsa_sim.tolist(),
                      'wall_s': dt})
    print(f"#{len(_eval_log):3d}  loss={loss:.5f}  "
          f"k1={params['k1']:.2e}  k2={params['k2']:.2e}  "
          f"k3={params['k3']:.2e}  krdp={params['krdp']:.2e}  "
          f"({dt:5.1f}s, {status})")
    return {'loss': loss, 'status': status}


# %% ---------- Cell C: run the optimisation ---------------------------------
# Search space: log-uniform, ~2 orders of magnitude either side of the
# notebook's starting values (k1=3e-9, k2=1e-13, k3=1e-15, krdp=1e-10).
# Tighten these once you see where TPE concentrates its samples.
space = {
    'k1':   hp.loguniform('k1',   np.log(1e-11), np.log(1e-7 )),
    'k2':   hp.loguniform('k2',   np.log(1e-15), np.log(1e-11)),
    'k3':   hp.loguniform('k3',   np.log(1e-17), np.log(1e-13)),
    'krdp': hp.loguniform('krdp', np.log(1e-12), np.log(1e-8 )),
}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=40,                                  # raise once you know it runs
    trials=trials,
    rstate=np.random.default_rng(0),
    show_progressbar=False,
)
print("\nBest parameters found by hyperopt:")
for k, v in best.items():
    print(f"  {k:5s} = {v:.4e}")


# %% ---------- Cell D: visualise the fit and the search history -------------
ecsa_best = simulate_ecsa(best['k1'], best['k2'], best['k3'], best['krdp'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) ECSA fit
axes[0].plot(cycles_x, exp_data,  'o--', color='grey', label='Experimental')
axes[0].plot(cycles_x, ecsa_best, '^-',  color='C0',   label='Simulation (best)')
axes[0].set_xlabel('Cycles')
axes[0].set_ylabel(r'ECSA / ECSA$_0$')
axes[0].set_title(
    f"k1={best['k1']:.2e}   k2={best['k2']:.2e}\n"
    f"k3={best['k3']:.2e}   krdp={best['krdp']:.2e}",
    fontsize=10,
)
axes[0].grid(); axes[0].legend()

# (2) Loss vs. evaluation
losses_all  = [t['result']['loss'] for t in trials.trials]
losses_best = np.minimum.accumulate(losses_all)
axes[1].plot(losses_all,  '.', alpha=0.4, label='per-evaluation loss')
axes[1].plot(losses_best, 'r-',           label='best so far')
axes[1].set_xlabel('Evaluation #')
axes[1].set_ylabel('MSE loss')
axes[1].set_yscale('log')
axes[1].grid(); axes[1].legend()

plt.tight_layout()
plt.show()
