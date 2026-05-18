"""Live-monitoring window for the AST square-load simulation.

Single-file converted version of `square load.ipynb` in the same folder.
A small Tk window runs the same dual-scale PEMFC integration as the
notebook (BDF, ``max_step = 0.1``) and reports progress live: every call
to ``dxdt`` peeks at the wall clock and -- at most once per
``UI_TICK_INTERVAL_S`` seconds -- pushes a ``(t, S_N)`` sample to the UI
thread, so the plot and the progress bar advance continuously even while
a single BDF chunk is still running.

Run the window:

    python "simulation/Test_AST/square_load.py"

Run a headless self-test (short simulation, prints the tick / chunk
stream to stdout):

    python "simulation/Test_AST/square_load.py" --test

Adjust ``CHUNK_SECONDS`` / ``UI_TICK_INTERVAL_S`` / ``MAX_STEP_S`` /
``T_FINAL`` near the top to change behaviour.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")            # must be set before importing pyplot / Figure
import matplotlib.pyplot as plt    # noqa: E402,F401
import numpy as np                 # noqa: E402
import tkinter as tk               # noqa: E402
from tkinter import ttk            # noqa: E402
from matplotlib.figure import Figure                              # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # noqa: E402
from scipy.integrate import solve_ivp                             # noqa: E402

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
project_root = next(p for p in [HERE, *HERE.parents] if p.name == "MFC2024")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.initialize import operating_inputs, parameters, init_x       # noqa: E402
from config.settings import solver_variable_names, solver_flux_names     # noqa: E402
from model.dualscale import PEMFC                                         # noqa: E402
from model.inst_values import getECSA                                     # noqa: E402
from modules.signals import generate_step_load                            # noqa: E402

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
T_FINAL             = 5000 * 6   # full AST duration                       (s)
CHUNK_SECONDS       = 60         # simulated seconds per BDF chunk         (s)
UI_TICK_INTERVAL_S  = 1.0        # min wall-clock interval between live ticks (s)
MAX_STEP_S          = 1e-3       # tight BDF step cap needed for the high-RH /
                                 # high-pressure conditions (Phi_c_des=0.85,
                                 # 1.8 bar) combined with the steep square-load
                                 # transitions; larger max_step lets dxdt
                                 # produce NaNs during the ramps.
TSTART, TEND        = 0.0, 6.0   # square-load period parameters             (s)
I_LOW, I_HIGH       = 20.0, 12000.0   # baseline / plateau current      [A/m^2]
# Symmetric square wave: rise edge centred at TAU_SWITCH, fall edge at
# (TEND - TSTART) - TAU_SWITCH. T_SWITCH controls the ramp width.
TAU_SWITCH          = 1.5
T_SWITCH            = 0.5

# Operating conditions for this AST setup.
PHI_C_DES           = 0.85
SA, SC              = 1.2, 2.0
PA_DES, PC_DES      = 1.8e5, 1.8e5
TFC_K               = 353.15

# Calibrated parameter overrides. With alpha_c = 0.95 (set in
# model/coefficients.py) these produce ~0.95 V at the low plateau and
# ~0.60 V at the high plateau (steady state).
PARAM_OVERRIDES = {
    "OCV":         0.975,
    "epsilon_gdl": 0.7,
    "epsilon_cl":  0.5,
    "i0_c_ref":   10.0,
    "kappa_c":     0.05,
    "a_slim":      0.2,
    "b_slim":      0.2,
    "Hgdl":      1e-4,
    "Re":        1e-7,
}

# Experimental ECSA reference from
# simulation/parameter calibration/micro scale/ECSA.ipynb
# (Baroody & Kjeang, J. Electrochem. Soc. 168, 044524, 2021).
# Each AST cycle in this notebook is (TEND - TSTART) = 6 s of simulated
# time, so cycle k corresponds to t = 6 * k seconds.
REF_CYCLES  = [0,    5000, 10000, 15000, 20000, 25000, 30000]
REF_ECSA    = [1.0,  0.71, 0.69,  0.595, 0.61,  0.58,  0.39]
REF_TIME_S  = [c * (TEND - TSTART) for c in REF_CYCLES]


def _reference_within(t_final: float):
    """Return ``(times, ecsa)`` lists with only the reference points that
    fall within ``[0, t_final]`` -- so the reference adapts to whatever
    simulation window the user has chosen."""
    pairs = [(t, e) for t, e in zip(REF_TIME_S, REF_ECSA) if t <= t_final + 1e-9]
    if not pairs:
        return [], []
    times, ecsa = zip(*pairs)
    return list(times), list(ecsa)


# ---------------------------------------------------------------------------
# Model + initial state (built once, shared with the worker)
# ---------------------------------------------------------------------------
operating_inputs["current_density"] = generate_step_load(
    TSTART, TEND, I_LOW, I_HIGH, TAU_SWITCH, T_SWITCH)
operating_inputs["Phi_c_des"] = PHI_C_DES
operating_inputs["Sa"]        = SA
operating_inputs["Sc"]        = SC
operating_inputs["Pa_des"]    = PA_DES
operating_inputs["Pc_des"]    = PC_DES
operating_inputs["Tfc"]       = TFC_K

# Apply the calibrated parameter overrides on top of the project defaults.
parameters.update(PARAM_OVERRIDES)

model = PEMFC(param=parameters, operating_inputs=operating_inputs,
              variable_names=solver_variable_names, flux_names=solver_flux_names)
y0 = np.asarray(init_x(operating_inputs, parameters), dtype=float)
print(f"PEMFC built with {len(y0)} states; AST duration = {T_FINAL} s.")

SN_INDICES = np.array(
    [model.variable_names.index(f"S_N_ccl_{i}")
     for i in range(1, model.parameters["n_group_pt"] + 1)]
)
ECSA0 = getECSA(model.parameters["prd0"], model.parameters["r_m"])
R_M   = model.parameters["r_m"]


# ---------------------------------------------------------------------------
# Worker thread: chunked solve_ivp + intra-chunk live ticks
# ---------------------------------------------------------------------------
def simulation_worker(ui_queue: queue.Queue, stop_event: threading.Event,
                      t_final: float = T_FINAL,
                      chunk_seconds: float = CHUNK_SECONDS,
                      tick_interval_s: float = UI_TICK_INTERVAL_S,
                      max_step_s: float = MAX_STEP_S):
    """Integrate the model in ``chunk_seconds``-long slices of simulated
    time. Inside each slice, a wrapped ``dxdt`` emits a *tick* event on
    ``ui_queue`` every ``tick_interval_s`` wall-clock seconds, so the UI
    sees progress even while BDF is still busy in one slice."""

    wall_start = time.perf_counter()
    # Local-variable cache for the closures below.
    raw_dxdt  = model.dxdt
    sn_idx    = SN_INDICES
    ecsa0     = ECSA0
    r_m       = R_M
    interval  = float(tick_interval_s)

    # Mutable state shared with the dxdt wrapper.
    tick_state = {"last_wall": time.perf_counter() - interval,
                  "stop_flag": False}

    def dxdt_with_tick(t, y):
        if stop_event.is_set():
            tick_state["stop_flag"] = True
            # Returning zeros stops the state from moving; solve_ivp will
            # still finish its current step, but the outer loop bails.
            return np.zeros_like(y)
        out = raw_dxdt(t, y)
        now = time.perf_counter()
        if now - tick_state["last_wall"] >= interval:
            tick_state["last_wall"] = now
            try:
                sn = getECSA(y[sn_idx], r_m) / ecsa0
            except Exception:
                sn = float("nan")
            ui_queue.put({"type": "tick",
                          "t": float(t), "sn": float(sn),
                          "wall": now - wall_start})
        return out

    t_cur = 0.0
    y_cur = y0.copy()
    while t_cur < t_final:
        if stop_event.is_set():
            break
        t_next = min(t_cur + chunk_seconds, t_final)
        sol = solve_ivp(
            fun=dxdt_with_tick,
            t_span=(t_cur, t_next),
            y0=y_cur,
            method="BDF",
            max_step=max_step_s,
        )
        if tick_state["stop_flag"] or stop_event.is_set():
            ui_queue.put({"type": "stopped",
                          "t": float(sol.t[-1]) if len(sol.t) else t_cur,
                          "wall": time.perf_counter() - wall_start})
            return
        if not sol.success:
            ui_queue.put({"type": "error",
                          "msg": sol.message,
                          "t": float(sol.t[-1]) if len(sol.t) else t_cur})
            return
        # Push the chunk's BDF-internal trajectory so the line gets the
        # exact points solve_ivp produced (the ticks were intra-chunk
        # samples, not necessarily on the BDF grid).
        sn_chunk = [
            getECSA(sol.y[sn_idx, j], r_m) / ecsa0
            for j in range(len(sol.t))
        ]
        ui_queue.put({"type": "chunk",
                      "t": list(map(float, sol.t)),
                      "sn": sn_chunk,
                      "t_cur": float(t_next),
                      "wall": time.perf_counter() - wall_start})
        t_cur = t_next
        y_cur = sol.y[:, -1]

    ui_queue.put({"type": "done",
                  "t": float(t_cur),
                  "wall": time.perf_counter() - wall_start})


# ---------------------------------------------------------------------------
# Tk UI
# ---------------------------------------------------------------------------
class LiveWindow:
    def __init__(self, t_final: float = T_FINAL,
                 chunk_seconds: float = CHUNK_SECONDS,
                 tick_interval_s: float = UI_TICK_INTERVAL_S):
        self.t_final         = t_final
        self.chunk_seconds   = chunk_seconds
        self.tick_interval_s = tick_interval_s

        self.root = tk.Tk()
        self.root.title("PEMFC AST square load -- live")
        self.root.geometry("820x560")

        # Progress bar.
        self.progress = ttk.Progressbar(
            self.root, length=780, mode="determinate", maximum=t_final,
        )
        self.progress.pack(padx=12, pady=(10, 4), fill="x")

        # Status line.
        self.status = tk.Label(
            self.root, anchor="w", font=("Consolas", 9),
            text=(f"Ready.  chunk = {chunk_seconds:g} s  "
                  f"tick = {tick_interval_s:g} s wall."),
        )
        self.status.pack(padx=12, pady=(0, 6), fill="x")

        # Embedded matplotlib figure.
        self.fig = Figure(figsize=(8, 3.5))
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_xlabel("Simulated time (s)")
        self.ax.set_ylabel("ECSA ratio  S_N  (-)")
        # X-axis matches the requested simulation window exactly.
        self.ax.set_xlim(0, t_final)
        self.ax.set_ylim(0.0, 1.05)
        self.ax.grid(True, alpha=0.3)
        # Reference curve from the ECSA notebook, truncated to the
        # simulation window so the x-axis stays at [0, t_final] -- only
        # reference points that fall within the requested simulation
        # range are drawn. Extend `T_FINAL` if you want to see more of
        # the published curve.
        ref_t, ref_e = _reference_within(t_final)
        if ref_t:
            self.ax.plot(ref_t, ref_e, marker="o", linestyle="--",
                         color="0.45", linewidth=1.2, markersize=5,
                         label="reference (Baroody & Kjeang 2021)")
        # Two live artists: a continuous line for completed BDF samples,
        # and a single moving dot for the latest intra-chunk tick.
        self.line,      = self.ax.plot([], [], linewidth=1.4, color="#0072B2", label="trajectory")
        self.tick_dot,  = self.ax.plot([], [], marker="o", linestyle="",
                                       markersize=6, color="#D55E00", label="live")
        self.ax.legend(loc="lower left", fontsize=8)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(padx=12, pady=4, fill="both", expand=True)

        # Buttons.
        bar = tk.Frame(self.root)
        bar.pack(padx=12, pady=(4, 10), fill="x")
        self.start_btn = tk.Button(bar, text="Start", width=10, command=self.on_start)
        self.start_btn.pack(side="left")
        self.stop_btn  = tk.Button(bar, text="Stop",  width=10,
                                   state="disabled", command=self.on_stop)
        self.stop_btn.pack(side="left", padx=(8, 0))

        # State.
        self.t_data:  list[float] = []
        self.sn_data: list[float] = []
        self.queue       = queue.Queue()
        self.stop_event  = threading.Event()
        self.worker: threading.Thread | None = None

    def on_start(self):
        self.t_data.clear(); self.sn_data.clear()
        self.line.set_data([], []); self.tick_dot.set_data([], [])
        self.progress["value"] = 0
        self.stop_event.clear()
        self.start_btn["state"] = "disabled"
        self.stop_btn["state"]  = "normal"
        self.status.config(text="Running...")
        self.canvas.draw()

        self.worker = threading.Thread(
            target=simulation_worker,
            args=(self.queue, self.stop_event,
                  self.t_final, self.chunk_seconds, self.tick_interval_s),
            daemon=True,
        )
        self.worker.start()
        self.root.after(150, self._poll_queue)

    def on_stop(self):
        self.stop_event.set()
        self.status.config(text="Stop requested -- finishing current step...")
        self.stop_btn["state"] = "disabled"

    def _poll_queue(self):
        drained = False
        try:
            while True:
                self._handle(self.queue.get_nowait())
                drained = True
        except queue.Empty:
            pass
        if drained:
            self.canvas.draw_idle()
        if self.worker is not None and self.worker.is_alive():
            self.root.after(150, self._poll_queue)
        else:
            # Final drain after the worker exits.
            try:
                while True:
                    self._handle(self.queue.get_nowait())
            except queue.Empty:
                pass
            self.canvas.draw()
            self.start_btn["state"] = "normal"
            self.stop_btn["state"]  = "disabled"

    def _handle(self, msg: dict):
        kind = msg.get("type")
        if kind == "tick":
            t  = msg["t"]
            sn = msg["sn"]
            self.tick_dot.set_data([t], [sn])
            self.progress["value"] = t
            self.status.config(text=(
                f"running  t = {t:>7.1f} / {self.t_final} s "
                f"({t/self.t_final*100:5.1f} %)   "
                f"wall = {msg['wall']:>6.1f} s   "
                f"S_N = {sn:.4f}"
            ))
        elif kind == "chunk":
            self.t_data.extend(msg["t"])
            self.sn_data.extend(msg["sn"])
            self.line.set_data(self.t_data, self.sn_data)
            self.progress["value"] = msg["t_cur"]
            self.status.config(text=(
                f"chunk    t = {msg['t_cur']:>7.1f} / {self.t_final} s   "
                f"wall = {msg['wall']:>6.1f} s   "
                f"S_N = {self.sn_data[-1]:.4f}"
            ))
        elif kind == "error":
            self.status.config(text=f"Solver failed at t={msg['t']:.1f} s : {msg['msg']}")
        elif kind == "stopped":
            self.status.config(text=f"Stopped by user at t={msg['t']:.1f} s "
                                    f"(wall {msg['wall']:.1f} s).")
        elif kind == "done":
            self.status.config(text=f"Finished. simulated t = {msg['t']:.0f} s   "
                                    f"wall = {msg['wall']:.0f} s.")

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Headless test entry point: drive the worker without a Tk window so
# CI / scripts can verify the live-update plumbing.
# Usage:  python square_load.py --test
# ---------------------------------------------------------------------------
def _headless_test(t_final: float = 1.0, chunk_seconds: float = 0.5,
                   tick_interval_s: float = 0.25, deadline_s: float = 120.0):
    print(f"[test] t_final={t_final} s  chunk={chunk_seconds} s  "
          f"tick={tick_interval_s} s  deadline={deadline_s} s wall")
    q = queue.Queue()
    stop = threading.Event()
    th = threading.Thread(
        target=simulation_worker,
        args=(q, stop, t_final, chunk_seconds, tick_interval_s),
        daemon=True,
    )
    t0 = time.perf_counter()
    th.start()

    n_tick = n_chunk = 0
    done = None

    def handle(msg):
        nonlocal n_tick, n_chunk, done
        kind = msg["type"]
        if kind == "tick":
            n_tick += 1
            print(f"  tick  t={msg['t']:6.2f}  S_N={msg['sn']:.4f}  wall={msg['wall']:.2f}")
        elif kind == "chunk":
            n_chunk += 1
            print(f"  CHUNK t_cur={msg['t_cur']:6.2f}  pts={len(msg['t']):3d}  "
                  f"S_N_last={msg['sn'][-1]:.4f}  wall={msg['wall']:.2f}")
        elif kind in ("done", "error", "stopped"):
            done = msg
            print(f"  {kind.upper():>7s}  t={msg.get('t', 0):.2f}  wall={msg.get('wall', 0):.2f}")

    while (th.is_alive() or not q.empty()) and (time.perf_counter() - t0) < deadline_s:
        try:
            handle(q.get(timeout=0.5))
        except queue.Empty:
            continue
        if done is not None:
            break
    if th.is_alive():
        stop.set()
        th.join(timeout=10)
    # Final drain.
    while not q.empty():
        handle(q.get_nowait())
    print(f"[test] ticks={n_tick}  chunks={n_chunk}  "
          f"final={None if done is None else done.get('type')}")
    return n_tick, n_chunk, done


if __name__ == "__main__":
    if "--test" in sys.argv:
        _headless_test()
    else:
        LiveWindow().run()
