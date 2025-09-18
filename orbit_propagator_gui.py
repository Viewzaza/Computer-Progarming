

# RK4 two-body orbit propagator + optional TLE parsing (sgp4) + batch + GUI + visualization
# Works on Windows: run with `python orbit_propagator_gui.py`
# Dependencies: numpy, matplotlib. Optional: sgp4 (pip install sgp4)

import os
import math
import threading
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# optional sgp4
SGP4_AVAILABLE = False
try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except Exception:
    SGP4_AVAILABLE = False

# ---------------------------
# Constants (SI units)
# ---------------------------
MU_EARTH = 3.986004418e14  # m^3 / s^2

# ---------------------------
# Physics: two-body & RK4
# ---------------------------
def accel_two_body(r, mu=MU_EARTH):
    rnorm = np.linalg.norm(r)
    if rnorm == 0:
        return np.zeros(3)
    return -mu * r / (rnorm**3)

def state_derivative(state, mu=MU_EARTH):
    # state: [rx, ry, rz, vx, vy, vz]
    r = state[0:3]
    v = state[3:6]
    a = accel_two_body(r, mu)
    return np.hstack((v, a))

def rk4_step_state(state, dt, mu=MU_EARTH):
    # state is numpy array length 6
    k1 = state_derivative(state, mu)
    k2 = state_derivative(state + 0.5*dt*k1, mu)
    k3 = state_derivative(state + 0.5*dt*k2, mu)
    k4 = state_derivative(state + dt*k3, mu)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def propagate_rk4_state(r0, v0, dt, steps, mu=MU_EARTH, record_every=1):
    """Return list of (t_offset_s, r_vec (m), v_vec (m/s))."""
    state = np.hstack((r0, v0)).astype(float)
    history = []
    history.append((0.0, state[0:3].copy(), state[3:6].copy()))
    t = 0.0
    for n in range(1, steps+1):
        state = rk4_step_state(state, dt, mu)
        t = n * dt
        if n % record_every == 0:
            history.append((t, state[0:3].copy(), state[3:6].copy()))
    return history

# ---------------------------
# Utilities: state -> orbital elements (osculating)
# ---------------------------
def norm(v):
    return np.linalg.norm(v)

def state_to_orbital(r, v, mu=MU_EARTH):
    r = np.array(r, dtype=float)
    v = np.array(v, dtype=float)
    r_norm = norm(r)
    v_norm = norm(v)
    # angular momentum
    h = np.cross(r, v)
    h_norm = norm(h)
    # eccentricity vector
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = norm(e_vec)
    # energy
    energy = 0.5 * v_norm**2 - mu / r_norm
    if energy == 0:
        a = float('inf')
    else:
        a = -mu / (2 * energy)
    # inclination
    i = math.acos(h[2] / h_norm) if h_norm != 0 else 0.0
    # node vector
    K = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(K, h)
    n_norm = norm(n_vec)
    # RAAN
    RAAN = math.atan2(n_vec[1], n_vec[0]) if n_norm != 0 else 0.0
    # argument of perigee
    if n_norm != 0 and e > 1e-12:
        arg_perigee = math.atan2(np.dot(np.cross(n_vec, e_vec), h)/(h_norm*n_norm),
                                 np.dot(n_vec, e_vec)/(n_norm))
    else:
        arg_perigee = 0.0
    # true anomaly
    if e > 1e-12:
        true_anom = math.atan2(np.dot(np.cross(e_vec, r), h)/(h_norm*e*r_norm),
                               np.dot(e_vec, r)/(e*r_norm))
    else:
        # circular: use node vector to find reference for true anomaly
        if n_norm != 0:
            true_anom = math.atan2(np.dot(np.cross(n_vec, r), h)/(h_norm*n_norm),
                                   np.dot(n_vec, r)/(n_norm))
        else:
            true_anom = 0.0
    # normalize angles [0,2pi)
    def norm_ang(x):
        return x % (2*math.pi)
    RAAN = norm_ang(RAAN)
    arg_perigee = norm_ang(arg_perigee)
    true_anom = norm_ang(true_anom)
    return {
        'a': a, 'e': e, 'i': i,
        'RAAN': RAAN, 'arg_perigee': arg_perigee, 'true_anomaly': true_anom,
        'h_norm': h_norm, 'energy': energy
    }

# ---------------------------
# TLE -> state using sgp4 (optional)
# ---------------------------
def tle_to_state(line1, line2):
    """Return (r_m, v_m_s, epoch_datetime) using sgp4. Raises if sgp4 not installed or error."""
    if not SGP4_AVAILABLE:
        raise RuntimeError("sgp4 not installed. Install with `pip install sgp4` to use TLE input.")
    sat = Satrec.twoline2rv(line1, line2)
    jd = sat.jdsatepoch + sat.jdsatepochF
    jd_int = int(jd)
    fr = jd - jd_int
    e, r_km, v_km_s = sat.sgp4(jd_int, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error code {e}")
    r_m = np.array(r_km) * 1000.0
    v_m_s = np.array(v_km_s) * 1000.0
    epoch_dt = julian_day_to_datetime(jd)
    return r_m, v_m_s, epoch_dt

def julian_day_to_datetime(jd):
    # convert JD -> datetime (UTC)
    jd += 0.5
    Z = int(jd)
    F = jd - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25)/36524.25)
        A = Z + 1 + alpha - int(alpha/4)
    B = A + 1524
    C = int((B - 122.1)/365.25)
    D = int(365.25*C)
    E = int((B - D)/30.6001)
    day = B - D - int(30.6001*E) + F
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        year = C - 4716
    else:
        year = C - 4715
    day_int = int(day)
    frac = day - day_int
    hours = frac * 24.0
    hh = int(hours)
    minutes = (hours - hh) * 60.0
    mm = int(minutes)
    seconds = (minutes - mm) * 60.0
    ss = seconds
    try:
        dt = datetime(year, month, day_int, hh, mm, int(ss), int((ss - int(ss))*1e6))
    except Exception:
        dt = datetime.utcnow()
    return dt

# ---------------------------
# File I/O helpers
# ---------------------------
def read_batch_states_file(path):
    entries = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            vals = list(map(float, parts[:6]))
            r = np.array(vals[0:3], dtype=float)
            v = np.array(vals[3:6], dtype=float)
            entries.append((r, v))
    return entries

def read_tle_file(path):
    with open(path, 'r') as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    sats = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('1 ') or lines[i].startswith('2 '):
            l1 = lines[i]
            l2 = lines[i+1] if i+1 < len(lines) else ""
            name = f"SAT_{len(sats)}"
            sats.append((name, l1, l2))
            i += 2
        else:
            name = lines[i]
            l1 = lines[i+1] if i+1 < len(lines) else ""
            l2 = lines[i+2] if i+2 < len(lines) else ""
            sats.append((name, l1, l2))
            i += 3
    return sats

def save_results_csv(out_path, history_list):
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["sat_name", "epoch_utc", "t_offset_s", "rx_m", "ry_m", "rz_m", "vx_m_s", "vy_m_s", "vz_m_s",
                  "a_m", "e", "i_rad", "RAAN_rad", "arg_perigee_rad", "true_anomaly_rad"]
        writer.writerow(header)
        for sat in history_list:
            name = sat.get('name', '')
            epoch = sat.get('epoch', '')
            rows = sat.get('history', [])
            for t_offset, r, v in rows:
                elems = state_to_orbital(r, v)
                writer.writerow([name, epoch.isoformat() if hasattr(epoch, 'isoformat') else epoch,
                                 f"{t_offset:.6f}",
                                 f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}",
                                 f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}",
                                 f"{elems['a']:.6f}" if np.isfinite(elems['a']) else "",
                                 f"{elems['e']:.6f}",
                                 f"{elems['i']:.6f}",
                                 f"{elems['RAAN']:.6f}",
                                 f"{elems['arg_perigee']:.6f}",
                                 f"{elems['true_anomaly']:.6f}"])
    return out_path

# ---------------------------
# GUI Application
# ---------------------------
class OrbitGUI:
    def __init__(self, master):
        self.master = master
        master.title("Orbit Propagator Calculator (RK4)")

        self.dt_var = tk.DoubleVar(value=10.0)
        self.steps_var = tk.IntVar(value=600)
        self.record_every_var = tk.IntVar(value=1)
        self.mu_var = tk.DoubleVar(value=MU_EARTH)
        self.use_tle_var = tk.BooleanVar(value=SGP4_AVAILABLE)
        self.batch_list = []  # dicts: name, r0, v0, epoch
        self.history_results = []

        # layout frames
        top = ttk.Frame(master); top.pack(fill=tk.X, padx=6, pady=6)
        frm_input = ttk.LabelFrame(top, text="Input / Parameters"); frm_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frm_buttons = ttk.Frame(top); frm_buttons.pack(side=tk.RIGHT, padx=6)

        # input widgets
        ttk.Checkbutton(frm_input, text="Use TLE (sgp4)", variable=self.use_tle_var).grid(row=0, column=0, sticky="w")
        ttk.Button(frm_input, text="Load TLE file...", command=self.load_tle_file).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(frm_input, text="Load state vectors file...", command=self.load_states_file).grid(row=0, column=2, padx=4, pady=2)

        ttk.Label(frm_input, text="dt (s):").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm_input, textvariable=self.dt_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(frm_input, text="steps:").grid(row=1, column=2, sticky="e")
        ttk.Entry(frm_input, textvariable=self.steps_var, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(frm_input, text="record every N step:").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm_input, textvariable=self.record_every_var, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(frm_input, text="mu (m^3/s^2):").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm_input, textvariable=self.mu_var, width=16).grid(row=2, column=3, sticky="w")

        # buttons
        ttk.Button(frm_buttons, text="Run Batch", command=self.run_batch).pack(fill=tk.X, pady=2)
        ttk.Button(frm_buttons, text="Plot Results", command=self.plot_results).pack(fill=tk.X, pady=2)
        ttk.Button(frm_buttons, text="Export CSV", command=self.export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(frm_buttons, text="Clear", command=self.clear_all).pack(fill=tk.X, pady=2)

        # status and log
        frm_status = ttk.Frame(master); frm_status.pack(fill=tk.X, padx=6)
        self.progress = ttk.Progressbar(frm_status, orient='horizontal', mode='determinate'); self.progress.pack(fill=tk.X, padx=4, pady=2)
        self.log_text = tk.Text(master, height=10); self.log_text.pack(fill=tk.BOTH, padx=6, pady=6)

        # plot area
        frm_plot = ttk.LabelFrame(master, text="Trajectory Visualization"); frm_plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.fig = plt.Figure(figsize=(9,5))
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax2d = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.log("Ready. sgp4 available: {}".format(SGP4_AVAILABLE))

    def log(self, *args):
        s = " ".join(str(a) for a in args)
        ts = datetime.utcnow().isoformat()
        self.log_text.insert(tk.END, f"[{ts}] {s}\n")
        self.log_text.see(tk.END)

    def load_tle_file(self):
        if not SGP4_AVAILABLE:
            messagebox.showwarning("sgp4 not installed", "sgp4 not installed. Install with `pip install sgp4` to parse TLE automatically.")
        path = filedialog.askopenfilename(title="Select TLE file", filetypes=[("Text files","*.tle *.txt *.dat"),("All files","*.*")])
        if not path:
            return
        sats = read_tle_file(path)
        self.batch_list = []
        for (name, l1, l2) in sats:
            if SGP4_AVAILABLE:
                try:
                    r, v, epoch = tle_to_state(l1, l2)
                except Exception as e:
                    self.log(f"Failed to convert TLE {name}: {e}")
                    continue
            else:
                self.log(f"Skipping TLE {name} (sgp4 not installed).")
                continue
            self.batch_list.append({'name': name, 'r0': r, 'v0': v, 'epoch': epoch})
        self.log(f"Loaded {len(self.batch_list)} from TLE file: {os.path.basename(path)}")

    def load_states_file(self):
        path = filedialog.askopenfilename(title="Select states file", filetypes=[("Text files","*.txt *.dat *.csv"),("All files","*.*")])
        if not path:
            return
        entries = read_batch_states_file(path)
        self.batch_list = []
        for idx, (r, v) in enumerate(entries):
            self.batch_list.append({'name': f"SAT_{idx}", 'r0': r, 'v0': v, 'epoch': datetime.utcnow()})
        self.log(f"Loaded {len(self.batch_list)} states from: {os.path.basename(path)}")

    def run_batch(self):
        if not self.batch_list:
            messagebox.showerror("No input", "No TLE or state vectors loaded.")
            return
        try:
            dt = float(self.dt_var.get())
            steps = int(self.steps_var.get())
            record_every = int(self.record_every_var.get())
            mu = float(self.mu_var.get())
        except Exception as e:
            messagebox.showerror("Invalid parameters", f"Check dt/steps/record_every/mu: {e}")
            return
        thread = threading.Thread(target=self._run_batch_thread, args=(dt, steps, record_every, mu), daemon=True)
        thread.start()

    def _run_batch_thread(self, dt, steps, record_every, mu):
        total = len(self.batch_list)
        self.history_results = []
        self.progress['value'] = 0
        self.progress['maximum'] = total
        for idx, sat in enumerate(self.batch_list):
            name = sat['name']
            r0 = sat['r0']
            v0 = sat['v0']
            epoch = sat.get('epoch', datetime.utcnow())
            self.log(f"Propagating {name} (epoch {epoch.isoformat()}) ...")
            hist = propagate_rk4_state(r0, v0, dt, steps, mu=mu, record_every=record_every)
            self.history_results.append({'name': name, 'epoch': epoch, 'history': hist})
            self.progress['value'] = idx + 1
            self.master.after(1, lambda: None)
        self.log("Batch propagation finished for {} satellites.".format(len(self.history_results)))
        messagebox.showinfo("Done", "Batch propagation completed.")
        self.master.after(10, self.plot_results)

    def plot_results(self):
        if not self.history_results:
            messagebox.showwarning("No results", "No results to plot. Run batch propagation first.")
            return
        self.ax3d.cla()
        self.ax2d.cla()
        cmap = plt.cm.get_cmap('tab10')
        for idx, sat in enumerate(self.history_results):
            name = sat['name']
            hist = sat['history']
            rs = np.array([row[1] for row in hist])
            self.ax3d.plot(rs[:,0], rs[:,1], rs[:,2], label=name, color=cmap(idx%10))
            self.ax2d.plot(rs[:,0], rs[:,1], label=name, color=cmap(idx%10))
            self.ax3d.scatter(rs[0,0], rs[0,1], rs[0,2], marker='o', s=20, color=cmap(idx%10))
            self.ax3d.scatter(rs[-1,0], rs[-1,1], rs[-1,2], marker='x', s=20, color=cmap(idx%10))
            self.ax2d.scatter(rs[0,0], rs[0,1], marker='o', s=20, color=cmap(idx%10))
            self.ax2d.scatter(rs[-1,0], rs[-1,1], marker='x', s=20, color=cmap(idx%10))
        self.ax3d.set_title("3D Trajectories (m)")
        self.ax3d.set_xlabel("x (m)"); self.ax3d.set_ylabel("y (m)"); self.ax3d.set_zlabel("z (m)")
        self.ax2d.set_title("XY Projection"); self.ax2d.set_xlabel("x (m)"); self.ax2d.set_ylabel("y (m)")
        self.ax2d.axis('equal')
        self.ax3d.legend(); self.ax2d.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def export_csv(self):
        if not self.history_results:
            messagebox.showwarning("No results", "No results to export.")
            return
        path = filedialog.asksaveasfilename(title="Save results CSV", defaultextension=".csv", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            return
        save_results_csv(path, self.history_results)
        self.log(f"Saved results to {path}")
        messagebox.showinfo("Saved", f"Results saved to:\n{path}")

    def clear_all(self):
        self.batch_list = []
        self.history_results = []
        self.progress['value'] = 0
        self.ax3d.cla(); self.ax2d.cla(); self.canvas.draw()
        self.log_text.delete(1.0, tk.END)
        self.log("Cleared all data.")

# ---------------------------
# Main
# ---------------------------
def main():
    root = tk.Tk()
    app = OrbitGUI(root)
    root.geometry("1100x700")
    root.mainloop()

if __name__ == "__main__":
    main()
