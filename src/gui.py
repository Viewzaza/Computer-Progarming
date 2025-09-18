"""
Main GUI for the Orbit Propagator application.

This module builds the user interface using tkinter and integrates the backend
logic from the other modules (physics, io_handler, utils).
"""

import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Local application imports
from .physics import MU_EARTH, propagate_rk4
from .io_handler import (
    SGP4_AVAILABLE, tle_to_state, read_tle_file,
    read_batch_states_file, save_results_csv
)

class OrbitPropagatorGUI:
    def __init__(self, master):
        """Initializes the main GUI window."""
        self.master = master
        master.title("Orbit Propagator Calculator (RK4)")
        master.geometry("1100x700")

        # --- State Variables ---
        self.batch_list = []  # List of dicts: {'name', 'r0', 'v0', 'epoch'}
        self.history_results = [] # List of propagation results

        # --- Tkinter Variables ---
        self.dt_var = tk.DoubleVar(value=10.0)
        self.steps_var = tk.IntVar(value=600)
        self.record_every_var = tk.IntVar(value=10)
        self.mu_var = tk.DoubleVar(value=MU_EARTH)

        # --- Build GUI ---
        self._create_widgets()

        self.log(f"Welcome! SGP4 library available: {SGP4_AVAILABLE}")
        if not SGP4_AVAILABLE:
            self.log("Install 'sgp4' (pip install sgp4) to enable TLE parsing.")

    def _create_widgets(self):
        """Creates and lays out all the GUI widgets."""
        # --- Top Frame for Inputs and Controls ---
        top_frame = ttk.Frame(self.master)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        input_frame = ttk.LabelFrame(top_frame, text="Input & Parameters")
        input_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        controls_frame = ttk.LabelFrame(top_frame, text="Controls")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # --- Input Widgets ---
        ttk.Button(input_frame, text="Load TLE File...", command=self.load_tle_file).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(input_frame, text="Load States File...", command=self.load_states_file).grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        ttk.Label(input_frame, text="dt (s):").grid(row=1, column=0, sticky="e", padx=5)
        ttk.Entry(input_frame, textvariable=self.dt_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(input_frame, text="Steps:").grid(row=1, column=2, sticky="e", padx=5)
        ttk.Entry(input_frame, textvariable=self.steps_var, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(input_frame, text="Record every:").grid(row=2, column=0, sticky="e", padx=5)
        ttk.Entry(input_frame, textvariable=self.record_every_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(input_frame, text="μ (m³/s²):").grid(row=2, column=2, sticky="e", padx=5)
        ttk.Entry(input_frame, textvariable=self.mu_var, width=18).grid(row=2, column=3, sticky="w")

        # --- Control Buttons ---
        ttk.Button(controls_frame, text="Run Propagation", command=self.run_propagation).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(controls_frame, text="Plot Results", command=self.plot_results).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(controls_frame, text="Export CSV", command=self.export_csv).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(controls_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, padx=5, pady=2)

        # --- Status and Log Area ---
        status_frame = ttk.Frame(self.master)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.progress = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=2)

        log_frame = ttk.LabelFrame(self.master, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=8)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # --- Plot Area ---
        plot_frame = ttk.LabelFrame(self.master, text="Trajectory Visualization")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = plt.Figure(figsize=(10, 6))
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax2d = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Appends a timestamped message to the log widget."""
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        self.log_text.insert(tk.END, f"[{ts} UTC] {message}\n")
        self.log_text.see(tk.END)

    def load_tle_file(self):
        if not SGP4_AVAILABLE:
            messagebox.showwarning("SGP4 Not Found", "The 'sgp4' library is required to parse TLE files. Please install it (`pip install sgp4`).")
            return
        path = filedialog.askopenfilename(title="Select TLE File", filetypes=[("Text files", "*.tle *.txt"), ("All files", "*.*")])
        if not path: return

        try:
            sats = read_tle_file(path)
            if not sats:
                self.log(f"No valid TLE entries found in {os.path.basename(path)}.")
                return

            self.clear_all(keep_log=True)
            for name, l1, l2 in sats:
                try:
                    r0, v0, epoch = tle_to_state(l1, l2)
                    self.batch_list.append({'name': name, 'r0': r0, 'v0': v0, 'epoch': epoch})
                except Exception as e:
                    self.log(f"Error parsing TLE for {name}: {e}")
            self.log(f"Loaded {len(self.batch_list)} satellites from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("File Read Error", f"Failed to read or parse TLE file:\n{e}")
            self.log(f"Error loading TLE file: {e}")

    def load_states_file(self):
        path = filedialog.askopenfilename(title="Select State Vectors File", filetypes=[("Text files", "*.txt *.csv"), ("All files", "*.*")])
        if not path: return

        try:
            entries = read_batch_states_file(path)
            if not entries:
                self.log(f"No valid state vectors found in {os.path.basename(path)}.")
                return

            self.clear_all(keep_log=True)
            for i, (r, v) in enumerate(entries):
                self.batch_list.append({'name': f"State_{i+1}", 'r0': r, 'v0': v, 'epoch': datetime.utcnow()})
            self.log(f"Loaded {len(self.batch_list)} states from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("File Read Error", f"Failed to read state vector file:\n{e}")
            self.log(f"Error loading state vector file: {e}")

    def run_propagation(self):
        if not self.batch_list:
            messagebox.showerror("Input Error", "No initial states loaded. Please load a TLE or state vector file.")
            return
        try:
            dt = self.dt_var.get()
            steps = self.steps_var.get()
            record_every = self.record_every_var.get()
            mu = self.mu_var.get()
            if dt <= 0 or steps <= 0 or record_every <= 0:
                raise ValueError("dt, steps, and record_every must be positive.")
        except Exception as e:
            messagebox.showerror("Invalid Parameters", f"Please check your input parameters.\nError: {e}")
            return

        # Run in a separate thread to keep the GUI responsive
        thread = threading.Thread(target=self._run_propagation_thread, args=(dt, steps, record_every, mu), daemon=True)
        thread.start()

    def _run_propagation_thread(self, dt, steps, record_every, mu):
        self.history_results = []
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.batch_list)

        for i, sat in enumerate(self.batch_list):
            self.log(f"Propagating '{sat['name']}'...")
            hist = propagate_rk4(sat['r0'], sat['v0'], dt, steps, mu, record_every)
            self.history_results.append({'name': sat['name'], 'epoch': sat['epoch'], 'history': hist})
            self.master.after(0, lambda: self.progress.step())

        self.log("Batch propagation finished for all satellites.")
        self.master.after(100, lambda: messagebox.showinfo("Done", "Propagation complete. Results are ready to be plotted or exported."))
        self.master.after(100, self.plot_results) # Auto-plot after run

    def plot_results(self):
        if not self.history_results:
            messagebox.showwarning("No Data", "No results to plot. Please run a propagation first.")
            return

        self.ax3d.cla()
        self.ax2d.cla()
        cmap = plt.get_cmap('tab10')

        for i, sat_res in enumerate(self.history_results):
            color = cmap(i % 10)
            name = sat_res['name']
            history = sat_res['history']
            if not history: continue

            rs = np.array([row[1] for row in history])

            # 3D Plot
            self.ax3d.plot(rs[:, 0], rs[:, 1], rs[:, 2], label=name, color=color)
            self.ax3d.scatter(rs[0, 0], rs[0, 1], rs[0, 2], marker='o', s=30, color=color, label=f'{name} Start')

            # 2D Plot (XY Projection)
            self.ax2d.plot(rs[:, 0], rs[:, 1], label=name, color=color)
            self.ax2d.scatter(rs[0, 0], rs[0, 1], marker='o', s=30, color=color)

        # Earth sphere for scale
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        r_earth = 6371e3
        x = r_earth * np.outer(np.cos(u), np.sin(v))
        y = r_earth * np.outer(np.sin(u), np.sin(v))
        z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax3d.plot_surface(x, y, z, color='blue', alpha=0.3)

        self.ax3d.set_title("3D Trajectory")
        self.ax3d.set_xlabel("X (m)"); self.ax3d.set_ylabel("Y (m)"); self.ax3d.set_zlabel("Z (m)")
        self.ax3d.legend()

        self.ax2d.set_title("2D Projection (X-Y)")
        self.ax2d.set_xlabel("X (m)"); self.ax2d.set_ylabel("Y (m)")
        self.ax2d.grid(True); self.ax2d.axis('equal')

        self.fig.tight_layout()
        self.canvas.draw()
        self.log("Plotted results.")

    def export_csv(self):
        if not self.history_results:
            messagebox.showwarning("No Data", "No results to export. Please run a propagation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"orbit_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not path: return

        try:
            mu = self.mu_var.get()
            save_results_csv(path, self.history_results, mu)
            self.log(f"Successfully saved results to {path}")
            messagebox.showinfo("Export Successful", f"Results saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save results:\n{e}")
            self.log(f"Error exporting CSV: {e}")

    def clear_all(self, keep_log=False):
        self.batch_list = []
        self.history_results = []
        self.progress['value'] = 0
        self.ax3d.cla(); self.ax2d.cla()
        self.canvas.draw()
        if not keep_log:
            self.log_text.delete(1.0, tk.END)
            self.log("Cleared all data and plots.")
        else:
            self.log("Cleared previous data and plots.")
