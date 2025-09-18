"""
Handles all file Input/Output operations and TLE parsing.
- TLE parsing using the sgp4 library.
- Reading of initial state vector files.
- Writing propagation results to CSV files.
"""

import csv
import numpy as np

# Local application imports
from .utils import julian_day_to_datetime, state_to_orbital_elements

# Conditionally import sgp4 and set a flag
try:
    from sgp4.api import Satrec
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

def tle_to_state(line1, line2):
    """
    Converts TLE (Two-Line Element) set to a state vector.

    This function requires the 'sgp4' library to be installed.

    Args:
        line1 (str): The first line of the TLE.
        line2 (str): The second line of the TLE.

    Returns:
        tuple: A tuple containing (r_vector_m, v_vector_m_s, epoch_datetime).
               Returns (None, None, None) if sgp4 is not available or fails.

    Raises:
        RuntimeError: If sgp4 is not installed or if there is an SGP4 error.
    """
    if not SGP4_AVAILABLE:
        raise RuntimeError("SGP4 library not installed. Please run 'pip install sgp4' to use TLE functionality.")

    satellite = Satrec.twoline2rv(line1, line2)

    # Get the TLE epoch in Julian Day format
    jd_epoch = satellite.jdsatepoch + satellite.jdsatepochF

    # Propagate to the TLE epoch to get the state vector
    error_code, r_km, v_km_s = satellite.sgp4(jd_epoch, 0) # jd_int, jd_frac

    if error_code != 0:
        # See https://rhodesmill.org/sgp4/api-python.html#error-codes
        raise RuntimeError(f"SGP4 propagation error, code: {error_code}")

    # Convert from km and km/s to m and m/s
    r_m = np.array(r_km) * 1000.0
    v_m_s = np.array(v_km_s) * 1000.0

    epoch_dt = julian_day_to_datetime(jd_epoch)

    return r_m, v_m_s, epoch_dt

def read_batch_states_file(path):
    """
    Reads a file with initial state vectors (r_x, r_y, r_z, v_x, v_y, v_z).

    Args:
        path (str): The path to the file.

    Returns:
        list: A list of tuples, each containing (r_vector, v_vector).
    """
    entries = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue

            try:
                parts = s.split()
                if len(parts) < 6:
                    continue
                vals = list(map(float, parts[:6]))
                r = np.array(vals[0:3], dtype=float)
                v = np.array(vals[3:6], dtype=float)
                entries.append((r, v))
            except (ValueError, IndexError):
                # Silently skip malformed lines
                continue
    return entries

def read_tle_file(path):
    """
    Reads a TLE file, which can contain multiple 2-line or 3-line entries.

    Args:
        path (str): The path to the TLE file.

    Returns:
        list: A list of tuples, each containing (satellite_name, line1, line2).
    """
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    sats = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Heuristic to find the start of a TLE entry
        is_line1 = line.startswith('1 ') and len(line) == 69
        is_line2 = line.startswith('2 ') and len(line) == 69

        if is_line1 and i + 1 < len(lines) and lines[i+1].startswith('2 '):
            # Case 1: 2-line TLE
            name = f"SAT_{len(sats) + 1}"
            sats.append((name, lines[i], lines[i+1]))
            i += 2
        elif not is_line1 and not is_line2 and i + 2 < len(lines) and \
             lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
            # Case 2: 3-line TLE (name + 2 lines)
            name = line
            sats.append((name, lines[i+1], lines[i+2]))
            i += 3
        else:
            # Unrecognized format, advance to the next line
            i += 1

    return sats

def save_results_csv(out_path, history_list, mu):
    """
    Saves the results of one or more propagations to a CSV file.

    Args:
        out_path (str): The path for the output CSV file.
        history_list (list): A list of result dictionaries. Each dict should have
                             'name', 'epoch', and 'history' keys.
        mu (float): The gravitational parameter used in the simulation.
    """
    header = [
        "sat_name", "epoch_utc", "t_offset_s",
        "rx_m", "ry_m", "rz_m",
        "vx_m_s", "vy_m_s", "vz_m_s",
        "a_m", "e", "i_rad", "RAAN_rad", "arg_perigee_rad", "true_anomaly_rad"
    ]

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for sat_result in history_list:
            name = sat_result.get('name', 'UNKNOWN')
            epoch = sat_result.get('epoch', '')
            history = sat_result.get('history', [])

            epoch_str = epoch.isoformat() if hasattr(epoch, 'isoformat') else str(epoch)

            for t_offset, r, v in history:
                elems = state_to_orbital_elements(r, v, mu)
                row = [
                    name, epoch_str,
                    f"{t_offset:.3f}",
                    f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}",
                    f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.3f}",
                    f"{elems['a']:.3f}" if np.isfinite(elems['a']) else "inf",
                    f"{elems['e']:.8f}",
                    f"{elems['i']:.6f}",
                    f"{elems['RAAN']:.6f}",
                    f"{elems['arg_perigee']:.6f}",
                    f"{elems['true_anomaly']:.6f}"
                ]
                writer.writerow(row)

    return out_path
