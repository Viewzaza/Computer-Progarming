"""
Utility functions for orbital mechanics calculations and time conversions.
- State vector to orbital elements conversion
- Julian Date to datetime conversion
"""

import math
from datetime import datetime, timezone
import numpy as np

from .physics import MU_EARTH

def state_to_orbital_elements(r, v, mu=MU_EARTH):
    """
    Converts a state vector (position and velocity) to classical orbital elements.

    Args:
        r (np.array): Position vector (m).
        v (np.array): Velocity vector (m/s).
        mu (float): Gravitational parameter of the central body (m^3/s^2).

    Returns:
        dict: A dictionary containing the orbital elements:
              'a' (semi-major axis, m), 'e' (eccentricity), 'i' (inclination, rad),
              'RAAN' (Right Ascension of Ascending Node, rad),
              'arg_perigee' (Argument of Perigee, rad),
              'true_anomaly' (True Anomaly, rad),
              'h_norm' (specific angular momentum), 'energy'.
    """
    r = np.array(r, dtype=float)
    v = np.array(v, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Specific angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    # Eccentricity vector
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    # Specific mechanical energy
    energy = 0.5 * v_norm**2 - mu / r_norm

    # Semi-major axis
    if abs(energy) < 1e-12: # Parabolic orbit
        a = float('inf')
    else:
        a = -mu / (2 * energy)

    # Inclination
    i = math.acos(h[2] / h_norm) if h_norm > 1e-12 else 0.0

    # Node vector
    K = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(K, h)
    n_norm = np.linalg.norm(n_vec)

    # Right Ascension of the Ascending Node (RAAN)
    if n_norm > 1e-12:
        RAAN = math.atan2(n_vec[1], n_vec[0])
    else:
        RAAN = 0.0 # For equatorial orbits

    # Argument of Perigee
    if n_norm > 1e-12 and e > 1e-12:
        # General case for inclined, eccentric orbits
        arg_perigee = math.atan2(np.dot(e_vec, np.cross(h, n_vec)) / (h_norm * n_norm),
                                 np.dot(e_vec, n_vec) / n_norm)
    else:
        arg_perigee = 0.0 # Not well-defined for circular or equatorial orbits

    # True Anomaly
    if e > 1e-12:
        # General case for eccentric orbits
        true_anomaly = math.atan2(np.dot(r, np.cross(h, e_vec)) / (h_norm * e),
                                  np.dot(r, e_vec) / e)
    else:
        # For circular orbits, true anomaly is measured from the ascending node
        if n_norm > 1e-12:
            true_anomaly = math.atan2(np.dot(r, np.cross(h, n_vec)) / (h_norm * n_norm),
                                      np.dot(r, n_vec) / n_norm)
        else:
            true_anomaly = math.atan2(r[1], r[0]) # Equatorial circular, measured from x-axis

    # Normalize angles to be in [0, 2*pi)
    def normalize_angle(angle):
        return angle % (2 * math.pi)

    return {
        'a': a,
        'e': e,
        'i': i,
        'RAAN': normalize_angle(RAAN),
        'arg_perigee': normalize_angle(arg_perigee),
        'true_anomaly': normalize_angle(true_anomaly),
        'h_norm': h_norm,
        'energy': energy
    }

def julian_day_to_datetime(jd):
    """
    Converts a Julian Day to a Python datetime object (UTC).

    Args:
        jd (float): The Julian Day.

    Returns:
        datetime: The corresponding datetime object.
    """
    jd_ref = 2440587.5  # Julian day for 1970-01-01 00:00:00 UTC
    seconds_since_epoch = (jd - jd_ref) * 86400.0
    try:
        # More robust conversion
        return datetime.fromtimestamp(seconds_since_epoch, tz=timezone.utc)
    except (ValueError, OSError):
        # Fallback for very large or small dates that fromtimestamp might not handle
        # This logic is a simplified version of the one in the reference file
        jd += 0.5
        Z = int(jd)
        F = jd - Z
        if Z < 2299161:
            A = Z
        else:
            alpha = int((Z - 1867216.25) / 36524.25)
            A = Z + 1 + alpha - int(alpha / 4)
        B = A + 1524
        C = int((B - 122.1) / 365.25)
        D = int(365.25 * C)
        E = int((B - D) / 30.6001)
        day_frac = B - D - int(30.6001 * E) + F
        day = int(day_frac)
        if E < 14:
            month = E - 1
        else:
            month = E - 13
        if month > 2:
            year = C - 4716
        else:
            year = C - 4715

        frac_of_day = day_frac - day
        h = frac_of_day * 24
        m = (h - int(h)) * 60
        s = (m - int(m)) * 60
        us = (s - int(s)) * 1e6

        return datetime(year, month, day, int(h), int(m), int(s), int(us), tzinfo=timezone.utc)
