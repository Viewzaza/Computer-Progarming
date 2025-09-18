"""
Core physics calculations for the orbit propagator.
- Two-body gravitational acceleration
- RK4 numerical integrator
"""

import numpy as np

# ---------------------------
# Constants (SI units)
# ---------------------------
MU_EARTH = 3.986004418e14  # Earth's gravitational parameter, m^3 / s^2

# ---------------------------
# Physics: two-body & RK4
# ---------------------------

def accel_two_body(r, mu=MU_EARTH):
    """
    Calculates acceleration based on the two-body equation of motion.

    Args:
        r (np.array): Position vector (m).
        mu (float): Gravitational parameter of the central body (m^3/s^2).

    Returns:
        np.array: Acceleration vector (m/s^2).
    """
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.zeros(3)
    return -mu * r / (r_norm**3)

def state_derivative(state, mu=MU_EARTH):
    """
    Calculates the derivative of the state vector [r, v].

    Args:
        state (np.array): State vector [rx, ry, rz, vx, vy, vz].
        mu (float): Gravitational parameter.

    Returns:
        np.array: Derivative of the state vector [vx, vy, vz, ax, ay, az].
    """
    r = state[0:3]
    v = state[3:6]
    a = accel_two_body(r, mu)
    return np.hstack((v, a))

def rk4_step(state, dt, mu=MU_EARTH):
    """
    Performs a single Runge-Kutta 4th order integration step.
    (Note: Renamed from rk4_step_state for clarity in this module)

    Args:
        state (np.array): Current state vector.
        dt (float): Time step (s).
        mu (float): Gravitational parameter.

    Returns:
        np.array: New state vector after the time step.
    """
    k1 = state_derivative(state, mu)
    k2 = state_derivative(state + 0.5 * dt * k1, mu)
    k3 = state_derivative(state + 0.5 * dt * k2, mu)
    k4 = state_derivative(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def propagate_rk4(r0, v0, dt, steps, mu=MU_EARTH, record_every=1):
    """
    Propagates an orbit using the RK4 numerical integrator.

    Args:
        r0 (np.array): Initial position vector (m).
        v0 (np.array): Initial velocity vector (m/s).
        dt (float): Time step (s).
        steps (int): Total number of integration steps.
        mu (float): Gravitational parameter.
        record_every (int): The frequency of recording history (e.g., 1 = every step).

    Returns:
        list: A list of tuples, where each tuple contains
              (time_offset_s, r_vector_m, v_vector_m_s).
    """
    state = np.hstack((r0, v0)).astype(float)
    history = []

    # Record initial state
    if 0 % record_every == 0:
         history.append((0.0, state[0:3].copy(), state[3:6].copy()))

    t = 0.0
    for n in range(1, steps + 1):
        state = rk4_step(state, dt, mu)
        t = n * dt
        if n % record_every == 0:
            history.append((t, state[0:3].copy(), state[3:6].copy()))

    return history
