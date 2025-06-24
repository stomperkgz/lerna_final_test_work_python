import numpy as np
from scipy.integrate import solve_ivp

g = 9.8

def y_analytical(x, v0, alpha_deg):
    alpha = np.radians(alpha_deg)
    return x * np.tan(alpha) - (g * x**2) / (2 * v0**2 * np.cos(alpha)**2)

def analytical_trajectory(v0, alpha_deg):
    alpha = np.radians(alpha_deg)
    t_flight = 2 * v0 * np.sin(alpha) / g
    x = np.linspace(0, v0 * np.cos(alpha) * t_flight, 500)
    y = y_analytical(x, v0, alpha_deg)
    return x, y

def projectile_rhs(t, Y, a, b):
    x, y, vx, vy = Y
    v = np.sqrt(vx**2 + vy**2)
    dvx = -a * vx - b * v * vx
    dvy = -1 - a * vy - b * v * vy
    return [vx, vy, dvx, dvy]

def simulate(alpha_deg, a_val, b_val, v0=1.0):
    alpha = np.radians(alpha_deg)
    vx0 = np.cos(alpha)
    vy0 = np.sin(alpha)

    def event_ground(t, Y):
        return Y[1]
    event_ground.terminal = True
    event_ground.direction = -1

    sol = solve_ivp(
        fun=lambda t, Y: projectile_rhs(t, Y, a_val, b_val),
        t_span=[0, 10],
        y0=[0, 0, vx0, vy0],
        dense_output=True,
        events=event_ground,
        max_step=0.05
    )
    return sol