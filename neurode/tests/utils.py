import numpy as np
from scipy.integrate import odeint


def generate_data():
    def ode_func2(y, t):
        x, y = y
        dx = 2 * x - 0.02 * x * y
        dy = 0.0002 * x * y - 0.8 * y
        return [dx, dy]

    t = np.linspace(0, 10, 500)
    return odeint(ode_func2, [5000, 120], t), t
