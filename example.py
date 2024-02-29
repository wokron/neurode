import matplotlib.pyplot as plt
import numpy as np

from neurode import ODE, derivate as d, var

x1 = var("x1")
x2 = var("x2")
r1 = var("r1")
r2 = var("r2")
a1 = var("a1")
a2 = var("a2")
in1 = var("in1")
in2 = var("in2")

ode = ODE(
    [
        d(x1) == r1 * x1 * (1 - x1 * in1) - a1 * x1 * x2,
        d(x2) == r2 * x2 * (1 - x2 * in2) - a2 * x1 * x2,
    ]
)

t = np.array([0, 10, 15, 30, 36, 40, 42])
y = np.array([
    [100, 150],
    [165, 283],
    [197, 290],
    [280, 276],
    [305, 269],
    [318, 266],
    [324, 264],
])

ode.params = {
    "r1": 0.1,
    "r2": 0.1,
    "a1": 0.001,
    "a2": 0.001,
    "in1": 0.001,
    "in2": 0.001,
}
ode.fit(t, y, verbose=True, epoches=5000)
print(ode.params)

t2 = np.linspace(0, 50)
y2 = ode.calc(t2, y[0])

plt.scatter(t, y[:, 0])
plt.scatter(t, y[:, 1])
plt.plot(t2, y2)
plt.show()
