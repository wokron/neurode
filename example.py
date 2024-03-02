import matplotlib.pyplot as plt
import numpy as np

from neurode import ODE, derivate as d, var

S, E, I, R, D = var("S"), var("E"), var("I"), var("R"), var("D")
N = S + E + I + R + D
beta = var("beta")
sigma = var("sigma")
gamma = var("gamma")
alpha = var("alpha")
omega = var("omega")

ode = ODE(
    [
        d(S) == omega * R - beta * I * S / N,
        d(E) == beta * I * S / N - sigma * E,
        d(I) == sigma * E - gamma * I - alpha * I,
        d(R) == gamma * I - omega * R,
        d(D) == alpha * I,
    ]
)

t = np.linspace(0, 250)

ode.params = {
    "beta": 1,
    "sigma": 0.4,
    "gamma": 0.4,
    "alpha": 0.001,
    "omega": 0.01,
}
y = ode.calc(t, [100000, 10, 0, 0, 0])

ode.params = {
    "beta": 1,
    "sigma": 0.1,
    "gamma": 0.1,
    "alpha": 0.004,
    "omega": 0.04,
}
ode.fit(t, y, verbose=True, lr=4e-2, max_step=1, epoches=2000)
print(ode.params)

y2 = ode.calc(t, [100000, 10, 0, 0, 0])

plt.plot(t, y2)
plt.scatter(t, y[:, 0], label="S")
plt.scatter(t, y[:, 1], label="E")
plt.scatter(t, y[:, 2], label="I")
plt.scatter(t, y[:, 3], label="R")
plt.scatter(t, y[:, 4], label="D")
plt.legend()
plt.show()
