from neurode import ODE, derivate as d, var

x1 = var("x1")
x2 = var("x2")
r1 = var("r1")
r2 = var("r2")
a1 = var("a1")
a2 = var("a2")
n1 = var("n1")
n2 = var("n2")

ode = ODE(
    [
        d(x1) == r1 * x1 * (1 - x1 / n1) - a1 * x1 * x2,
        d(x2) == r2 * x2 * (1 - x2 / n2) - a2 * x1 * x2,
    ]
)

t = [0, 10, 15, 30, 36, 40, 42]
y = [
    [100, 150],
    [165, 283],
    [197, 290],
    [280, 276],
    [305, 269],
    [318, 266],
    [324, 264],
]

ode.params = {
    "r1": 0.1,
    "r2": 0.1,
    "a1": 0.001,
    "a2": 0.001,
    "n1": 1000,
    "n2": 1000,
}
ode.fit(t, y, verbose=True)

print(ode.params)
