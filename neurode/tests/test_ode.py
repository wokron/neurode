from neurode.tests.utils import generate_data
from neurode.ode import ODE
from neurode.calc.equation import Equations, Derivation as d
from neurode.calc.node import Placeholder
import numpy as np


def test_ode_calc():
    x = Placeholder("x")
    y = Placeholder("y")
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    gamma = Placeholder("gamma")
    delta = Placeholder("delta")

    ode = ODE(
        Equations(
            [
                d(x) == alpha * x - beta * x * y,
                d(y) == gamma * x * y - delta * y,
            ]
        )
    )

    assert ode.params == {"alpha": 0, "beta": 0, "gamma": 0, "delta": 0}

    ode.params.update({"alpha": 2, "beta": 0.02, "gamma": 0.0002, "delta": 0.8})

    t = np.linspace(0, 10)
    val = ode.calc([5000, 100], t)

    assert len(t) == len(val)
    assert (3000 <= val[:, 0]).all() and (val[:, 0] <= 5000).all()
    assert (80 <= val[:, 1]).all() and (val[:, 1] <= 120).all()


def test_ode_fit():
    x = Placeholder("x")
    y = Placeholder("y")
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    gamma = Placeholder("gamma")
    delta = Placeholder("delta")

    ode = ODE(
        Equations(
            [
                d(x) == alpha * x - beta * x * y,
                d(y) == gamma * x * y - delta * y,
            ]
        )
    )

    init_params = {"alpha": 2, "beta": 0.02, "gamma": 0.0002, "delta": 0.8}

    ode.params.update(init_params)

    y, t = generate_data()

    ode.fit(y, t, epoches=10)

    assert ode.params != init_params
