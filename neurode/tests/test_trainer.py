import numpy as np
from neurode.net.data import ODEDataset
from neurode.net.model import NeuroODE
from neurode.net.trainer import ODETrainer
import torch
from scipy.integrate import odeint


def ode_func(y0, t, params):
    x, y = y0
    dx = params["alpha"] * x - params["beta"] * x * y
    dy = params["gamma"] * x * y - params["delta"] * y
    return [dx, dy]


def generate_data():
    def ode_func2(y, t):
        x, y = y
        dx = 2 * x - 0.02 * x * y
        dy = 0.0002 * x * y - 0.8 * y
        return [dx, dy]

    t = np.linspace(0, 10, 500)
    return odeint(ode_func2, [5000, 120], t), t


def test_trainer():
    y, t = generate_data()

    trainer = ODETrainer(
        device=torch.device("cpu"),
        epoches=100,
        lr=1e-6,
    )

    model = NeuroODE(
        {
            "alpha": 2,
            "beta": 0.02,
            "gamma": 0.0002,
            "delta": 0.8,
        },
        ode_func,
    )

    trainer.train(model, y, t)
    pass
