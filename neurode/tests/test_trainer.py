from neurode.net.model import NeuroODE
from neurode.net.trainer import ODETrainer
from neurode.tests.utils import generate_data
import torch


def ode_func(y0, t, params):
    x, y = y0
    dx = params["alpha"] * x - params["beta"] * x * y
    dy = params["gamma"] * x * y - params["delta"] * y
    return [dx, dy]


def test_trainer():
    y, t = generate_data()

    trainer = ODETrainer(
        device=torch.device("cpu"),
        epoches=10,
        lr=1e-6,
    )

    init_params = {
        "alpha": 2,
        "beta": 0.02,
        "gamma": 0.0002,
        "delta": 0.8,
    }

    model = NeuroODE(
        init_params,
        ode_func,
        max_step=0.01,
    )

    trainer.train(model, t, y)

    assert model.get_params() != init_params
