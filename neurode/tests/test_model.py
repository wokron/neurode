from ..net.model import NeuroODE
import torch


def test_forward():
    def ode_func(y0, t, params):
        x, y = y0
        dx = params["alpha"] * x - params["beta"] * x * y
        dy = params["gamma"] * x * y - params["delta"] * y
        return [dx, dy]

    model = NeuroODE(
        {
            "alpha": 2,
            "beta": 0.02,
            "gamma": 0.0002,
            "delta": 0.8,
        },
        ode_func,
    )

    t = [0]
    y = [[5000], [100]]
    steps = [1]
    t = torch.tensor(t)
    y = torch.tensor(y)
    steps = torch.tensor(steps)
    result = model(t, y, steps)
    assert result.shape == (2, 1)
