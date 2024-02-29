from neurode.net.model import NeuroODE
import torch


def ode_func(y0, t, params):
    x, y = y0
    dx = params["alpha"] * x - params["beta"] * x * y
    dy = params["gamma"] * x * y - params["delta"] * y
    return [dx, dy]


def test_forward():
    model = NeuroODE(
        {
            "alpha": 2,
            "beta": 0.02,
            "gamma": 0.0002,
            "delta": 0.8,
        },
        ode_func,
        max_step=0.5,
    )

    # batch_size = 1, y_num = 2
    t = torch.tensor([0, 1, 2, 3, 4, 5])
    y0 = torch.tensor([5000, 100])
    result = model(t, y0)
    assert result.shape == (6, 2)


def test_backward():
    model = NeuroODE(
        {
            "alpha": 2,
            "beta": 0.02,
            "gamma": 0.0002,
            "delta": 0.8,
        },
        ode_func,
        max_step=0.5,
    )

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)

    t = torch.tensor([0, 1, 2, 3, 4, 5], dtype=float)
    y0 = torch.tensor([5000, 100], dtype=float)
    expect = torch.ones((6, 2), dtype=float)

    result = model(t, y0)
    loss = loss_fn(result, expect)

    assert loss.item() != 0

    optim.zero_grad()
    loss.backward()
    optim.step()
