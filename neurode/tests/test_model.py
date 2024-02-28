from ..net.model import NeuroODE
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
    )

    # y_num = 2, batch_size = 1
    t = torch.tensor([0])
    y = torch.tensor([[5000], [100]])
    steps = torch.tensor([1])
    result = model(t, y, steps)
    assert result.shape == (2, 1)


def test_backward():
    model = NeuroODE(
        {
            "alpha": 2,
            "beta": 0.02,
            "gamma": 0.0002,
            "delta": 0.8,
        },
        ode_func,
    )

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)

    # y_num = 2, batch_size = 1
    t = torch.tensor([0], dtype=float)
    y = torch.tensor([[5000], [100]], dtype=float)
    steps = torch.tensor([1], dtype=float)
    y_next = torch.tensor([[4800], [120]], dtype=float)

    result = model.forward(t, y, steps)
    loss = loss_fn(result, y_next)

    assert loss.item() != 0

    optim.zero_grad()
    loss.backward()
    optim.step()
