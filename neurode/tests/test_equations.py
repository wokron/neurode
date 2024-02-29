from neurode.calc.equations import Derivation as d, Equations
from neurode.calc.calc_node import Placeholder as var
import torch


def test_equations_calculation():
    """
    dA/dt = A * 2 * alpha

    dB/dt = B * 3 + beta
    """

    A = var("A")
    B = var("B")
    alpha = var("alpha")
    beta = var("beta")

    equations = Equations(
        [
            d(A) == A * 2 * alpha,
            d(B) == B * 3 + beta,
        ]
    )

    val = equations.calc([10, 20], {"alpha": 1, "beta": 2})
    assert val == [20, 62]


def test_equations_tensor_calculation():
    """
    dA/dt = A * 2 * alpha

    dB/dt = B * 3 + beta
    """

    A = var("A")
    B = var("B")
    alpha = var("alpha")
    beta = var("beta")

    equations = Equations(
        [
            d(A) == A * 2 * alpha,
            d(B) == B * 3 + beta,
        ]
    )

    val = equations.calc(
        torch.tensor([[10, 20], [10, 20]]).transpose(0, 1),
        {"alpha": torch.tensor(1), "beta": torch.tensor(2)},
    )
    assert (
        torch.stack(val, dim=0) == torch.tensor([[20, 62], [20, 62]]).transpose(0, 1)
    ).all()
