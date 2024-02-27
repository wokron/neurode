from ..calc.equations import Derivation as d, Equations
from ..calc.calc_node import Placeholder as var


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
