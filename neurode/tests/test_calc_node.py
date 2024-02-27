from ..calc.calc_node import Placeholder


def test_basic_calculation():
    a = Placeholder("a")
    b = Placeholder("b")
    data = {"a": 1, "b": 2}

    # a + b = 3
    val = a + b
    assert val(data) == 3

    # a + 1 = 2
    val = a + 1
    assert val(data) == 2

    # a + (-1) = 0
    val = a + (-1)
    assert val(data) == 0

    # 1 + a = 2
    val = 1 + a
    assert val(data) == 2

    # a - b = -1
    val = a - b
    assert val(data) == -1

    # a * b = 2
    val = a * b
    assert val(data) == 2

    # a / b = 1/2
    val = a / b
    assert val(data) == 1 / 2
