from typing import Any

from neurode.calc.node import Placeholder, CalcNode


class Derivation:
    def __init__(self, variable: Placeholder) -> None:
        self.variable = variable

    def __eq__(self, other: CalcNode):
        return Equation(self, other)


class Equation:
    def __init__(self, derivation: Derivation, expression: CalcNode) -> None:
        self.derivation = derivation
        self.expression = expression


class Equations:
    def __init__(self, equation_list: list[Equation]) -> None:
        self.equations = equation_list

        placeholders: set[Placeholder] = set()
        variables: list[Placeholder] = []
        for equation in equation_list:
            variable = equation.derivation.variable
            variables.append(variable)
            placeholders.update(set(equation.expression.placeholders))

        self.params = placeholders - set(variables)
        self.variables = [(var.name, var) for var in variables]

    def calc(self, y: list[Any], params: dict[str, Any]):
        data: dict[str, Any] = {}
        for no, (var_name, _) in enumerate(self.variables):
            data[var_name] = y[no]

        data.update(params)
        val: list[Any] = []

        for equation in self.equations:
            val.append(equation.expression(data))

        return val

    def get_ode_fn(self):
        def ode_fn(y, t, params: dict[str, Any]):
            params.update({"t": t})
            return self.calc(y, params)

        return ode_fn
