from typing import Any
from .calc_node import Placeholder, CalcNode


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

    def calc(self, y0: list[Any], params: dict[str, Any]):
        data: dict[str, Any] = {}
        for no, (var_name, _) in enumerate(self.variables):
            data[var_name] = y0[no]

        data.update(params)
        val: list[Any] = []

        for equation in self.equations:
            val.append(equation.expression(data))

        return val
