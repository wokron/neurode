from typing import Any
from ..calc import Equations
from scipy.integrate import odeint


class ODE:
    def __init__(self, equations: Equations) -> None:
        self.equations = equations
        self.params: dict[str, Any] = {}

        for param in self.equations.params:
            self.params[param.name] = 0

    def calc(self, y0, t):
        def ode_func(y, t, params: dict[str, Any]):
            params.update({"t": t})
            return self.equations.calc(y, params)

        return odeint(ode_func, y0, t, args=(self.params,))
