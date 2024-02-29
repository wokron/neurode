from typing import Any

from scipy.integrate import odeint

from neurode.net.model import NeuroODE
from neurode.net.trainer import ODETrainer
from neurode.calc import Equations


class ODE:
    def __init__(self, equations: Equations | list) -> None:
        if type(equations) != Equations:
            equations = Equations(equations)

        self.equations = equations
        self.params: dict[str, Any] = {}

        for param in self.equations.params:
            self.params[param.name] = 0

    def calc(self, y0, t):
        return odeint(self.__calc_derivative, y0, t, args=(self.params,))

    def fit(self, y, t, device="cpu", epoches=100, lr=1e-4, max_step=0.2, verbose=False):
        trainer = ODETrainer(
            device=device,
            epoches=epoches,
            lr=lr,
            verbose=verbose,
        )

        model = NeuroODE(
            self.params,
            self.__calc_derivative,
            max_step=max_step,
        )

        trainer.train(model, y, t)

        self.params.update(model.get_params())

    def __calc_derivative(self, y, t, params: dict[str, Any]):
        params.update({"t": t})
        return self.equations.calc(y, params)
