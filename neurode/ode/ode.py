from typing import Any

from neurode.net.model import NeuroODE
from neurode.net.trainer import ODETrainer
from ..calc import Equations
from scipy.integrate import odeint


class ODE:
    def __init__(self, equations: Equations) -> None:
        self.equations = equations
        self.params: dict[str, Any] = {}

        for param in self.equations.params:
            self.params[param.name] = 0

    def calc(self, y0, t):
        return odeint(self.__calc_derivative, y0, t, args=(self.params,))

    def fit(self, y, t, device="cpu", epoches=100, lr=1e-4, max_step=0.2):
        trainer = ODETrainer(
            device=device,
            epoches=epoches,
            lr=lr,
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
