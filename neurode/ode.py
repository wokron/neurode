from typing import Any

from scipy.integrate import odeint

from neurode.net.model import NeuroODE
from neurode.net.trainer import ODETrainer
from neurode.calc import Equations
from neurode.ode_next import ode_next_euler


class ODE:
    def __init__(self, equations: Equations | list = None, ode_fn=None) -> None:
        self.params: dict[str, Any] = {}

        if equations != None:
            if type(equations) != Equations:
                equations = Equations(equations)

            self.ode_fn = equations.get_ode_fn()

            for param in self.equations.params:
                self.params[param.name] = 0
        elif ode_fn != None:
            self.ode_fn = ode_fn
        else:
            raise ValueError("either equations or ode_fn must be provided")

    def int(self, t, y0):
        return odeint(self.ode_fn, y0, t, args=(self.params,))

    def fit(
        self,
        t,
        y,
        device="cpu",
        epoches=100,
        lr=1e-4,
        max_step=0.2,
        ode_next_fn=ode_next_euler,
        verbose=False,
    ):
        model = NeuroODE(
            self.params,
            self.ode_fn,
            max_step=max_step,
            ode_next_fn=ode_next_fn,
        )

        trainer = ODETrainer(
            device=device,
            epoches=epoches,
            lr=lr,
            verbose=verbose,
        )

        result = trainer.train(model, t, y)

        self.params.update(result["params"])

        return result
