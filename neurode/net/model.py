from typing import Callable
from torch import nn


class NeuroODE(nn.Module):
    def __init__(
        self,
        params: dict[str, float],
        # (y: list[float], t: float, params: dict[str, float])
        ode_func: Callable[[list[float], float, dict[str, float]], list[float]],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        params_no: dict[str, int] = {}
        weights = []
        for no, param_name in enumerate(params):
            params_no[param_name] = no
            weights.append(params[param_name])

        self.params_no = params_no
        self.weights = nn.Parameter(weights, requires_grad=True)
        self.ode_func = ode_func

    def get_params(self):
        return {
            param_name: self.weights[self.params_no[param_name]].item()
            for param_name in self.params_no
        }

    def get_params_weights(self):
        return {
            param_name: self.weights[self.params_no[param_name]]
            for param_name in self.params_no
        }

    def forward(self, t, y, steps):
        """
        steps: (batch_size, )
        t: (batch_size, )
        y: (batch_size, y_num)
        """
        dy = self.ode_func(y, t, self.get_params_weights())  # (batch_size, y_num)

        y_next = y + steps * dy

        return y_next
