from typing import Callable
from torch import nn
import torch


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
        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=float), requires_grad=True
        )
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

    def forward(self, t: torch.Tensor, y: torch.Tensor, steps: torch.Tensor):
        """
        t: (batch_size, )
        y: (batch_size, y_num)
        steps: (batch_size, )
        """
        y = y.transpose(0, 1)  # (y_num, batch_size)
        dy = self.ode_func(y, t, self.get_params_weights())  # (y_num, batch_size)
        dy = torch.stack(dy, dim=0)

        y_next = y + steps * dy  # (y_num, batch_size)
        y_next = y_next.transpose(0, 1)  # (batch_size, y_num)

        return y_next
