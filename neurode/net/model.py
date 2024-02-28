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

    def step_do(self, t: torch.Tensor, yi: torch.Tensor, steps: torch.Tensor):
        """
        t: ()
        y: (y_num, )
        steps: ()
        """
        dy = self.ode_func(yi, t, self.get_params_weights())  # (y_num, batch_size)
        dy = torch.stack(dy, dim=0)

        y_next = yi + steps * dy  # (y_num, )

        return y_next

    def forward(self, y0: torch.Tensor, t: torch.Tensor):
        """
        y0: (y_num, )
        t: (sample_num, )
        """

        steps = torch.diff(t)  # (sample_num-1, )
        t = t[:-1]  # (sample_num-1, )

        y_arr = []
        yi = y0
        for no in range(t.shape[0]):
            yi = self.step_do(t[no], yi, steps[no])
            y_arr.append(yi)

        return torch.stack(y_arr, dim=0)  # (sample_num, )
