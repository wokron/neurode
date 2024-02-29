import math
from typing import Callable
from torch import nn
import torch


class NeuroODE(nn.Module):
    def __init__(
        self,
        params: dict[str, float],
        # (y: list[float], t: float, params: dict[str, float])
        ode_func: Callable[[list[float], float, dict[str, float]], list[float]],
        max_step: float,
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
        self.max_step = max_step

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

    def next(self, t: torch.Tensor, yi: torch.Tensor, step: torch.Tensor):
        """
        t: ()
        y: (y_num, )
        steps: ()
        """
        dy = self.ode_func(yi, t, self.get_params_weights())  # (y_num, batch_size)
        dy = torch.stack(dy, dim=0)

        y_next = yi + step * dy  # (y_num, )

        return y_next

    def step(self, t: torch.Tensor, yi: torch.Tensor, step: torch.Tensor):
        """
        t: ()
        y: (y_num, )
        steps: ()
        """
        step_n = math.ceil(step.item() / self.max_step)
        step = step / step_n

        for _ in range(step_n):
            yi = self.next(t, yi, step)
            t = t + step

        return yi

    def forward(self, y0: torch.Tensor, t: torch.Tensor):
        """
        y0: (y_num, )
        t: (sample_num, )
        """

        steps = torch.diff(t)  # (sample_num-1, )
        t = t[:-1]  # (sample_num-1, )

        y_arr = [y0]
        yi = y0
        for no in range(t.shape[0]):
            yi = self.step(t[no], yi, steps[no])
            y_arr.append(yi)

        return torch.stack(y_arr, dim=0)  # (sample_num, )
