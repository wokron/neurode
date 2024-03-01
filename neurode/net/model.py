import math
from typing import Callable

from torch import nn
import torch

from neurode.ode_next import ode_next_euler


class NeuroODE(nn.Module):
    def __init__(
        self,
        params: dict[str, float],
        # (y: list[float], t: float, params: dict[str, float])
        ode_fn: Callable[[list[float], float, dict[str, float]], list[float]],
        max_step: float,
        ode_next_fn=ode_next_euler,
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
            torch.nn.init.uniform_(torch.empty(len(weights)), a=0, b=1),
            requires_grad=True,
        )
        self.weights_ratios = weights

        self.ode_fn = ode_fn
        self.ode_next_fn = ode_next_fn
        self.max_step = max_step

    def get_params(self):
        return {
            param_name: self.weights[self.params_no[param_name]].item()
            * self.weights_ratios[self.params_no[param_name]]
            for param_name in self.params_no
        }

    def get_params_tensors(self):
        return {
            param_name: self.weights[self.params_no[param_name]]
            * self.weights_ratios[self.params_no[param_name]]
            for param_name in self.params_no
        }

    def next(self, t: torch.Tensor, yi: torch.Tensor, step: torch.Tensor):
        """
        t: ()
        y: (y_num, )
        steps: ()
        """

        def calc_dy(t, yi):
            dy = self.ode_fn(yi, t, self.get_params_tensors())  # (y_num, batch_size)
            dy = torch.stack(dy, dim=0)
            return dy

        return self.ode_next_fn(t, yi, step, calc_dy)

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

    def forward(self, t: torch.Tensor, y0: torch.Tensor):
        """
        t: (sample_num, )
        y0: (y_num, )
        """

        steps = torch.diff(t)  # (sample_num-1, )
        t = t[:-1]  # (sample_num-1, )

        y_arr = [y0]
        yi = y0
        for no in range(t.shape[0]):
            yi = self.step(t[no], yi, steps[no])
            y_arr.append(yi)

        return torch.stack(y_arr, dim=0)  # (sample_num, )
