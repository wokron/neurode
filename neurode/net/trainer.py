import numpy as np
import torch

from neurode.net.model import NeuroODE


class ODETrainer:
    def __init__(
        self,
        device: str,
        epoches: int,
        lr: float,
    ) -> None:
        self.device = torch.device(device)
        self.epoches = epoches
        self.lr = lr

    def train(self, model: NeuroODE, y: np.ndarray, t: np.ndarray):
        y = torch.tensor(y, dtype=float).to(self.device)
        t = torch.tensor(t, dtype=float).to(self.device)

        model = model.to(self.device)

        loss_fn = torch.nn.MSELoss().to(self.device)

        optim = torch.optim.Adam(model.parameters(), self.lr)

        for _ in range(self.epoches):
            result = model(y[0], t)

            loss = loss_fn(result, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
