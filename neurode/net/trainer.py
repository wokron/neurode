import numpy as np
import torch
from tqdm import trange

from neurode.net.model import NeuroODE


class ODETrainer:
    def __init__(
        self,
        device: str,
        epoches: int,
        lr: float,
        verbose: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.epoches = epoches
        self.lr = lr
        self.verbose = verbose

    def train(self, model: NeuroODE, y: np.ndarray, t: np.ndarray):
        y = torch.tensor(y, dtype=float).to(self.device)
        t = torch.tensor(t, dtype=float).to(self.device)

        model = model.to(self.device)

        loss_fn = torch.nn.MSELoss().to(self.device)

        optim = torch.optim.Adam(model.parameters(), self.lr)

        range_epoches = trange(self.epoches, disable=(not self.verbose))
        for _ in range_epoches:
            result = model(y[0], t)

            loss = loss_fn(result, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            range_epoches.set_postfix(loss=loss.item())
