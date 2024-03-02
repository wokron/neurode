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

    def train(self, model: NeuroODE, t: np.ndarray, y: np.ndarray):
        t = torch.tensor(t, dtype=float).to(self.device)
        y = torch.tensor(y, dtype=float).to(self.device)

        model = model.to(self.device)
        loss_fn = torch.nn.MSELoss().to(self.device)
        optim = torch.optim.Adam(model.parameters(), self.lr)

        range_epoches = trange(self.epoches, disable=(not self.verbose))
        loss_record: list[float] = []
        last_epoch = 0

        try:
            for epoch in range_epoches:
                result = model(t, y[0])
                loss = loss_fn(result, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_value = loss.item()
                range_epoches.set_postfix(loss_value)
                loss_record.append(loss_value)
                last_epoch = epoch

        except KeyboardInterrupt:
            pass

        return {
            "loss_record": loss_record,
            "last_epoch": last_epoch,
            "params": model.get_params(),
        }
