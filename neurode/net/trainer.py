from typing import Type
import torch
from torch.utils.data import DataLoader
from neurode.net.data import ODEDataset
from neurode.net.model import NeuroODE


class ODETrainer:
    def __init__(
        self,
        device: torch.device,
        epoches: int,
        optim_cls: Type[torch.optim.Optimizer],
        batch_size: int,
        num_workers: int,
        lr: float,
    ) -> None:
        self.device = device
        self.optim_cls = optim_cls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoches = epoches
        self.lr = lr

    def train(self, model: NeuroODE, dataset: ODEDataset):
        dataloader = DataLoader(
            dataset,
            self.batch_size,
            num_workers=self.num_workers,
        )

        model = model.to(self.device)

        loss_fn = torch.nn.MSELoss().to(self.device)

        optim = self.optim_cls(model.parameters(), self.lr)

        loss_arr = []

        for _ in range(self.epoches):
            for data in dataloader:
                t, y, steps, y_next = [elm.to(self.device) for elm in data]

                result = model(t, y, steps)

                loss = loss_fn(result, y_next)

                loss_arr.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()

        pass
