from torch.utils.data import Dataset
import torch
import numpy as np


class ODEDataset(Dataset):
    def __init__(self, y: np.ndarray, t: np.ndarray) -> None:
        """
        y: (sample_num, y_num)
        t: (sample_num, )
        """
        super().__init__()
        self.y = y[:-1, :]  # (sample_num-1, y_num)
        self.y_next = y[1:, :]  # (sample_num-1, y_num)
        self.steps = np.diff(t)  # (sample_num-1, )
        self.t = t[:-1]  # (sample_num-1, )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, no):
        t = torch.tensor(self.t[no], dtype=float)
        y = torch.tensor(self.y[no], dtype=float)
        steps = torch.tensor(self.steps[no], dtype=float)
        y_next = torch.tensor(self.y_next[no], dtype=float)
        return t, y, steps, y_next
