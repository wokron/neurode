from ..net.data import ODEDataset
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader


def test_data_process():
    y = np.array(
        [
            [1, 2],
            [3, 4],
            [4, 5],
            [5, 6],
        ]
    )
    t = np.array([1, 2, 3, 4])
    dataset = ODEDataset(y, t)

    assert len(dataset) == 3
    t, y, steps, y_next = dataset[0]
    assert t.numpy() == 1
    assert (y.numpy() == [1, 2]).all()
    assert steps.numpy() == 1
    assert (y_next.numpy() == [3, 4]).all()

    t, y, steps, y_next = dataset[-1]
    assert t.numpy() == 3
    assert (y.numpy() == [4, 5]).all()
    assert steps.numpy() == 1
    assert (y_next.numpy() == [5, 6]).all()


def test_dataloader():
    y = np.array(
        [
            [1, 2],
            [3, 4],
            [4, 5],
            [5, 6],
        ]
    )
    t = np.array([1, 2, 3, 4])
    dataset = ODEDataset(y, t)

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(dataloader))
    assert len(batch) == 4

    t, y, steps, y_next = batch
    assert (t.numpy() == [1, 2, 3]).all()
    assert (y.numpy() == [[1, 2], [3, 4], [4, 5]]).all()
    assert (steps.numpy() == [1, 1, 1]).all()
    assert (y_next.numpy() == [[3, 4], [4, 5], [5, 6]]).all()
