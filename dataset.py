import os
import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_self_play_dataset(data_dir: str) -> TensorDataset:
    """Load and concatenate all .npz self-play files from a directory.

    Each .npz file is expected to contain ``boards``, ``policies`` and ``values``
    arrays. The returned :class:`~torch.utils.data.TensorDataset` yields
    ``(board, policy, value)`` tuples suitable for training.
    """
    pattern = os.path.join(data_dir, '*.npz')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No .npz files found in {data_dir!r}')

    boards, policies, values = [], [], []
    for path in files:
        data = np.load(path)
        boards.append(data['boards'])
        policies.append(data['policies'])
        values.append(data['values'])

    boards = torch.from_numpy(np.concatenate(boards)).float()
    policies = torch.from_numpy(np.concatenate(policies)).float()
    values = torch.from_numpy(np.concatenate(values)).float()
    return TensorDataset(boards, policies, values)
