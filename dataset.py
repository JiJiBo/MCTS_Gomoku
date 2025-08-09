import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GomokuDataset(Dataset):
    """Dataset reading self-play data stored in .npz files.

    Each .npz file should contain arrays 'boards' [N,4,H,W],
    'policies' [N,H*W] and 'values' [N]. All files in the directory
    are concatenated together.
    """

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if not self.files:
            raise RuntimeError(f'No npz files found in {data_dir}')
        boards, policies, values = [], [], []
        for f in self.files:
            data = np.load(f)
            boards.append(data['boards'])
            policies.append(data['policies'])
            values.append(data['values'])
        self.boards = np.concatenate(boards, axis=0)
        self.policies = np.concatenate(policies, axis=0)
        self.values = np.concatenate(values, axis=0)

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        board = torch.from_numpy(self.boards[idx]).float()
        policy = torch.from_numpy(self.policies[idx]).float()
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return board, policy, value
