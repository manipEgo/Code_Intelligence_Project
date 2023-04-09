import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, seq_length, token2idx, sample_size=None):
        self.seq_length = seq_length
        self.data = [token2idx[token] for token in token2idx.keys()]
        if sample_size is not None:
            self.data = self.data[:sample_size]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (torch.tensor(self.data[index:index + self.seq_length]),
                torch.tensor(self.data[index + 1:index + 1 + self.seq_length]))
