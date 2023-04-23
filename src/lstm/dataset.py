import torch
import numpy as np

from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, seq_length, words, token2idx, idx2token, sample_size=None):
        self.seq_length = seq_length
        self.words = words
        self.idx2token = idx2token
        self.token2idx = token2idx
        self.data = []
        for token in self.words:
            try:
                self.data.append(token2idx[token])
            except KeyError:
                self.data.append(token2idx['<unknown/>'])
        # self.data = [token2idx[token] for token in self.words]
        self.sample_size = sample_size

    def __len__(self):
        if self.sample_size is not None:
            return self.sample_size
        return len(self.data) - self.seq_length
    
    def translate(self, batch: torch.Tensor):
        res = [self.idx2token[idx] for idx in batch.view(-1).tolist()]
        return np.array(res).reshape(batch.size())

    def __getitem__(self, index):
        return (torch.tensor(self.data[index:index+self.seq_length]),
                torch.tensor(self.data[index+1:index+1+self.seq_length]))
