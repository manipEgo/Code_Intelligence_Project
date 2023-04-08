import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, filepath, token2idx, sample_size=None):
        self.data = []
        self.max_len = -torch.inf
        self.token2idx = token2idx
        with open(filepath, 'r') as f:
            for index, line in enumerate(f):
                if index == sample_size:
                    break
                tokens = line.strip().split()
                for i in range(len(tokens) - 1):
                    self.data.append((tokens[:i + 1], tokens[i + 1]))
                self.max_len = max(self.max_len, len(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_seq, output_token = self.data[index]
        input_vec = torch.tensor([self.token2idx[token] for token in input_seq])
        # Padding to the same length
        input_vec = F.pad(input_vec, (0, self.max_len - len(input_seq)), mode='constant', value=0)
        output_vec = torch.tensor(self.token2idx[output_token])
        return input_vec, output_vec
