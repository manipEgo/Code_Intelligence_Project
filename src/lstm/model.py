import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, token_num, embedding_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Embedding(token_num, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, token_num)

    def forward(self, x, hidden):
        out = self.encoder(x)
        out, hidden = self.lstm(out, hidden)
        out = self.decoder(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
