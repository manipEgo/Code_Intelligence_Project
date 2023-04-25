import torch
import torch.nn as nn


class AttentionalLSTM(nn.Module):
    r"""Attentional LSTM model (https://arxiv.org/abs/1711.09573)


    Context attention:

    .. math::
        \begin{array}{ll}
            A_t = v^T \tanh (W^m M_t + (W^h h_t)1^T_L) \\
            \alpha_t = softmax(A_t) \\
            c_t = M_t \alpha^T_t
        \end{array}

    where :math:`W^m`, :math:`W^h` and :math:`v \in \mathbb{R}^k` are trainable parameters.
    :math:`k` is the size of hidden states, i.e. dimension of :math:`h_t`. :math:`1_L` represents
    an L-dimensional vector of ones.
    """

    def __init__(self,
                 device,
                 token_num,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1):
        super(AttentionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedd = nn.Embedding(token_num, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.v = nn.Linear(hidden_dim, 1)
        self.Wm = nn.Linear(hidden_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_dim, token_num)

    def forward(self, x, hidden):
        seq_length = x.size(1)
        out = self.embedd(x)
        M_t, (h_t, _) = self.lstm(out, hidden)
        h_t = h_t.transpose(0, 1)
        A_t = self.v(self.tanh(self.Wm(M_t) + torch.matmul(torch.ones(seq_length, 1).to(self.device), self.Wh(h_t))))
        alpha_t = self.softmax(A_t)
        c_t = torch.matmul(alpha_t.transpose(1, 2), M_t)
        out = self.fc(c_t)
        return out

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
