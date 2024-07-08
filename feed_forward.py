import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, d_model, n_hidden, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.L1 = nn.Linear(d_model, n_hidden, bias=True)
        self.ReLU = nn.ReLU
        self.dropout = nn.Dropout(p=dropout)
        self.L2 = nn.Linear(n_hidden, d_model, bias=True)

    def forward(self, tensor: torch.Tensor):
        tensor = self.L1(tensor)
        tensor = self.ReLU(tensor)
        tensor = self.dropout(tensor)
        tensor = self.L2(tensor)
        return tensor


