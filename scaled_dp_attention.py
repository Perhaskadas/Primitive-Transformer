import numpy as np
from torch import nn
import torch
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    from typing import Tuple
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # qkv are 4 dimension tensors [batch_size, num_heads, seq_len, d_k]
        batch_size, num_heads, seq_len, d_k = k.size()

        # Compute the dot product between the query and the key
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply the mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
    
        # Softmax the (seq_len, seq_len) scores tensor
        scores = self.softmax(scores)

        # Compute the weighted sum of the values
        output = torch.matmul(scores, v)
        
        return output, scores
    


