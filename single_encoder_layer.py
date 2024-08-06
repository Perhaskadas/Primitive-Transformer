import torch
from torch import nn 
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward

class SingleEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n_hidden: int, use_rotary_embed: bool = False, dropout: float = 0.1):
        super(SingleEncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(d_model, num_heads, use_rotary_emb=use_rotary_embed)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForward(d_model, n_hidden)
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # One day I will test to see whether or not I should do Attention -> Drop -> Residual -> Normal
        residual_connect = x
        x = self.multihead(x, x, x, mask) # Calculate Attention
        x = self.layernorm1(x + residual_connect) # Add and Normalize
        x = self.dropout(x) # Drop connections to avoid overfit

        residual_connect = x
        x = self.feedforward(x) # Feed Forward
        x = self.layernorm2(x + residual_connect) # Add and Normalize
        x = self.dropout(x) # Drop connections to avoid overfit
        return x













