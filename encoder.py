from torch import nn
import torch
from input_encoding import InputEmbeddingAndPositionalEncoding
from attention import MultiHeadAttention
import hyperparameters as hp
from misc_layers import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, encoder_embedding: nn.Module):
        super(EncoderBlock, self).__init__()
        self.inputembedding = encoder_embedding
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(hp.num_layers)])
    
    def forward(self, x, causal_mask) -> torch.Tensor:
        """
        Args: 
            x: Tensor of shape (batch_size, seq_len)
            causal_mask: Tensor of shape (batch_size, seq_len, seq_len) or None
        Returns:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.inputembedding(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention()
        self.layernorm1 = nn.LayerNorm(hp.embedding_dim)
        self.dropout = nn.Dropout(hp.dropout)
        self.feedforward = FeedForward()
        self.layernorm2 = nn.LayerNorm(hp.embedding_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Tensor of shape (batch_size, seq_len, seq_len) or None
        Returns:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        residual_connect = x
        x = self.multihead(x, x, x, mask) # Calculate Attention
        x = self.dropout(x) # Drop connections to avoid overfit
        x = self.layernorm1(x + residual_connect) # Add and Normalize

        residual_connect = x
        x = self.feedforward(x) # Feed Forward
        x = self.dropout(x) # Drop connections to avoid overfit
        x = self.layernorm2(x + residual_connect) # Add and Normalize
        return x