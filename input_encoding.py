from torch import nn
import math
import torch
import hyperparameters as hp

class InputEmbeddingAndPositionalEncoding(nn.Module):
    def __init__(self):
        super(InputEmbeddingAndPositionalEncoding, self).__init__()
        self.embedding = InputEmbedding()
        self.positional_encoding = PositionalEncoding()
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return x

class InputEmbedding(nn.Module):
    def __init__(self):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(hp.vocab_size, hp.embedding_dim) # Embedding Matrix (vocab_size, embedding_dim)
        self.scale = math.sqrt(hp.embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)

        Returns:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.embedding(x) * self.scale
        return x

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(hp.dropout)
        pe = torch.zeros(hp.seq_len, hp.embedding_dim) # (seq_len, embedding_dim)
        position = torch.arange(0, hp.seq_len, dtype=torch.float) # (seq_len)
        position = position.unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, hp.embedding_dim, 2).float() * (-math.log(10000.0) / hp.embedding_dim)) # (d_model / 2)

        # Apply sine and cosine to odd and even indices respectively
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # nograd since its not learned and is a buffer
        return self.dropout(x)