from torch import nn
import math
import torch
import logging

logging.basicConfig(filename = "train2.log", 
		    level=logging.DEBUG,
		    format="%(asctime)s:%(levelname)s:%(message)s")

class InputEmbedding(nn.Module):
    # Used with ROPE
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class InputEmbeddingWithSinEncode(nn.Module):
    # Use if not using ROPE
    def __init__(self, d_model: int, vocab_size: int, max_len: int = 350) -> None:
        super(InputEmbeddingWithSinEncode, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        logging.debug(f"Vocabsize: {vocab_size}")
        self.embedding = nn.Embedding(vocab_size, d_model)
        logging.debug(f"Embedding shape: {self.embedding.weight.shape}")
        self.max_len = max_len
        
        self.positional_encoding = SinPositionalEncoding(d_model, max_len)

    def forward(self, x):
        logging.debug(f"InputEmbeddingWithSinEncode: x shape: {x.shape}")
        logging.debug(f"InputEmbeddingWithSinEncode: x min: {x.min().item()}, x max: {x.max().item()}")
        if x.max().item() >= self.vocab_size or x.min().item() < 0:
            raise ValueError(f"Input tensor containsout-of-range indices: min {x.min().item()}, max {x.max().item()}")
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        encoded = self.positional_encoding(embedded)
        return encoded

    
class SinPositionalEncoding(nn.Module):
    # Taken from https://github.com/hkproj/pytorch-transformer/blob/main/model.py
    def __init__(self, d_model: int, max_len: int = 350) -> None:
        super(SinPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(0.1)
        base = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        divisor = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # (d_model // 2)
        
        # Apply sine to even base indices
        base[:, 0::2] = torch.sin(position * divisor)
        # Apply cosine to odd base indices
        base[:, 1::2] = torch.cos(position * divisor)

        # Add a batch dimension to the positional encoding
        base = base.unsqueeze(0) # (1, max_len, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('SinCosPosEncoding', base)

    def forward(self, x):
        x = x + (self.SinCosPosEncoding[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)