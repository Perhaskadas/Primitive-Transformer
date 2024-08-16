from torch import nn
import torch
import hyperparameters as hp
import math
from typing import Tuple, Optional

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = hp.num_heads
        assert hp.embedding_dim % self.num_heads == 0 # embedding_dim must be divisible by num_heads
        self.Wq = nn.Linear(hp.embedding_dim, hp.embedding_dim, bias=False) # (embedding_dim, embedding_dim)
        self.Wk = nn.Linear(hp.embedding_dim, hp.embedding_dim, bias=False) # (embedding_dim, embedding_dim)
        self.Wv = nn.Linear(hp.embedding_dim, hp.embedding_dim, bias=False) # (embedding_dim, embedding_dim)
        self.Wconcat = nn.Linear(hp.embedding_dim, hp.embedding_dim, bias=False) # (embedding_dim, embedding_dim)
        self.sdp_attention = ScaledDotProductAttention()

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q: Tensor of shape (batch_size, seq_len, embedding_dim)
            k: Tensor of shape (batch_size, seq_len, embedding_dim)
            v: Tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Tensor of shape (batch_size, seq_len, seq_len) or None
        Returns:
            final_result: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # MM qkv with their weight matrices
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # Split the query, key, and value tensors into multiple heads
        q = self.split(q) # (batch_size, num_heads, seq_len, embedding_dim // num_heads)
        k = self.split(k) # (batch_size, num_heads, seq_len, embedding_dim // num_heads)
        v = self.split(v) # (batch_size, num_heads, seq_len, embedding_dim // num_heads)

        # Compute the scaled dot-product attention
        output, scores = self.sdp_attention(q, k, v, mask)

        # Concatenate the multiple heads
        result = self.concat(output) # (batch_size, seq_len, embedding_dim)
        final_result = self.Wconcat(result)

        return final_result

    def split(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = qkv.size() # (batch_size, seq_len, embedding_dim)
        
        # Split the embedding_dim dimension into num_heads
        qkv = qkv.view(batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads) # (batch_size, seq_len, num_heads, embedding_dim // num_heads)
        
        # Transpose the dimensions to align the head dimension with the batch dimension
        return qkv.transpose(1, 2) # (batch_size, num_heads, seq_len, embedding_dim // num_heads)

    def concat(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = qkv.size() # (batch_size, num_heads, seq_len, d_k)

        qkv = qkv.transpose(1, 2) # (batch_size, seq_len, num_heads, d_k)

        cat_qkv = qkv.contiguous().view(batch_size, seq_len, self.num_heads * d_k) # (batch_size, seq_len, embedding_dim)

        return cat_qkv 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(hp.dropout)
    
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Tensor of shape (batch_size, num_heads, seq_len, embedding_size)
            k: Tensor of shape (batch_size, num_heads, seq_len, embedding_size)
            v: Tensor of shape (batch_size, num_heads, seq_len, embedding_size)
            mask: Tensor of shape (batch_size, seq_len, seq_len) or None
        Returns:
            output: Tensor of shape (batch_size, num_heads, seq_len, embedding_size)
            scores: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, num_heads, seq_len, embedding_size = k.size()

        # Compute the dot product between the query and the key
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(embedding_size) # (batch_size, num_heads, seq_len, seq_len)

        # Apply the mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax the (seq_len, seq_len) scores tensor and dropout
        scores = self.dropout(self.softmax(scores))

        # Compute the weighted sum of the values
        output = torch.matmul(scores, v)

        return output, scores