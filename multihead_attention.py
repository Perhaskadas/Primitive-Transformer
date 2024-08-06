from rotary_embedding_torch import RotaryEmbedding
import torch
import torch.nn as nn
from scaled_dp_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rotary_emb: bool = False) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.weight_q = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)
        self.weight_k = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)
        self.weight_v = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)
        self.weight_concat = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)
        self.SDPAttention = ScaledDotProductAttention()
        self.use_rotary_emb = use_rotary_emb
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # MM qkv with their weight matrices
        q = self.weight_q(q)
        k = self.weight_k(k)
        v = self.weight_v(v)
        # Split the query, key, and value tensors into multiple heads
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        q = q.cpu()
        k = k.cpu()
        # Apply rotation to q and k
        if self.use_rotary_emb:
            rotary_emb = RotaryEmbedding(dim = 32)
            q = rotary_emb.rotate_queries_or_keys(q).to(device)
            k = rotary_emb.rotate_queries_or_keys(k).to(device)
        
        # Compute the scaled dot-product attention
        output, scores = self.SDPAttention(q, k, v, mask)

        # Concatenate the multiple heads
        result = self.concat(output)
        final_result = self.weight_concat(result)

        return final_result
    
    def split(self, qkv: torch.Tensor) -> torch.Tensor:
        # qkv is a 3 dimension tensor [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = qkv.size()
        
        # Split the d_model dimension into num_heads
        qkv = qkv.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads)
        
        # Transpose the dimensions to align the head dimension with the batch dimension
        return qkv.transpose(1, 2)

    def concat(self, qkv: torch.Tensor) -> torch.Tensor:
        # qkv is a 4 dimension tensor [batch_size, num_heads, seq_len, d_k]
        batch_size, num_heads, seq_len, d_k = qkv.size()
        
        # Transpose the dimensions to restore the original tensor shape
        qkv = qkv.transpose(1, 2).contiguous()
        
        # Combine the num_heads dimension into the d_model dimension
        cat_qkv = qkv.view(batch_size, seq_len, self.num_heads * d_k)
        
        return cat_qkv

        
