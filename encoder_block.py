import torch
from single_encoder_layer import SingleEncoderLayer
import torch.nn as nn
from input_embedding import InputEmbedding, InputEmbeddingWithSinEncode

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, n_hidden: int, num_layers: int, vocab_size: int, d_model: int, embedding: nn.Module, use_rotary_emb: bool = False, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()
        self.inputembedding = embedding
        self.use_rotary_emb = use_rotary_emb
        self.layers = nn.ModuleList([SingleEncoderLayer(d_model, num_heads, n_hidden, use_rotary_embed=self.use_rotary_emb, dropout = dropout) for _ in range(num_layers)])

    def forward(self, x, causal_mask):
        x = self.inputembedding(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return x
