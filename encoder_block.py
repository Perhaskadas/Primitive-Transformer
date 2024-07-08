import torch
from single_encoder_layer import SingleEncoderLayer
import torch.nn as nn
from input_embedding import InputEmbedding

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, n_hidden: int, num_layers: int, vocab_size: int, d_model: int, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()
        self.inputembedding = InputEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SingleEncoderLayer(d_model, num_heads, n_hidden, dropout = dropout) for _ in range(num_layers)])

    def forward(self, x, causal_mask):
        x = self.inputembedding(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return x
