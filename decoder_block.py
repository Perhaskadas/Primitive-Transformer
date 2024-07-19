import torch
from single_decoder_layer import SingleDecoderLayer
import torch.nn as nn
from input_embedding import InputEmbedding

class Decoderblock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n_hidden: int, vocab_size: int, embedding: InputEmbedding, dropout: float = 0.1):
        super(Decoderblock, self).__init__()
        self.inputembedding = embedding
        self.layers = nn.ModuleList([SingleDecoderLayer(d_model, num_heads, n_hidden)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor):
        x = self.inputembedding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, padding_mask)
        return x