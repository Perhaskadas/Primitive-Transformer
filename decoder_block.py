import torch
from single_decoder_layer import SingleDecoderLayer
import torch.nn as nn
from input_embedding import InputEmbedding, InputEmbeddingWithSinEncode

class Decoderblock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n_hidden: int, num_layers: int, vocab_size: int, embedding: nn.Module, use_rotary_emb: bool = False, dropout: float = 0.1):
        super(Decoderblock, self).__init__()
        self.inputembedding = embedding
        self.use_rotary_embed = use_rotary_emb
        self.layers = nn.ModuleList([SingleDecoderLayer(d_model, num_heads, n_hidden, use_rotary_embed=self.use_rotary_embed) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor):
        x = self.inputembedding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, padding_mask)
        return x
