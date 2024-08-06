import torch
from torch import nn 
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward

class SingleDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n_hidden: int, use_rotary_embed: bool = False, dropout: float = 0.1):
        super(SingleDecoderLayer, self).__init__()
        self.maskedSelfAttention = MultiHeadAttention(d_model, num_heads, use_rotary_emb=use_rotary_embed)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.crossAttention = MultiHeadAttention(d_model, num_heads)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.feedforward = FeedForward(d_model, n_hidden)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor):
        # causal_mask is for the self attention, to make sure the NN doesn't cheat during training
        # padding_mask is for encoder-decoder attention and tells the NN to ignore padding

        residual_connect = decoder
        output = self.maskedSelfAttention(decoder, decoder, decoder, causal_mask)
        output = self.layernorm1(output + residual_connect)
        output = self.dropout(output)

        residual_connect = output
        output = self.crossAttention(encoder, encoder, decoder, padding_mask)
        output = self.layernorm2(output + residual_connect)
        output = self.dropout(output)

        residual_connect = output
        output = self.feedforward(output)
        output = self.layernorm3(output + residual_connect)
        output = self.dropout(output)

        return output



        

