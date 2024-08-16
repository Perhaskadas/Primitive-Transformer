from torch import nn
import torch
from input_encoding import InputEmbeddingAndPositionalEncoding
from attention import MultiHeadAttention
import hyperparameters as hp
from misc_layers import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, decoder_embedding: nn.Module):
        super(DecoderBlock, self).__init__()
        self.inputembedding = decoder_embedding
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(hp.num_layers)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        """
        Args: 
            x: Tensor of shape (batch_size, seq_len)
            encoder_output: Tensor of shape (batch_size, seq_len, embedding_dim)
            causal_mask: Tensor of shape (batch_size, seq_len, seq_len)
            padding_mask: Tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.inputembedding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.maskedSelfAttention = MultiHeadAttention()
        self.layernorm1 = nn.LayerNorm(hp.embedding_dim)
        self.dropout = nn.Dropout(hp.dropout)

        self.crossAttention = MultiHeadAttention()
        self.layernorm2 = nn.LayerNorm(hp.embedding_dim)

        self.feedforward = FeedForward()
        self.layernorm3 = nn.LayerNorm(hp.embedding_dim)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor, encoder_mask: torch.Tensor, decoder_padding_mask: torch.Tensor):
        """
        Args: 
            decoder: Tensor of shape (batch_size, seq_len, embedding_dim)
            encoder: Tensor of shape (batch_size, seq_len, embedding_dim)
            causal_mask: Tensor of shape (batch_size, seq_len, seq_len)
            padding_mask: Tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            output: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        residual_connect = decoder
        output = self.maskedSelfAttention(decoder, decoder, decoder, decoder_padding_mask)
        output = self.dropout(output)
        output = self.layernorm1(output + residual_connect)

        residual_connect = output
        output = self.crossAttention(decoder, encoder, encoder, encoder_mask)
        output = self.dropout(output)
        output = self.layernorm2(output + residual_connect)

        residual_connect = output
        output = self.feedforward(output)
        output = self.dropout(output)
        output = self.layernorm3(output + residual_connect)

        return output
