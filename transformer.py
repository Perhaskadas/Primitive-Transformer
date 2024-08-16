import torch
from torch import nn
from decoder import DecoderBlock
from encoder import EncoderBlock
from misc_layers import ProjectionToVocab
from input_encoding import InputEmbeddingAndPositionalEncoding

class Transformer(nn.Module):
    def __init__(self, encoder_padding_tokID: torch.Tensor, decoder_padding_tokID: torch.Tensor):
        super(Transformer, self).__init__()
        self.encoder_padding_tokID = encoder_padding_tokID
        self.decoder_padding_tokID = decoder_padding_tokID
        
        self.encoder_embedding = InputEmbeddingAndPositionalEncoding()
        self.decoder_embedding = InputEmbeddingAndPositionalEncoding()

        self.encoder = EncoderBlock(self.encoder_embedding)
        self.decoder = DecoderBlock(self.decoder_embedding)

        self.projection_to_vocab = ProjectionToVocab()
    
    def encode(self, enc_input: torch.Tensor, encoder_padding_mask: torch.Tensor):
        encoder_output = self.encoder(enc_input, encoder_padding_mask)
        return encoder_output
    
    def decode(self, dec_input: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        decoder_output = self.decoder(dec_input, encoder_output, encoder_mask, decoder_mask)
        return decoder_output
    
    def decode(self, dec_input: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        decoder_output = self.decoder(dec_input, encoder_output, encoder_mask, decoder_mask)
        decoder_output = self.projection_to_vocab(decoder_output)
        return decoder_output
    
    def project(self, x: torch.Tensor):
        x = self.projection_to_vocab(x)
        return x