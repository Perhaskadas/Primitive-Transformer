import torch
import torch.nn as nn
from decoder_block import Decoderblock
from encoder_block import EncoderBlock

class Transformer(nn.Module):
    # Note I recently just learned that super().__init__() and super(classname, self).__init__() are the same
    # But one is for super().__init__() is from python 3
    def __init__(self, heads: int, ffn_hidden: int, layers: int, enc_vocab: int, dec_vocab: int, 
                 model: int, enc_padding_tokID: torch.Tensor, dec_padding_tokID: torch.Tensor, dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.enc_padding_tokID = enc_padding_tokID
        self.dec_padding_tokID = dec_padding_tokID

        self.encoder = EncoderBlock(
            num_heads = heads, 
            n_hidden = ffn_hidden, 
            num_layers = layers, 
            vocab_size = enc_vocab, 
            d_model = model,
            dropout = dropout
            )
        
        self.decoder = Decoderblock(
            d_model = model,
            num_heads = heads,
            n_hidden = ffn_hidden,
            vocab_size = dec_vocab,
            dropout = dropout
        )
        self.linear = nn.Linear(model, dec_vocab) # Final Linear Layer
    
    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor):
        encoder_mask = self.encoder_padding_mask(enc_input)
        decoder_mask = self.decoder_padding_causal_mask(dec_input)
        encoder_output = self.encoder(enc_input, encoder_mask)
        decoder_output = self.decoder(dec_input, encoder_output, decoder_mask, encoder_mask)
        output = self.linear(decoder_output)
        return output

    def encoder_padding_mask(self, input_tensor_enc: torch.Tensor):
        """
        Creates a padding mask tensor for the encoder input tensor.
        """
        # Create a bool tensor with the same shape as the input tensor 
        padding_mask = input_tensor_enc != self.enc_padding_tokID # True = not padding, False = padding
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) # Add dimensions to match the shape of the attention tensor
        return padding_mask

    def decoder_padding_causal_mask(self, input_tensor_dec: torch.Tensor):
        """
        Creates a padding mask tensor combined with a causal mask tensor for the decoder input tensor.
        """
        # Creating a bool tensor with the same shape as the input tensor as padding mask
        padding_mask = input_tensor_dec != self.dec_padding_tokID # (batch_size, seq_length)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)

        # Making the causal mask
        seq_length = input_tensor_dec.size(1) # Get the sequence length
        causal_mask = torch.ones(seq_length, seq_length, dtype=torch.bool) # Create a bool tensor of ones with the shape (seq_length, seq_length)

        # Create a tensor of zeros above the diagonal: 1 = attend, 0 = don't attend/future tokens
        causal_mask = torch.tril(causal_mask)

        # Use bitwise AND to combine the two masks
        combined_mask = padding_mask & causal_mask # This operations gives us (batch_size, 1, seq_length, seq_length)
        return combined_mask 






        
        







