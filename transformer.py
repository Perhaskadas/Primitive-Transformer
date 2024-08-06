import torch
import torch.nn as nn
from decoder_block import Decoderblock
from encoder_block import EncoderBlock
from input_embedding import InputEmbedding, InputEmbeddingWithSinEncode

class Transformer(nn.Module):
    # Note I recently just learned that super().__init__() and super(classname, self).__init__() are the same
    # But one is for super().__init__() is from python 3
    def __init__(self, heads: int, 
                 ffn_hidden: int, 
                 layers: int, 
                 vocab_size1: int, 
                 vocab_size2: int, 
                 model: int, 
                 enc_padding_tokID: torch.Tensor, 
                 dec_padding_tokID: torch.Tensor, 
                 use_rotary_embed: bool = False,
                 max_len: int = 350, 
                 dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.enc_padding_tokID = enc_padding_tokID
        self.dec_padding_tokID = dec_padding_tokID
        if use_rotary_embed:
            encoder_embedding = InputEmbedding(vocab_size=vocab_size1, d_model=model, max_len=max_len)
            decoder_embedding = InputEmbedding(vocab_size=vocab_size2, d_model=model, max_len=max_len)
        else:
            encoder_embedding = InputEmbeddingWithSinEncode(vocab_size=vocab_size1, d_model=model, max_len=max_len)
            decoder_embedding = InputEmbeddingWithSinEncode(vocab_size=vocab_size2,d_model=model, max_len=max_len)
        self.encoder = EncoderBlock(
            num_heads = heads, 
            n_hidden = ffn_hidden, 
            num_layers = layers, 
            vocab_size = vocab_size1, 
            d_model = model,
            embedding = encoder_embedding,
            use_rotary_emb=use_rotary_embed,
            dropout = dropout
            )
        
        self.decoder = Decoderblock(
            d_model = model,
            num_heads = heads,
            n_hidden = ffn_hidden,
            num_layers = layers,
            vocab_size = vocab_size2,
            embedding = decoder_embedding,
            use_rotary_emb=use_rotary_embed,
            dropout = dropout
        )
        self.linear = nn.Linear(model, vocab_size2) # Final Linear Layer
    
    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor):
        encoder_mask = self.encoder_padding_mask(enc_input)
        decoder_mask = self.decoder_padding_causal_mask(dec_input)
        encoder_output = self.encoder(enc_input, encoder_mask)
        decoder_output = self.decoder(dec_input, encoder_output, decoder_mask, encoder_mask)
        output = self.linear(decoder_output)
        return output
    
    def encode(self, enc_input: torch.Tensor, enc_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        return self.encoder(enc_input, enc_mask)
    
    def decode(self, dec_input: torch.Tensor, encoder_output: torch.Tensor, decoder_mask: torch.Tensor, encoder_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        return self.decoder(dec_input, encoder_output, decoder_mask, encoder_mask)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Creating a bool tensor with the same shape as the input tensor as padding mask
        padding_mask = input_tensor_dec != self.dec_padding_tokID # (batch_size, seq_length)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)
        padding_mask = padding_mask.to(device)
        # Making the causal mask
        seq_length = input_tensor_dec.size(1) # Get the sequence length
        causal_mask = torch.ones(seq_length, seq_length, dtype=torch.bool) # Create a bool tensor of ones with the shape (seq_length, seq_length)
        # Create a tensor of zeros above the diagonal: 1 = attend, 0 = don't attend/future tokens
        causal_mask = torch.tril(causal_mask)
        causal_mask = causal_mask.to(device)

        # Use bitwise AND to combine the two masks
        combined_mask = padding_mask & causal_mask # This operations gives us (batch_size, 1, seq_length, seq_length)
        return combined_mask 






        
        







