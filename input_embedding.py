from torch import nn

class InputEmbedding(nn.Embedding):
    """
    Input embedding layer.

    This class represents an input embedding layer that maps input tokens to their corresponding embeddings.
    It extends the `nn.Embedding` class from the PyTorch library.

    Parameters:
        vocab_size (int): The size of the vocabulary, i.e., the total number of unique tokens.
        d_model (int): The dimensionality of the output embeddings.

    Attributes:
        padding_idx (int): The index used for padding tokens.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)