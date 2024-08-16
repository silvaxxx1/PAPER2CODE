import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    """
    A simple Bigram Language Model implemented using PyTorch. This model uses an embedding layer
    to predict the next token in a sequence based on the current token. 

    Attributes:
        look_up_table (nn.Embedding): Embedding layer that maps each token to a vector of logits
    """

    def __init__(self, vocab_size):
        """
        Initializes the BigramLanguageModel with a lookup table of size (vocab_size, vocab_size).
        
        Args:
            vocab_size (int): Size of the vocabulary. This determines the dimensions of the embedding layer.
        """
        super().__init__()
        # Initialize the lookup table where each token's embedding size equals the vocab size.
        self.look_up_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, target=None):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token indices.
            target (torch.Tensor, optional): Target tensor of shape (batch_size, sequence_length) for loss computation.
        
        Returns:
            logits (torch.Tensor): Logits of shape (batch_size, sequence_length, vocab_size) representing token predictions.
            loss (torch.Tensor or None): Computed cross-entropy loss if targets are provided; otherwise, None.
        """
        # Get logits from the lookup table
        logits = self.look_up_table(x)
        
        # Compute the loss if targets are provided
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape  # Unpack the dimensions of logits
            logits = logits.view(B * T, C)  # Reshape logits to (B*T, C)
            target = target.view(B * T)  # Reshape targets to (B*T)
            loss = F.cross_entropy(logits, target)  # Compute the cross-entropy loss

        return logits, loss
