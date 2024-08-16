import torch
import torch.nn as nn
from torch.nn import functional as F

class LSTMLanguageModel(nn.Module):
    """
    A language model based on LSTM (Long Short-Term Memory) architecture.

    Args:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers (default: 2).

    Forward Pass:
        x (Tensor): Input tensor of shape (batch_size, sequence_length).
        target (Tensor, optional): Target tensor for computing loss.

    Returns:
        logits (Tensor): Logits of shape (batch_size, sequence_length, vocab_size).
        loss (Tensor or None): Cross-entropy loss (if target is provided).
    """

    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super().__init__()
        
        # Embedding layer to convert token indices to dense vectors
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # LSTM layers with num_layers stacked layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Output layer projecting hidden state output to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, target=None):
        """
        Forward pass of the LSTM language model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).
            target (Tensor, optional): Target tensor of shape (batch_size, sequence_length).

        Returns:
            logits (Tensor): Logits of shape (batch_size, sequence_length, vocab_size).
            loss (Tensor or None): Cross-entropy loss (if target is provided).
        """

        # Get embeddings for input x
        x = self.embed(x)

        # Pass through the LSTM layers
        x, _ = self.lstm(x)  # We don't need the hidden state output here

        # Final layer projecting to vocabulary size
        logits = self.fc(x)

        if target is None:
            loss = None
        else:
            # Reshape logits and targets to match the dimensions expected by cross_entropy
            B, T, C = logits.shape  # Unpack the dimensions of logits
            logits = logits.view(B * T, C)  # Reshape logits to (B*T, C)
            target = target.view(B * T)  # Reshape targets to (B*T)
            loss = F.cross_entropy(logits, target)  # Compute the cross-entropy loss

        return logits, loss
