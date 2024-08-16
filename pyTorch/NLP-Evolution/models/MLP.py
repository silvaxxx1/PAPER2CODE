import torch 
import torch.nn as nn
from torch.nn import functional as F

class MLPLanguageModel(nn.Module):
    """
    A Multilayer Perceptron (MLP) based language model.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        hidden_size (int): Number of units in the hidden layers.
    """
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # Embedding layer to convert token indices to dense vectors
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # First fully connected layer followed by batch normalization
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second fully connected layer followed by batch normalization
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Output layer projecting to vocabulary size
        self.fc3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, target=None):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of token indices.
            target (torch.Tensor, optional): Target tensor for calculating loss. Default is None.

        Returns:
            logits (torch.Tensor): The output logits from the model.
            loss (torch.Tensor or None): The calculated cross-entropy loss if target is provided, otherwise None.
        """
        # Convert token indices to dense vectors
        x = self.embed(x)
        
        # Pass through the first fully connected layer and apply tanh activation
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        
        # Pass through the second fully connected layer and apply tanh activation
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        
        # Final layer producing logits for each token in the vocabulary
        logits = self.fc3(x)

        if target is None:
            loss = None
        else:
            # Reshape logits and targets for loss calculation
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits, target)
        
        return logits, loss
