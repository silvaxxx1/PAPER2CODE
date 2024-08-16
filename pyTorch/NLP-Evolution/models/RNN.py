 
import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    """
    A simple RNN-based language model for text generation.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        hidden_size (int): Number of units in the hidden layer of the RNN.
        num_layers (int): Number of RNN layers.

    Attributes:
        embedding (nn.Embedding): Embedding layer to map token indices to dense vectors.
        rnn (nn.RNN): RNN layer to process the sequence data.
        fc (nn.Linear): Fully connected layer to produce the output logits.
    """
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(RNNLanguageModel, self).__init__()
        
        # Initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Initialize the RNN layer
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Initialize the fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, targets=None):
        """
        Forward pass through the RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): Target tensor of shape (batch_size, sequence_length). If provided, calculates the loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor or None]: 
                - logits (torch.Tensor): Output tensor of shape (batch_size, sequence_length, vocab_size).
                - loss (torch.Tensor, optional): Loss value if targets are provided.
        """
        # Get embeddings for the input tokens
        embeds = self.embedding(x)
        
        # Process the embeddings with the RNN
        rnn_out, _ = self.rnn(embeds)
        
        # Compute the logits for each token in the sequence
        logits = self.fc(rnn_out)

        if targets is not None:
            # Calculate the loss if targets are provided
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits
