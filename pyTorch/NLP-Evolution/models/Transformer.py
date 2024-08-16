import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """
    Self-attention head for the transformer model.
    
    Args:
        head_size (int): Size of each attention head.
        n_embd (int): Dimensionality of the embeddings.
    """

    def __init__(self, head_size, n_embd):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_embd, n_embd)))

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through the attention head.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output of the attention mechanism.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute the attention weights
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        n_embd (int): Dimensionality of the embeddings.
        n_head (int): Number of attention heads.
        head_size (int): Size of each attention head.
    """

    def __init__(self, n_embd, n_head, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through the multi-head attention mechanism.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor after multi-head attention.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """
    Feed-forward network used within the transformer block.

    Args:
        n_embd (int): Dimensionality of the embeddings.
    """

    def __init__(self, n_embd):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor after the feed-forward network.
        """
        return self.feedforward(x)
    

class Block(nn.Module):
    """
    Transformer block consisting of multi-head attention and feed-forward network.

    Args:
        n_embd (int): Dimensionality of the embeddings.
        n_head (int): Number of attention heads.
        head_size (int): Size of each attention head.
    """

    def __init__(self, n_embd, n_head, head_size):
        super().__init__()

        self.attention = MultiHeadAttention(n_embd, n_head, head_size)
        self.feedforward = FeedForward(n_embd)
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.layerNorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor after the transformer block.
        """
        x = x + self.attention(self.layerNorm1(x))
        x = x + self.feedforward(self.layerNorm2(x))
        return x
    

class Transformer(nn.Module):
    """
    Transformer model for sequence modeling tasks.

    Args:
        n_embd (int): Dimensionality of the embeddings.
        n_head (int): Number of attention heads.
        head_size (int): Size of each attention head.
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_layers (int): Number of transformer blocks.
    """

    def __init__(self, n_embd, n_head, head_size, vocab_size, block_size, n_layers):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.positional_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_size) for _ in range(n_layers)])
        self.layerNorm = nn.LayerNorm(n_embd)
        self.linear_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        """
        Forward pass through the transformer model.

        Args:
            x (Tensor): Input tensor of shape (B, T).
            targets (Tensor, optional): Target tensor of shape (B, T).

        Returns:
            logits (Tensor): Logits of shape (B, T, vocab_size).
            loss (Tensor or None): Cross-entropy loss (if targets are provided).
        """
        B, T = x.shape
        token_embedding = self.token_embed(x)  # (B, T, n_embd)
        position_embedding = self.positional_embed(torch.arange(T, device=x.device))  # (T, n_embd)
        x = token_embedding + position_embedding  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.layerNorm(x)  # (B, T, n_embd)
        logits = self.linear_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for loss computation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
