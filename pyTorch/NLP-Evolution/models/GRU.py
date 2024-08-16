import torch 
import torch.nn as nn
from torch.nn import functional as F

class GRULanguageModel(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, target=None):

        x = self.embed(x)

        x, _ = self.gru(x)

        logits = self.fc(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    