import torch
from torch import nn

class Decoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.ff = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h = self.ff(e)
        logits = self.out(h)
        return logits