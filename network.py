import torch
from torch import nn


class MLP(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            dropout: float = 0.2,
        ):
        super().__init__()
        self.ff = nn.Linear(hidden_dim, 4*hidden_dim)
        self.proj = nn.Linear(4*hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.ff(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            dropout: float,
        ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.MLP = MLP(hidden_dim, dropout)

    def forward(self, x):
        x = x + self.MLP(self.ln_1(x))
        return x


class Decoder(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            dropout: float,
            num_layers: int,
        ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            Block(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.logit_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.emb_proj(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
        logits = self.logit_layer(x)
        return logits