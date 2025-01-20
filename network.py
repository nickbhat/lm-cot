import math

import torch
from torch import nn
import torch.nn.functional as F


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


class CausalAttention(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            block_size: int,
            dropout: float,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(
                block_size, block_size
            )).view(1, 1, block_size, block_size) 
        )

    def forward(self, x):
        B, L, D = x.size()

        q, k, v = self.qkv_proj(x).split(
            self.hidden_dim, dim=2
        )
        q = q.view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = k.view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = v.view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        scores = q @ k.transpose(-2, -1)
        scores /= math.sqrt(k.size(-1))
        scores = scores.masked_fill(
            self.causal_mask[:, :, :L, :L] == 0,
            float("-inf")
        )
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)


        outs = probs @ v
        outs = outs.transpose(1, 2).contiguous().view(B, L, D)
        outs = self.resid_dropout(self.out_proj(outs))
        
        return outs


class Block(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            block_size: int,
            dropout: float,
        ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalAttention(hidden_dim, num_heads, block_size, dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.MLP = MLP(hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.MLP(self.ln_2(x))
        return x


class Decoder(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_heads: int,
            block_size: int,
            dropout: float,
            num_layers: int,
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            Block(hidden_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.logit_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.emb_proj(x)
        for block in self.blocks:
            x = block(x)
        logits = self.logit_layer(x)
        return logits
    
    def sample(self, x, max_tokens, temp, topk):
        y = x.unsqueeze(0)
        for _ in range(max_tokens):
            # Crop context
            ctx = y if y.size(1) <= self.block_size else y[:, -self.block_size:]
            logits = self(ctx)

            # Calculate probs for last position
            logits = logits[:, -1, :] / temp
            tops, _ = torch.topk(logits, min(topk, self.vocab_size))
            logits[logits < tops[:, [-1]]] = float("-Inf")
            probs = F.softmax(logits, dim=-1)

            # Sample tokens and add to end
            token = torch.multinomial(probs, num_samples=1)
            y = torch.cat((y, token), dim=1)

        return y