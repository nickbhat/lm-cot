import numpy as np
from pathlib import Path
import pickle as pkl

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

from data import BlockStreamDataset
from network import Decoder

def train(
        batch_size: int,
        block_size: int,
        emb_size: int,
        num_epochs: int,
        hidden_dim: int,
        epoch_size: int,
):
    # Handle paths and 
    data_path = Path(".") / Path("nanoGPT") / Path("data") / Path("shakespeare_char")
    pkl_path = data_path / Path("meta.pkl")
    with open(pkl_path, "rb") as f:
        metadata = pkl.load(f)

    train = BlockStreamDataset(data_path / Path("train.bin"), block_size)
    valid = BlockStreamDataset(data_path / Path("val.bin"), block_size)

    train_dl = DataLoader(train, batch_size)
    valid_dl = DataLoader(valid, batch_size)

    # Initialize Network 
    vocab_size = metadata["vocab_size"]
    decoder = Decoder(vocab_size, emb_size, hidden_dim)

    # Initialize Optimizer and Loss
    optimizer = AdamW(decoder.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(num_epochs):
        epoch_dl = iter(train_dl)
        for i in range(epoch_size):
            batch = next(epoch_dl)
            # Initial solution: Just slice off a token because shuffling.
            inp = batch[:, :-1]
            target = batch[:, 1:]
            logits = decoder(inp)
            logits = logits.transpose(1, 2)
            loss = loss_fn(logits, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(loss.item())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=27)
    parser.add_argument("--epoch_size", type=int, default=17)
    parser.add_argument("--emb_size", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=13)
    parser.add_argument("--num_epochs", type=int, default=20)
    args = parser.parse_args()

    train(
        batch_size = args.batch_size,
        block_size = args.block_size,
        epoch_size = args.epoch_size,
        num_epochs = args.num_epochs,
        emb_size = args.emb_size,
        hidden_dim = args.hidden_dim
    )