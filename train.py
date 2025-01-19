import numpy as np
from pathlib import Path
import pickle as pkl

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

from network import Decoder

def train(
        batch_size: int,
        emb_size: int,
        hidden_dim: int,
        epoch_size: int,
):
    # Load in shakespeare char
    data_path = Path(".") / Path("nanoGPT") / Path("data") / Path("shakespeare_char")
    train_np = np.fromfile(data_path / Path("train.bin"), dtype=np.int8)
    val_np = np.fromfile(data_path / Path("val.bin"), dtype=np.int8)
    pkl_path = data_path / Path("meta.pkl")
    with open(pkl_path, "rb") as f:
        metadata = pkl.load(f)

    train_tensor = torch.from_numpy(train_np).long()
    val_tensor = torch.from_numpy(val_np).long()

    train = TensorDataset(train_tensor)
    val = TensorDataset(val_tensor)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False)

    # Initialize Network 
    vocab_size = metadata["vocab_size"]
    decoder = Decoder(vocab_size, emb_size, hidden_dim)

    # Initialize Optimizer
    optimizer = AdamW(decoder.parameters())

    # Training Loop
    loss_fn = nn.CrossEntropyLoss()

    epoch_dl = iter(train_loader)
    for i in range(epoch_size):
        inp = next(epoch_dl)[0]
        logits = decoder(inp)
        loss = loss_fn(logits, inp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=17)
    parser.add_argument("--emb_size", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=13)
    args = parser.parse_args()

    train(
        batch_size = args.batch_size,
        epoch_size = args.epoch_size,
        emb_size = args.emb_size,
        hidden_dim = args.hidden_dim
    )