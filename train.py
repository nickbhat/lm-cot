from pathlib import Path
import pickle as pkl

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import BlockStreamDataset
from network import Decoder

def train(
        batch_size: int,
        max_iters: int,
        block_size: int,
        emb_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
):
    # Handle paths and 
    data_path = Path(".") / Path("nanoGPT") / Path("data") / Path("shakespeare_char")
    pkl_path = data_path / Path("meta.pkl")
    with open(pkl_path, "rb") as f:
        metadata = pkl.load(f)

    train = BlockStreamDataset(data_path / Path("train.bin"), block_size)
    valid = BlockStreamDataset(data_path / Path("val.bin"), block_size)

    train_dl = iter(DataLoader(train, batch_size))
    valid_dl = iter(DataLoader(valid, batch_size))

    # Initialize Network 
    vocab_size = metadata["vocab_size"]
    decoder = Decoder(
        vocab_size, 
        emb_size, 
        hidden_dim, 
        num_heads, 
        block_size, 
        0.2, 
        num_layers
    )

    # Initialize Optimizer and Loss
    optimizer = AdamW(decoder.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    print("Training")
    it = 0
    while True:
        batch = next(train_dl)
        # Initial solution: Just slice off a token because shuffling.
        inp = batch[:, :-1]
        target = batch[:, 1:]
        logits = decoder(inp)
        logits = logits.transpose(1, 2)
        loss = loss_fn(logits, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        it += 1

        if it % 50 == 0:
            print(f"Iter: {it}, Loss: {loss.item()}") 

        if it > max_iters:
            checkpoint = {
                "model": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": it,
                "loss": loss.item()
            }
            out_dir = Path("checkpoints")
            torch.save(checkpoint, out_dir / Path("ckpt.pt"))
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=27)
    parser.add_argument("--emb_size", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    args = parser.parse_args()

    train(
        batch_size = args.batch_size,
        max_iters = args.max_iters,
        block_size = args.block_size,
        emb_size = args.emb_size,
        hidden_dim = args.hidden_dim,
        num_heads = args.num_heads,
        num_layers = args.num_layers
    )