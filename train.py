import math
from pathlib import Path
import pickle as pkl

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import BlockStreamDataset
from network import Decoder


def get_lr(lr: float, min_lr: float, it: int, warmup_iters: int, lr_decay_iters: int):
    assert warmup_iters < lr_decay_iters
    # Linear warmup
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    # Constant lr at end
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1. + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def train(
        batch_size: int,
        max_iters: int,
        block_size: int,
        emb_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        lr: float,
        min_lr: float,
        warmup_iters: int,
        lr_decay_iters: int,
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
    model_args = {
        "vocab_size": vocab_size,
        "embedding_dim": emb_size,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "block_size": block_size,
        "dropout": dropout,
        "num_layers": num_layers
    }
    decoder = Decoder(**model_args)

    # Initialize Optimizer and Loss
    optimizer = decoder.configure_optimizers(
        weight_decay = 1e-1, lr = lr, betas = (0.9, 0.99)
    )
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

        lr = get_lr(lr, min_lr, it, warmup_iters, lr_decay_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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
                "loss": loss.item(),
                "model_args": model_args
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
        num_layers = args.num_layers,
        dropout = 0.0,
        lr = 1e-3,
        min_lr = 1e-4,
        warmup_iters = 100,
        lr_decay_iters = args.max_iters,
    )