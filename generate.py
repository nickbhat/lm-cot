from pathlib import Path
import pickle as pkl

import torch

from network import Decoder


def main(
        checkpoint_path: Path, 
        prompt_tokens: torch.LongTensor,
        itos,
    ):
    checkpoint = torch.load(checkpoint_path)
    model = Decoder(**checkpoint["model_args"])
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    with torch.no_grad():
        sampled = model.sample(
            prompt_tokens,
            max_tokens = 500,
            temp = 0.8,
            topk = 200
        )

    raw_toks = sampled.tolist()[0]
    encoded_sample = [itos[i] for i in raw_toks]
    encoded_str = "".join(encoded_sample)
    print(encoded_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    tmp = Path(args.checkpoint_dir)

    if not tmp.exists():
        raise FileNotFoundError
    
    checkpoint_path = tmp / Path("ckpt.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found")
    
    # Encode prompt
    pkl_path = Path(".") / Path("nanoGPT") / Path("data") / Path("shakespeare_char") / Path("meta.pkl")
    with open(pkl_path, "rb") as f:
        metadata = pkl.load(f)
    prompt = args.prompt
    stoi = metadata["stoi"]
    itos = metadata["itos"]
    prompt_tokens = torch.LongTensor([stoi[s] for s in prompt])

    main(checkpoint_path, prompt_tokens, itos)