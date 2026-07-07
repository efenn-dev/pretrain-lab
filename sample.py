"""Generate text from a pretrain-lab checkpoint.

Example:
  python sample.py --run-dir runs/shakespeare-nano --num-tokens 400
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from train import GPT, GPTConfig

log = logging.getLogger("sample")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="dir containing ckpt.pt + meta.json")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--num-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.run_dir / "ckpt.pt", map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("loaded checkpoint from iter %d (val loss %.4f)", ckpt["iter"], ckpt["val_loss"])

    meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["mode"] == "char":
        itos: list[str] = meta["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda ids: "".join(itos[i] for i in ids)
    else:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = enc.decode

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(idx, args.num_tokens, args.temperature, args.top_k)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
