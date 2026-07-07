"""Stage 2: supervised fine-tuning (SFT) — turn a pretrained checkpoint into a tiny chat model.

Streams SmolTalk conversations (built for small models), formats them as

    User: ...

    Assistant: ...
    <|endoftext|>

and continues next-token training at a low learning rate. Loss is computed on
all tokens (no user-turn masking) — simple and fine at this scale.

Run AFTER pretraining finishes (needs the GPU and the checkpoint):
  python sft.py --ckpt-dir runs/fw-124m --out-dir runs/fw-124m-sft
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from train import GPT, GPTConfig, estimate_loss, get_batch, lr_at

log = logging.getLogger("sft")

ROLE_NAMES = {"system": "System", "user": "User", "assistant": "Assistant"}


def prepare_sft_data(cache: Path, dataset: str, max_tokens: int) -> None:
    """Stream + tokenize a chat dataset into a flat uint16 token file."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(dataset, split="train", streaming=True)
    written = 0
    buf: list[int] = []
    with cache.open("wb") as f:
        for row in ds:
            parts = [f"{ROLE_NAMES.get(m['role'], 'User')}: {m['content']}" for m in row["messages"]]
            buf.extend(enc.encode_ordinary("\n\n".join(parts) + "\n"))
            buf.append(enc.eot_token)
            if len(buf) >= 1_000_000:
                np.array(buf, dtype=np.uint16).tofile(f)
                written += len(buf)
                buf.clear()
                log.info("%.0fM tokens prepared", written / 1e6)
            if written >= max_tokens:
                break
        if buf and written < max_tokens:
            np.array(buf, dtype=np.uint16).tofile(f)
    log.info("SFT data cached at %s", cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="dir with the pretrained ckpt.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--data-bin", type=Path, default=None,
                        help="pre-tokenized uint16 bin to train on instead of streaming --dataset")
    parser.add_argument("--max-tokens", type=int, default=30_000_000)
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.loads((args.ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["mode"] != "gpt2":
        raise SystemExit("SFT needs a token-mode (gpt2) checkpoint, not a char-level one")
    ckpt = torch.load(args.ckpt_dir / "ckpt.pt", map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    log.info("loaded %s (iter %d, val loss %.3f)", args.ckpt_dir, ckpt["iter"], ckpt["val_loss"])

    if args.data_bin:
        cache = args.data_bin
    else:
        cache = Path("data") / "smoltalk.bin"
        if not cache.exists():
            prepare_sft_data(cache, args.dataset, args.max_tokens)
    tokens = np.memmap(cache, dtype=np.uint16, mode="r")
    n = int(0.99 * len(tokens))
    train_data, val_data = tokens[:n], tokens[n:]
    log.info("SFT corpus: %d train / %d val tokens", n, len(tokens) - n)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "meta.json").write_text(json.dumps({"mode": "gpt2"}), encoding="utf-8")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda",
    )

    best_val = float("inf")
    block = cfg.block_size
    for it in range(args.max_iters + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(it, args.max_iters, args.lr, args.warmup_iters)

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(
                model, {"train": train_data, "val": val_data},
                block, args.batch_size, device, args.eval_iters,
            )
            log.info("iter %d: train loss %.4f, val loss %.4f", it, losses["train"], losses["val"])
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": asdict(cfg),
                     "iter": it, "val_loss": best_val},
                    args.out_dir / "ckpt.pt",
                )
        if it == args.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, block, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it % 50 == 0:
            log.info("iter %d: loss %.4f", it, loss.item())

    log.info("SFT done — best val loss %.4f", best_val)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    prompt = "User: What is a GPU?\n\nAssistant:"
    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(idx, 120, temperature=0.7, top_k=100)
    log.info("demo:\n%s", enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
