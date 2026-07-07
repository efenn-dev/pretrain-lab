"""Download + tokenize FineWeb-Edu into train.bin / val.bin for token-mode training.

FineWeb-Edu is the open web-crawl dataset (education-filtered) that modern
small-model pretraining recipes use. This streams the sample-10BT subset and
writes uint16 GPT-2 tokens. Full 10B tokens is ~20 GB on disk — use
--max-tokens for a smaller slice first.

Requires:  pip install datasets tiktoken

Example (2B-token slice, enough for a chinchilla-optimal 100M model):
  python prepare_fineweb.py --out-dir data/fineweb --max-tokens 2000000000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.0005)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    tokens_written = 0
    buf: list[int] = []
    tmp_path = args.out_dir / "all.bin"
    with tmp_path.open("wb") as f:
        for doc in ds:
            buf.extend(enc.encode_ordinary(doc["text"]))
            buf.append(eot)
            if len(buf) >= 1_000_000:
                np.array(buf, dtype=np.uint16).tofile(f)
                tokens_written += len(buf)
                buf.clear()
                if tokens_written % 100_000_000 < 1_000_000:
                    log.info("%.0fM tokens written", tokens_written / 1e6)
            if tokens_written >= args.max_tokens:
                break
        if buf and tokens_written < args.max_tokens:
            np.array(buf, dtype=np.uint16).tofile(f)
            tokens_written += len(buf)

    # split off a small validation tail
    all_tokens = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    n_val = max(1_000_000, int(len(all_tokens) * args.val_fraction))
    all_tokens[: len(all_tokens) - n_val].tofile(args.out_dir / "train.bin")
    all_tokens[len(all_tokens) - n_val :].tofile(args.out_dir / "val.bin")
    del all_tokens
    tmp_path.unlink()
    log.info("done: %d train + %d val tokens in %s", tokens_written - n_val, n_val, args.out_dir)


if __name__ == "__main__":
    main()
