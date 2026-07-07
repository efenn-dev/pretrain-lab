"""Generate worked-example SFT data for the synthetic tasks (capability seeding).

Emits chat-formatted examples

    User: What is 47 + 38?

    Assistant: 85
    <|endoftext|>

tokenized to a uint16 bin, mixed ~50/50 with existing SmolTalk tokens so the
model keeps its general chat ability. Answers lead with the bare result so the
RL verifier's first-number extraction sees them. Exact eval-set questions are
excluded so the RL eval stays honest.

  python gen_task_sft.py --out data/task_sft_mix.bin --n-examples 120000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from gen_rl_tasks import GENERATORS

log = logging.getLogger("gen_task_sft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/task_sft_mix.bin"))
    parser.add_argument("--n-examples", type=int, default=120_000)
    parser.add_argument("--eval-file", type=Path, default=Path("data/rl_tasks/eval.jsonl"),
                        help="questions to exclude from training data")
    parser.add_argument("--smoltalk-bin", type=Path, default=Path("data/smoltalk.bin"))
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(args.seed)
    banned = {
        json.loads(line)["prompt"]
        for line in args.eval_file.read_text(encoding="utf-8").splitlines() if line
    }

    tokens: list[int] = []
    made = skipped = 0
    while made < args.n_examples:
        q, a, _ = rng.choice(GENERATORS)(rng)
        if q in banned:
            skipped += 1
            continue
        tokens.extend(enc.encode_ordinary(f"User: {q}\n\nAssistant: {a}\n"))
        tokens.append(enc.eot_token)
        made += 1
    task_tokens = np.array(tokens, dtype=np.uint16)
    log.info("%d task examples -> %d tokens (%d eval collisions excluded)",
             made, len(task_tokens), skipped)

    smoltalk = np.memmap(args.smoltalk_bin, dtype=np.uint16, mode="r")
    n_mix = min(len(task_tokens), len(smoltalk))
    # tasks LAST: sft.py's val split is the tail of the file, and checkpointing
    # keys on val loss — a pure-chat tail would save the pre-training model forever
    mixed = np.concatenate([np.asarray(smoltalk[:n_mix]), task_tokens])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mixed.tofile(args.out)
    log.info("wrote %s: %d tokens (%.0f%% tasks, %.0f%% chat)",
             args.out, len(mixed), 100 * len(task_tokens) / len(mixed),
             100 * n_mix / len(mixed))


if __name__ == "__main__":
    main()
