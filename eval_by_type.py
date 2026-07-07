"""Per-category accuracy breakdown on the synthetic RL eval set.

Compares checkpoints (e.g., SFT vs RL) to show which task categories RL
actually improved. Categories are inferred from prompt wording.

  python eval_by_type.py --run-dirs runs/fw-124m-sft runs/fw-124m-rl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from grpo import EOT, NEWLINE, load_tasks, verify
from train import GPT, GPTConfig

log = logging.getLogger("eval_by_type")

CATEGORY_KEYWORDS = [
    ("Sort these", "sort-3"),  # must precede "largest" — sort prompts contain that word too
    ("largest", "max-of-3"),
    ("greater than", "compare y/n"),
    ("comes next", "sequence"),
    ("letter", "count letter"),
    (" - ", "subtract"),
    (" + ", "add"),
]


def categorize(prompt: str) -> str:
    for kw, name in CATEGORY_KEYWORDS:
        if kw in prompt:
            return name
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/rl_tasks/eval.jsonl"))
    parser.add_argument("--max-tasks", type=int, default=256)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--show", type=int, default=0, help="print the first N raw completions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tasks = load_tasks(args.tasks)[: args.max_tasks]

    for run_dir in args.run_dirs:
        ckpt = torch.load(run_dir / "ckpt.pt", map_location=device, weights_only=True)
        model = GPT(GPTConfig(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        log.info("%s: checkpoint from iter %d (val loss %.3f)", run_dir, ckpt["iter"], ckpt["val_loss"])
        shown = 0
        by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for task in tasks:
                ids = enc.encode_ordinary(f"User: {task['prompt']}\n\nAssistant:")
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                for _ in range(args.max_new):
                    logits, _ = model(idx[:, -model.cfg.block_size :])
                    tok = logits[:, -1, :].argmax(dim=-1)
                    idx = torch.cat([idx, tok[:, None]], dim=1)
                    if tok.item() in (EOT, NEWLINE):
                        break
                completion = enc.decode(idx[0, len(ids) :].tolist())
                if shown < args.show:
                    log.info("  Q: %s | expect %s | got: %r", task["prompt"], task["answer"], completion)
                    shown += 1
                cat = categorize(task["prompt"])
                by_cat[cat][0] += int(verify(completion, task))
                by_cat[cat][1] += 1
        total = sum(v[0] for v in by_cat.values()), sum(v[1] for v in by_cat.values())
        log.info("\n%s  (overall %d/%d = %.1f%%)", run_dir, total[0], total[1],
                 100 * total[0] / total[1])
        for cat in sorted(by_cat):
            hit, n = by_cat[cat]
            log.info("  %-14s %3d/%-3d  %5.1f%%", cat, hit, n, 100 * hit / n)


if __name__ == "__main__":
    main()
