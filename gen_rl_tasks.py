"""Generate synthetic verifiable tasks for RL (stage 3).

Every task has an exactly-checkable answer, so the RL reward is a program, not
a learned model — the "RL on verifiable rewards" recipe at a scale a 124M model
can learn from. Task types: 2/3-digit arithmetic, max-of-list, sort-3,
sequence continuation, greater-than, and letter counting.

  python gen_rl_tasks.py --out-dir data/rl_tasks --n-train 20000 --n-eval 500
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

log = logging.getLogger("gen_rl")

WORDS = [
    "elephant", "banana", "committee", "parallel", "mississippi", "bookkeeper",
    "assessment", "cheese", "letter", "balloon", "coffee", "pepper", "summer",
    "puzzle", "little", "engineer", "success", "tomorrow", "different", "happiness",
]


def gen_add2(rng: random.Random) -> tuple[str, str, str]:
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"What is {a} + {b}?", str(a + b), "int"


def gen_add3(rng: random.Random) -> tuple[str, str, str]:
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return f"What is {a} + {b}?", str(a + b), "int"


def gen_sub2(rng: random.Random) -> tuple[str, str, str]:
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if b > a:
        a, b = b, a
    return f"What is {a} - {b}?", str(a - b), "int"


def gen_max3(rng: random.Random) -> tuple[str, str, str]:
    nums = rng.sample(range(1, 100), 3)
    return f"Which number is largest: {nums[0]}, {nums[1]}, {nums[2]}?", str(max(nums)), "int"


def gen_sort3(rng: random.Random) -> tuple[str, str, str]:
    nums = rng.sample(range(1, 100), 3)
    ans = ", ".join(str(n) for n in sorted(nums))
    return (
        f"Sort these numbers from smallest to largest: {nums[0]}, {nums[1]}, {nums[2]}.",
        ans, "int_list",
    )


def gen_next_seq(rng: random.Random) -> tuple[str, str, str]:
    start, step = rng.randint(1, 20), rng.randint(2, 12)
    seq = [start + i * step for i in range(4)]
    return (
        f"What number comes next: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?",
        str(seq[3] + step), "int",
    )


def gen_compare(rng: random.Random) -> tuple[str, str, str]:
    a, b = rng.sample(range(1, 200), 2)
    return f"Is {a} greater than {b}? Answer yes or no.", "yes" if a > b else "no", "yesno"


def gen_count_letter(rng: random.Random) -> tuple[str, str, str]:
    word = rng.choice(WORDS)
    letter = rng.choice(sorted(set(word)))
    return (
        f"How many times does the letter '{letter}' appear in the word '{word}'?",
        str(word.count(letter)), "int",
    )


GENERATORS = [gen_add2, gen_add3, gen_sub2, gen_max3, gen_sort3, gen_next_seq,
              gen_compare, gen_count_letter]


def write_split(path: Path, n: int, rng: random.Random, seen: set[str]) -> None:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        while count < n:
            q, a, t = rng.choice(GENERATORS)(rng)
            if q in seen:
                continue
            seen.add(q)
            f.write(json.dumps({"prompt": q, "answer": a, "type": t}) + "\n")
            count += 1
    log.info("wrote %d tasks to %s", count, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/rl_tasks"))
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    write_split(args.out_dir / "eval.jsonl", args.n_eval, rng, seen)  # eval first, so train never overlaps
    write_split(args.out_dir / "train.jsonl", args.n_train, rng, seen)


if __name__ == "__main__":
    main()
