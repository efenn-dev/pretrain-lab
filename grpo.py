"""Stage 3: RL on verifiable tasks — GRPO (group-relative policy optimization).

For each task prompt, sample G completions from the policy, score each with an
exact-match verifier (a program, not a learned reward model), normalize rewards
within the group into advantages, and take clipped policy-gradient steps with a
KL penalty to the frozen starting policy. This is the frontier "RL on
verifiable rewards" recipe at hobby scale.

  python grpo.py --ckpt-dir runs/fw-124m-sft --out-dir runs/fw-124m-rl
  python grpo.py --smoke     # CPU plumbing test with a tiny random model
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from train import GPT, GPTConfig

log = logging.getLogger("grpo")

EOT = 50256
NEWLINE = 198  # GPT-2 token for "\n"


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(completion: str, task: dict) -> float:
    """Exact-match reward: 1.0 if the completion's answer matches, else 0.0."""
    text = completion.strip().lower()
    if task["type"] == "int":
        m = re.search(r"-?\d+", text)
        return 1.0 if m and m.group() == task["answer"] else 0.0
    if task["type"] == "int_list":
        nums = re.findall(r"-?\d+", text.split("\n")[0])
        return 1.0 if ", ".join(nums) == task["answer"] else 0.0
    if task["type"] == "yesno":
        m = re.search(r"[a-z]+", text)
        return 1.0 if m and m.group() == task["answer"] else 0.0
    return 0.0


def shaped_reward(completion: str, task: dict) -> float:
    """Partial credit for near misses (capped at 0.5 so exact answers dominate).

    Turns the sparse binary reward dense: a policy that answers 6 to "59 - 52"
    gets pulled toward 7 instead of learning nothing.
    """
    if verify(completion, task) == 1.0:
        return 1.0
    text = completion.strip().lower()
    if task["type"] == "int":
        if m := re.search(r"-?\d+", text):
            pred, true = int(m.group()), int(task["answer"])
            return 0.5 * max(0.0, 1.0 - abs(pred - true) / max(1.0, abs(true)))
    elif task["type"] == "int_list":
        nums = re.findall(r"-?\d+", text.split("\n")[0])
        want = task["answer"].split(", ")
        hits = sum(1 for a, b in zip(nums, want) if a == b)
        return 0.5 * hits / len(want)
    return 0.0


@torch.no_grad()
def sample_group(
    model: GPT, prompt_ids: list[int], group: int, max_new: int,
    temperature: float, device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `group` completions for one prompt.

    Returns (sequences (G, P+max_new), completion lengths (G,), old logprobs
    (G, max_new) — positions past a sequence's end hold 0 and are masked later).
    """
    G, P = group, len(prompt_ids)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).repeat(G, 1)
    old_lp = torch.zeros(G, max_new, device=device)
    finished = torch.zeros(G, dtype=torch.bool, device=device)
    lengths = torch.full((G,), max_new, dtype=torch.long, device=device)
    for t in range(max_new):
        window = idx[:, -model.cfg.block_size :]
        logits, _ = model(window)  # (G, 1, V) — last position only
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, num_samples=1).squeeze(1)
        tok = torch.where(finished, torch.full_like(tok, EOT), tok)
        old_lp[:, t] = torch.where(
            finished, torch.zeros(G, device=device),
            torch.log(probs.gather(1, tok[:, None]).squeeze(1) + 1e-10),
        )
        idx = torch.cat([idx, tok[:, None]], dim=1)
        just_ended = (~finished) & ((tok == EOT) | (tok == NEWLINE))
        lengths = torch.where(just_ended, torch.full_like(lengths, t + 1), lengths)
        finished = finished | just_ended
        if bool(finished.all()):
            idx = torch.cat(
                [idx, torch.full((G, max_new - t - 1), EOT, dtype=torch.long, device=device)],
                dim=1,
            )
            break
    return idx, lengths, old_lp


def completion_logprobs(
    model: GPT, seqs: torch.Tensor, prompt_len: int, max_new: int
) -> torch.Tensor:
    """Logprob of each sampled completion token under `model` — (G, max_new)."""
    logits, _ = model(seqs, seqs)  # targets passed so forward returns full logits
    logp = F.log_softmax(logits.float(), dim=-1)
    # token at position prompt_len + t is predicted by logits at prompt_len + t - 1
    pred_pos = torch.arange(prompt_len - 1, prompt_len - 1 + max_new, device=seqs.device)
    tgt = seqs[:, prompt_len : prompt_len + max_new]
    return logp[:, pred_pos, :].gather(2, tgt[:, :, None]).squeeze(2)


@torch.no_grad()
def evaluate(model: GPT, tasks: list[dict], enc, device: str, max_new: int) -> float:
    """Greedy-decode accuracy over an eval task list."""
    correct = 0
    for task in tasks:
        ids = enc.encode_ordinary(f"User: {task['prompt']}\n\nAssistant:")
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits, _ = model(idx[:, -model.cfg.block_size :])
            tok = logits[:, -1, :].argmax(dim=-1)
            idx = torch.cat([idx, tok[:, None]], dim=1)
            if tok.item() in (EOT, NEWLINE):
                break
        completion = enc.decode(idx[0, len(ids) :].tolist())
        correct += verify(completion, task)
    return correct / len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, help="SFT checkpoint to start from")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tasks-dir", type=Path, default=Path("data/rl_tasks"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group", type=int, default=8, help="completions sampled per prompt")
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-tasks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shaped", action="store_true",
                        help="partial-credit rewards for near misses (eval stays exact-match)")
    parser.add_argument("--smoke", action="store_true", help="CPU plumbing test, random tiny model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    if args.smoke:
        device = "cpu"
        cfg = GPTConfig(block_size=128, vocab_size=50304, n_layer=2, n_head=2, n_embd=64)
        policy, ref = GPT(cfg).to(device), GPT(cfg).to(device)
        ref.load_state_dict(policy.state_dict())
        args.steps, args.prompts_per_step, args.group, args.max_new = 2, 2, 4, 8
        args.eval_every, args.eval_tasks = 1, 4
        args.out_dir = None
    else:
        if not (args.ckpt_dir and args.out_dir):
            raise SystemExit("--ckpt-dir and --out-dir are required (or use --smoke)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.ckpt_dir / "ckpt.pt", map_location=device, weights_only=True)
        cfg = GPTConfig(**ckpt["config"])
        policy, ref = GPT(cfg).to(device), GPT(cfg).to(device)
        policy.load_state_dict(ckpt["model"])
        ref.load_state_dict(ckpt["model"])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "meta.json").write_text(json.dumps({"mode": "gpt2"}), encoding="utf-8")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    train_tasks = load_tasks(args.tasks_dir / "train.jsonl")
    eval_tasks = load_tasks(args.tasks_dir / "eval.jsonl")[: args.eval_tasks]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rng = torch.Generator().manual_seed(args.seed)

    policy.eval()
    baseline = evaluate(policy, eval_tasks, enc, device, args.max_new)
    policy.train()
    log.info("step 0: baseline eval accuracy %.4f (before any RL)", baseline)

    best_acc = 0.0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        picks = torch.randint(0, len(train_tasks), (args.prompts_per_step,), generator=rng)
        step_rewards: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        groups_used = 0
        for pi in picks.tolist():
            task = train_tasks[pi]
            prompt_ids = enc.encode_ordinary(f"User: {task['prompt']}\n\nAssistant:")
            policy.eval()
            seqs, lengths, old_lp = sample_group(
                policy, prompt_ids, args.group, args.max_new, args.temperature, device
            )
            policy.train()
            P = len(prompt_ids)
            reward_fn = shaped_reward if args.shaped else verify
            rewards = torch.tensor(
                [reward_fn(enc.decode(seqs[g, P : P + lengths[g]].tolist()), task)
                 for g in range(args.group)],
                device=device,
            )
            step_rewards.extend(rewards.tolist())
            if args.smoke:
                rewards = torch.rand(args.group, device=device)  # exercise the gradient path
            if rewards.std() < 1e-6:
                continue  # all same reward -> zero advantage, no gradient signal
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

            mask = (
                torch.arange(args.max_new, device=device)[None, :] < lengths[:, None]
            ).float()
            pol_lp = completion_logprobs(policy, seqs, P, args.max_new)
            with torch.no_grad():
                ref_lp = completion_logprobs(ref, seqs, P, args.max_new)
            ratio = torch.exp(pol_lp - old_lp)
            surr = torch.min(
                ratio * adv[:, None],
                ratio.clamp(1 - args.clip, 1 + args.clip) * adv[:, None],
            )
            kl = torch.exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1  # k3 estimator
            denom = mask.sum().clamp(min=1.0)
            loss = (-(surr * mask).sum() + args.kl_beta * (kl * mask).sum()) / denom
            (loss / args.prompts_per_step).backward()
            groups_used += 1

        if groups_used:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        mean_r = sum(step_rewards) / max(1, len(step_rewards))
        log.info("step %d: reward %.3f, %d/%d groups informative, %.1fs",
                 step, mean_r, groups_used, args.prompts_per_step, time.time() - t0)

        if step % args.eval_every == 0 or step == args.steps:
            policy.eval()
            acc = evaluate(policy, eval_tasks, enc, device, args.max_new)
            policy.train()
            log.info("step %d: eval accuracy %.4f (best %.4f)", step, acc, best_acc)
            if args.out_dir and acc > best_acc:
                best_acc = acc
                torch.save(
                    {"model": policy.state_dict(), "config": asdict(cfg),
                     "iter": step, "val_loss": 1.0 - acc},
                    args.out_dir / "ckpt.pt",
                )

    log.info("done — best eval accuracy %.4f", best_acc)


if __name__ == "__main__":
    main()
