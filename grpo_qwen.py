"""Lane (b): GRPO on real math (GSM8K) with Qwen3.5-4B via TRL + QLoRA.

The same recipe as grpo.py but industrial-strength: TRL's GRPOTrainer, 4-bit
base weights with LoRA adapters, and a real benchmark. Rewards: exact final
answer (1.0) + '####' format compliance (0.2).

  python grpo_qwen.py --out-dir runs/qwen35-4b-gsm8k-grpo
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from eval_gsm8k import SYSTEM, build_prompt, extract_gold, extract_pred

log = logging.getLogger("grpo_qwen")


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    return [1.0 if extract_pred(c) == a else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    return [0.2 if "####" in c else 0.0 for c in completions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/qwen35-4b-gsm8k-grpo"))
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--prompts-per-update", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-prompt", type=int, default=512)
    parser.add_argument("--max-completion", type=int, default=384)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--train-examples", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(args.train_examples))
    ds = ds.map(
        lambda ex: {"prompt": build_prompt(tok, ex["question"]), "answer": extract_gold(ex["answer"])},
        remove_columns=ds.column_names,
    )
    log.info("train prompts: %d (example prompt tail: %r)", len(ds), ds[0]["prompt"][-120:])

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    cfg = GRPOConfig(
        output_dir=str(args.out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.num_generations,  # one prompt group per micro-batch
        gradient_accumulation_steps=args.prompts_per_update,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        max_steps=args.max_steps,
        beta=args.beta,
        logging_steps=1,
        save_steps=25,  # 50 left a 14h run with zero checkpoints at step 31 — never again
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        model_init_kwargs={"quantization_config": bnb, "dtype": torch.bfloat16},
    )
    trainer = GRPOTrainer(
        model=args.model, args=cfg, train_dataset=ds,
        reward_funcs=[correctness_reward, format_reward], peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    log.info("saved adapter to %s", args.out_dir)


if __name__ == "__main__":
    main()
