"""GSM8K accuracy for a Hugging Face causal LM (optionally with a LoRA adapter).

Greedy decoding, non-thinking mode, strict answer matching on the number after
'####'. Writes results to runs/gsm8k-evals/<tag>.json.

  python eval_gsm8k.py --model Qwen/Qwen3.5-4B --tag baseline --max-examples 200
  python eval_gsm8k.py --model Qwen/Qwen3.5-4B --adapter runs/qwen35-4b-gsm8k-grpo --tag after-rl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch

log = logging.getLogger("gsm8k")

SYSTEM = (
    "Solve the math problem. Show brief reasoning, then give the final numeric "
    "answer on the last line in exactly this form: #### <number>"
)


def extract_gold(answer: str) -> str:
    """Gold answer: the number after '####' in the GSM8K reference solution."""
    return answer.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> str | None:
    """Model answer: number after '####' if present, else the last number."""
    tail = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # tokenizer without the enable_thinking kwarg
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_id: str, adapter: Path | None, four_bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant = None
    if four_bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return tok, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-new", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from datasets import load_dataset

    tok, model = load_model(args.model, args.adapter, not args.no_4bit)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.max_examples))
    log.info("evaluating %s%s on %d GSM8K test problems",
             args.model, f" + {args.adapter}" if args.adapter else "", len(ds))

    correct = done = 0
    for start in range(0, len(ds), args.batch_size):
        rows = ds.select(range(start, min(start + args.batch_size, len(ds))))
        prompts = [build_prompt(tok, q) for q in rows["question"]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        completions = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for completion, gold_raw in zip(completions, rows["answer"]):
            correct += int(extract_pred(completion) == extract_gold(gold_raw))
            done += 1
        log.info("%d/%d: accuracy %.4f", done, len(ds), correct / done)

    result = {"model": args.model, "adapter": str(args.adapter) if args.adapter else None,
              "n": done, "accuracy": round(correct / done, 4)}
    out_dir = Path("runs/gsm8k-evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(result), encoding="utf-8")
    log.info("final: %s", result)


if __name__ == "__main__":
    main()
