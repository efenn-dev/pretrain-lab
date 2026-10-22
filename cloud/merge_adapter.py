"""Merge the GRPO LoRA adapter into the base weights → a standalone bf16 model.

Produces an adapter-free checkpoint you can serve directly (transformers/vLLM) without
carrying the base + adapter separately. Run on a Linux GPU pod with the kernels installed
(the model's __init__ imports flash-linear-attention).

  python cloud/merge_adapter.py --adapter runs/qwen35-4b-gsm8k-grpo --out runs/qwen35-4b-gsm8k-grpo-merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", type=Path, default=Path("runs/qwen35-4b-gsm8k-grpo"))
    p.add_argument("--out", type=Path, default=Path("runs/qwen35-4b-gsm8k-grpo-merged"))
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Merge into a bf16 base (NOT 4-bit) — the standard QLoRA→merged path; LoRA deltas
    # apply to the same weight matrices regardless of the precision used during training.
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tok.save_pretrained(str(args.out))
    print("MERGED_OK ->", args.out)


if __name__ == "__main__":
    main()
