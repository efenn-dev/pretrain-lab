#!/usr/bin/env python
"""Merge a LoRA adapter into its base model and save fp16 weights for GGUF.

Why merge instead of shipping the adapter
-----------------------------------------
llama.cpp's convert_lora_to_gguf.py cannot convert adapters for Qwen3.5:
the architecture's conversion path calls `_reorder_v_heads`, which reshapes
attention tensors, and a low-rank A/B factorisation has no equivalent
reshape (`NotImplementedError: can't reshape the row size trivially`).

Merging sidesteps it entirely — the result is an ordinary dense model, and
the base already converts cleanly. The cost is disk and one extra quantise
step rather than a 156 MB adapter you can hot-swap.

Merge happens on CPU in fp16 deliberately: the GPU is not needed, and doing
it in 4-bit would bake quantisation error into the weights before the real
Q4_K_M quantisation.

    python merge_lora.py --adapter runs/qwen35-4b-multitool-sft --out S:/merged/qwen35-4b-multitool
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("merge_lora")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=None, help="override base model id/path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = args.base or cfg["base_model_name_or_path"]
    log.info("base    : %s", base)
    log.info("adapter : %s", args.adapter)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(args.adapter), device_map="cpu")
    log.info("merging ...")
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    log.info("saving merged fp16 -> %s", args.out)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tok.save_pretrained(str(args.out))
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
