#!/usr/bin/env python
"""QLoRA SFT: teach Qwen3.5-4B to emit PARALLEL tool calls.

Measured baseline (local-model-eval, category `multi_tool_use`, 2026-08-21):

    5/14 passed   easy 0/4   medium 1/5   hard 4/5
    calls emitted per turn: {0: 3, 1: 11}

The model never emitted two calls in one turn. Every call it did make named
the right tool with the right arguments, and it correctly declined to call
anything on the "don't act" cases. So this adapter targets ONE behaviour —
parallel emission — and the corpus deliberately keeps single-call and no-call
examples so the two things that already work do not regress.

Same recipe as grpo_qwen.py: 4-bit nf4 base + LoRA r=16/alpha=32 on all-linear.

    python sft_multitool.py --dry-run          # render 3 examples, no GPU
    python sft_multitool.py --out-dir runs/qwen35-4b-multitool-sft

GPU note: the inference stack holds ~8GB. Free it before a real run —
create C:/projects/llm-router/services.off, stop the :11437 server, and
delete services.off afterwards so the keeper brings it back.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("sft_multitool")

BASE_MODEL = "Qwen/Qwen3.5-4B"
RESPONSE_MARKER = "<|im_start|>assistant"


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no examples in {path}")
    return rows


def render(rows: list[dict], tok) -> list[dict]:
    """Render each example through the model's OWN chat template.

    Hand-rolling the <tool_call> syntax is how you train a model to speak a
    dialect its server cannot parse. The template that llama.cpp uses at
    inference is the one that must produce the training text.
    """
    out = []
    for r in rows:
        # Split into prompt / completion so TRL can mask the prompt
        # (completion_only_loss). TRL 1.7 dropped DataCollatorForCompletionOnly
        # LM, and assistant_only_loss needs `{% generation %}` markers this
        # template does not have — so derive the boundary from the template
        # itself rather than string-matching a marker.
        prompt = tok.apply_chat_template(
            r["messages"][:-1], tools=r.get("tools"), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        text = tok.apply_chat_template(
            r["messages"], tools=r.get("tools"), tokenize=False,
            add_generation_prompt=False,
            # Train/inference parity. Production serves this model with
            # LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":false} (see
            # llm-router/scripts/serve-chat-server.ps1). Rendering the corpus
            # with thinking ON would train the adapter against a prompt shape
            # — an empty <think></think> block before every answer — that the
            # server never produces.
            enable_thinking=False,
        )
        if not text.startswith(prompt):
            raise SystemExit("prompt is not a prefix of the rendered turn - "
                             "template drift; refusing to train on a bad split")
        out.append({"prompt": prompt, "completion": text[len(prompt):],
                    "text": text, "kind": r.get("kind", "?")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/multitool_sft.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/qwen35-4b-multitool-sft"))
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true",
                    help="render a few examples and exit (no GPU, no training)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    from transformers import AutoTokenizer

    rows = load_rows(args.data)
    tok = AutoTokenizer.from_pretrained(args.model)
    rendered = render(rows, tok)

    if args.dry_run:
        # Count ONLY inside the assistant turn. The template's system prompt
        # contains the literal string "<tool_call>" as an instruction to the
        # model, so a naive whole-text count reports every example — including
        # the no-call ones — as multi-call. That would have "verified" a corpus
        # without ever looking at what the assistant actually says.
        def assistant_part(text: str) -> str:
            return text.rsplit(RESPONSE_MARKER, 1)[-1] if RESPONSE_MARKER in text else ""

        for kind in ("multi3", "multi2", "single", "none"):
            ex = next((r for r in rendered if r["kind"] == kind), None)
            if not ex:
                continue
            print(f"\n{'=' * 70}\n{kind}  (assistant turn only)\n{'=' * 70}")
            print(assistant_part(ex["text"])[:700])
        n_tc = sum(1 for r in rendered if "<tool_call>" in assistant_part(r["text"]))
        multi = sum(1 for r in rendered if assistant_part(r["text"]).count("<tool_call>") >= 2)
        none_clean = all("<tool_call>" not in assistant_part(r["text"])
                         for r in rendered if r["kind"] == "none")
        print(f"\n  no-call examples free of tool calls: {none_clean}")
        print(f"\n  {len(rendered)} rendered")
        print(f"  containing <tool_call>      : {n_tc}")
        print(f"  containing 2+ <tool_call>   : {multi}")
        print(f"  response marker present     : {all(RESPONSE_MARKER in r['text'] for r in rendered)}")
        if multi == 0:
            raise SystemExit("no multi-call examples rendered — the template is not "
                             "emitting parallel <tool_call> blocks; fix before training")
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rendered]
    ).train_test_split(test_size=0.1, seed=20260821)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")

    # TRL 1.7.1 treats Qwen3.5 as a VLM (it is multimodal — hence the mmproj
    # file) and its chunked-CE patch reads `config.text_config`, which this
    # transformers version exposes only as `get_text_config()`. Without this
    # shim SFTTrainer construction dies with AttributeError before step 1.
    if not hasattr(model.config, "text_config"):
        try:
            model.config.text_config = model.config.get_text_config()
            log.info("shimmed config.text_config for TRL's VLM code path")
        except Exception:
            model.config.text_config = model.config

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    cfg = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        max_length=args.max_seq,
        # Loss on the assistant turn only. Without this the adapter spends
        # capacity learning to write the USER's half of the conversation,
        # which is not the behaviour under test.
        completion_only_loss=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
        peft_config=lora,
    )
    log.info("training on %d examples (%d held out)", len(ds["train"]), len(ds["test"]))
    trainer.train()
    trainer.save_model(str(args.out_dir))
    log.info("adapter saved to %s", args.out_dir)
    log.info("NEXT: convert to GGUF and re-run the eval — an unconverted adapter "
             "never touches llama.cpp inference, which is why the GSM8K run was never felt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
