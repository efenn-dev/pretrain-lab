# pretrain-lab

Train GPT-family language models **from scratch** on a single consumer GPU
(built and measured on an RTX 5070 Ti, 16 GB). Same architecture family as
GPT-2/3 — the only difference between this and a frontier lab is scale
(parameters, tokens, GPUs). The full three-stage frontier recipe — pretraining,
instruction tuning, and RL on verifiable rewards — runs end to end here, in
plain readable code (the from-scratch lane has zero dependencies beyond
PyTorch + numpy + tiktoken).

## Measured results (one day, one GPU)

**124M pretrained from scratch** on 2.5B FineWeb-Edu tokens (~10.5 h):
val loss 3.219, HellaSwag acc_norm **32.5%** — above the original GPT-2 124M
(~29–31%; random = 25%). Same architecture and size; the difference is a
decade of better training data.

**Five-run RLVR study** (124M, synthetic exactly-verifiable tasks,
strict exact-match eval throughout):

| Category | SFT | RL (binary) | Seeded SFT | Seed+RL (binary) | Seed+RL (shaped) |
|---|---|---|---|---|---|
| Add | 0% | 0% | 0% | 0% | **7.8%** |
| Subtract | 0% | 0% | 7.9% | 0% | **15.8%** |
| Count letter | 0% | 0% | 52.0% | **72.0%** | 56.0% |
| Compare y/n | 10.5% | 50.0% | 100% | 100% | 84.2% |
| **Overall** | **1.6%** | **11.7%** | **50.0%** | **49.2%** | **50.8%** |

Findings, each earned the hard way:
1. RL alone teaches *form*, not *function* — the unseeded RL column is pure format compliance.
2. One 7-minute SFT pass installed more capability than any amount of RL could (+48 points).
3. Binary rewards only sharpen what circuits already support (counting jumped; exact arithmetic destabilized).
4. Reward density gates what RL can reach: partial credit for near misses was the *only* method that moved addition off 0%.
5. Architecture is the final ceiling — no reward schedule makes a 124M compute exact multi-digit sums.

**Industrial lane:** the same GRPO recipe on Qwen3.5-4B (TRL 1.7 + QLoRA,
`grpo_qwen.py`) against GSM8K — zero-shot baseline 86.0%, RL run in progress.

## Quick start (verified working)

```bash
# 1. smoke test — 10M-param char-level model on Shakespeare, ~40 s
python train.py --data data/shakespeare.txt --out-dir runs/shakespeare-nano --preset nano --max-iters 1000

# 2. hear it talk
python sample.py --run-dir runs/shakespeare-nano --prompt "JULIET:" --num-tokens 350
```

## What this GPU can realistically pretrain

Measured throughput on the smoke test: ~530K tok/s at 10M params. Rough planning
table (chinchilla-optimal = ~20 tokens per parameter):

| Preset        | Params | Optimal tokens | Wall-clock (est.) | What you get |
|---------------|--------|----------------|-------------------|--------------|
| `nano`        | 10M    | 0.2B           | minutes           | toy / char-level fun |
| `gpt2`        | 124M   | 2.5B           | ~1 day            | coherent English, GPT-2-small quality |
| `gpt2` long   | 124M   | 10B            | ~3–4 days         | best-possible 124M (llm.c recipe) |
| `gpt2-medium` | 350M   | 7B             | ~1–2 weeks        | practical ceiling for this card |

Beyond ~350M from scratch, single-GPU wall-clock stops making sense — that's
where you rent an 8×H100 node for a weekend (~$500) or switch to fine-tuning an
existing open model (Qwen3 4B QLoRA fits in 16 GB easily).

## Scaling up to real data

```bash
pip install datasets tiktoken
# stream + tokenize FineWeb-Edu (2B-token slice ≈ 4 GB on disk)
python prepare_fineweb.py --out-dir data/fineweb --max-tokens 2000000000
# train GPT-2-small shape on it
python train.py --data data/fineweb --out-dir runs/fw-124m --preset gpt2 \
    --block-size 1024 --batch-size 16 --grad-accum 32 --lr 6e-4 \
    --max-iters 4800 --warmup-iters 200 --eval-interval 200
```

(`batch 16 × block 1024 × accum 32` ≈ 0.5M tokens per step, GPT-2 paper scale.
4800 steps ≈ 2.5B tokens.)

## Dashboard

```bash
python dashboard.py --port 7871    # or launch "Pretrain Lab" from the Hermes dashboard
```

http://127.0.0.1:7871 — live loss curve, iter/ETA, GPU stats, the roadmap, and a
shared notes box (persisted to `dashboard_notes.json`; Claude sessions read it,
so use it to leave instructions between sessions). Registered in app-manager
as `pretrain-lab`, default command `dashboard`.

## The full recipe

```bash
# stage 1 — pretrain (see above)
# stage 2 — instruction-tune the pretrained ckpt into a tiny chat model (~1h GPU)
python sft.py --ckpt-dir runs/fw-124m --out-dir runs/fw-124m-sft
# benchmark — HellaSwag accuracy (random = 25%, GPT-2 124M ≈ 29-31%)
python eval_hellaswag.py --ckpt-dir runs/fw-124m --max-examples 1000
```

## Architectures

`--arch gpt2` (default) is the faithful 2019 GPT-2: learned positions, LayerNorm,
GELU. `--arch modern` is the Llama-style stack: rotary positions (RoPE), RMSNorm,
SwiGLU, zero-init residual projections — same cost per step, reaches better loss;
use it for new runs. Checkpoints remember their arch, so `sample.py`, `sft.py`,
and evals work with both.

## Files

- `train.py` — model (both archs) + training loop (char-level `.txt` or GPT-2-token `.bin` dirs)
- `sample.py` — generation from a checkpoint
- `sft.py` — stage-2 supervised fine-tuning (SmolTalk chat data)
- `eval_hellaswag.py` — benchmark; writes `hellaswag.json` next to the ckpt
- `dashboard.py` + `index.html` — local tracking dashboard (stdlib server)
- `prepare_fineweb.py` — FineWeb-Edu → `train.bin`/`val.bin`
- `run_fineweb.ps1` — detached prep + 124M training (survives closed sessions)
- `runs/<name>/` — checkpoints (`ckpt.pt` keeps best val loss) + `meta.json` vocab

## Notes

- bf16 autocast + fused AdamW + flash attention (SDPA) — all verified on the
  Blackwell card. The Ollama/Blackwell corruption issue is inference-quant-side
  and does not affect PyTorch training.
- `--compile` needs Triton, which is unreliable on Windows — leave it off.
- Presets are the standard shapes: nano (6L/384d), gpt2 (12L/768d),
  gpt2-medium (24L/1024d).
