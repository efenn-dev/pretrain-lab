# pretrain-lab

Train GPT-family language models **from scratch** on the local RTX 5070 Ti (16 GB).
Same architecture family as GPT-2/3 — the only difference between this and a
frontier lab is scale (parameters, tokens, GPUs).

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
