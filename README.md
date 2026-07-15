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

**Architecture ablation — does a state-space model (Mamba) beat a Transformer,
and when?** Same from-scratch harness, `--arch mamba` vs `--arch modern`, matched
parameters and matched compute (identical tokens/iter and iters), changing one
variable at a time. FineWeb-Edu GPT-2 tokens, 3000 iters. Run on a rented RTX 4090
(RunPod) because the Mamba selective-scan kernels are Linux+CUDA only — see
`cloud/`. A Mamba block is ~half a Transformer block's parameters, so the match
bumps Mamba's layer count (6→12 layers at 30M, 12→22 at 124M).

| params | context | asymptote winner | Mamba final Δ vs Transformer |
|--------|--------:|------------------|-----------------------------:|
| 30M    | 512  | Transformer     | +0.050 *(Mamba loses)* |
| 30M    | 2048 | **Mamba**       | −0.036 |
| 124M   | 2048 | **Mamba**       | −0.048 *(with ~2% fewer params)* |

(val loss; `batch × block` held at 12,288 tokens/iter across the 512↔2048 change,
so only context length differs.)

Findings:
1. **Context length is the switch.** At 512 tokens attention wins the asymptote;
   quadruple the context to 2048 and Mamba wins every checkpoint — two matched
   runs, one variable, opposite winners.
2. **Not a toy-scale artifact.** Scaling params 4× (30M → 124M) at 2048 context
   preserves the Mamba win and slightly *grows* the margin (−0.036 → −0.048),
   even with Mamba carrying a ~2% parameter disadvantage.
3. **Mamba's edge is largest early and narrows toward the asymptote** (a
   sample-efficiency effect — at iter 500, 124M Mamba leads by 0.449). At long
   context the asymptote gap survives rather than closing; at short context it
   closes and flips.

The textbook state-space story — SSMs earn their keep as sequences get longer —
reproduced from scratch at a scale that runs in minutes. Honest limits: val loss
only (no downstream eval), fixed-compute not asymptotic (undertrained by
Chinchilla lights), single seed, one dataset. The head-to-head is fair (both
models see identical conditions); a stronger claim would add a seed sweep, a
downstream metric, and one more scale point (350M / 4096 context).

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
use it for new runs. `--arch mamba` swaps attention for a Mamba state-space mixer
(no positional encoding — the recurrence is order-aware); it needs the
`mamba-ssm` / `causal-conv1d` CUDA kernels (Linux only — `cloud/install_mamba_wheels.sh`
is the robust prebuilt-wheel install; `cloud/install_mamba.sh` source-builds but
fails when the box's nvcc is newer than torch's CUDA), and falls back to a slow
pure-PyTorch scan
elsewhere so the file still imports on Windows. Match its params to a Transformer
with `--n-layer` (a Mamba block is ~half a Transformer block). Checkpoints
remember their arch, so `sample.py`, `sft.py`, and evals work with all three.

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
