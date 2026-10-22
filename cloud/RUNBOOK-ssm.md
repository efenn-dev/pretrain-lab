# RunBook: Phase-1G SSM-vs-transformer experiment on RunPod

The Mamba architecture experiment from `llm-router/docs/PLAN-phase1g-resume.md`,
track A. Runs on RunPod because the Mamba kernels (mamba-ssm + causal-conv1d,
Triton) are **Linux+CUDA only** — the Windows 5070 Ti cannot run them. Cheap:
nano-scale (~30M params) on a small GPU. **Est. total ~$1–3** (well under the
$12 campaign cap).

**The `train.py` code is ready** (`--arch mamba` + `--n-layer`/`--n-embd`
overrides, verified locally). This runbook is the on-pod procedure.

## What it proves

Is the SSM (Mamba) architecture competitive with a same-budget transformer on
OUR data? Matched-param pair:
- **transformer:** `--arch modern --n-layer 6` → **29.9M params** (baseline)
- **mamba:** `--arch mamba --n-layer 12` → **~30M params** (verify the printout
  says ~30M; nudge `--n-layer` to land within ~5%)
Metric: **val loss (bits/token)** at matched params + matched compute (same
`--max-iters`, `--block-size`, tokens seen). Downstream sanity: `eval_hellaswag.py`.
**Stop / verdict:** if Mamba doesn't match-or-beat the transformer at equal
budget, PARK the architecture (write why). A null is a valid outcome.

## Cost & GPU

- GPU: **RTX 4090 or A5000** (community cloud) — nano needs no A100. ~$0.30–0.45/hr.
- Time: setup ~15–20 min (mamba-ssm compiles), shakespeare smoke ~10 min,
  fineweb comparison (2 models) ~1 hr. **~1.5–2 hr → ~$0.60–1.50.**
- **Terminate the pod the moment results are pulled** (RunPod bills by the second).

## Procedure

1. **Provision** (Emmitt / the payer — this is the billed step): RunPod → GPU
   Pod → Community Cloud → RTX 4090 → an official PyTorch/CUDA 12.x template →
   ~20 GB container disk. (Same flow as `RUNBOOK.md` §2.)
2. **Sync the repo** to the pod: `pack_for_cloud.ps1` → `runpodctl send`, or the
   web upload / `git clone` if it's pushed. Needs `train.py`, `eval_hellaswag.py`,
   `cloud/`, and `data/`.
3. **Data:** upload `data/shakespeare.txt` (tiny) for the smoke. For the real
   run, either upload `data/fineweb/` (3.8 GB — slow) OR re-tokenize on the pod
   with `python prepare_fineweb.py` (downloads the FineWeb sample; faster than a
   3.8 GB upload on most links).
4. **Install kernels:** `bash cloud/install_mamba.sh` (prints "mamba kernels OK").
5. **Smoke (proves both train + the kernels, ~10 min):**
   ```bash
   python train.py --data data/shakespeare.txt --out-dir runs/ssm/xf-smoke    --arch modern --n-layer 6  --max-iters 500 --eval-interval 100
   python train.py --data data/shakespeare.txt --out-dir runs/ssm/mamba-smoke --arch mamba  --n-layer 12 --max-iters 500 --eval-interval 100
   ```
   Confirm both print ~30M params and val loss falls.
6. **Real comparison (fineweb, matched compute):**
   ```bash
   python train.py --data data/fineweb --out-dir runs/ssm/xf-fw    --arch modern --n-layer 6  --max-iters 3000 --eval-interval 250 --batch-size 32 --block-size 512
   python train.py --data data/fineweb --out-dir runs/ssm/mamba-fw --arch mamba  --n-layer 12 --max-iters 3000 --eval-interval 250 --batch-size 32 --block-size 512
   ```
   Same iters/batch/block for both = matched compute. Record final val loss.
7. **Downstream sanity:** `python eval_hellaswag.py` on each checkpoint.
8. **Pull results** (`runs/ssm/*/` logs + the val-loss curve) back:
   `runpodctl receive`, then **TERMINATE THE POD**.
9. **Record the verdict** in `llm-router/docs/PLAN-phase1g-resume.md` (val loss
   Mamba vs transformer at matched budget → adopt-and-scale, or park).

## Guardrails

- Hard budget: stop and re-plan if spend approaches **$5** (cap is $12 for the
  whole campaign; this should cost ~$1–3).
- Terminate compute the second results are exported — no idle billing, no
  always-on endpoint.
- This is an advisory research prototype. Nothing wires into live routing
  without the shadow→eval-gate→promote discipline the router brain used.
