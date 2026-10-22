#!/usr/bin/env bash
# Install the hybrid-attention kernels Qwen3.5-4B needs for GPU-speed training/eval.
# WITHOUT these, transformers silently falls back to a CPU path ("The fast path is not
# available…") → ~900s/step regardless of GPU. Triton is required and does NOT work on
# Windows, so this only works on a Linux GPU box. Idempotent — safe to re-run.
set -euo pipefail

echo "== flash-linear-attention (Triton, no compile) =="
pip install --quiet flash-linear-attention

echo "== causal-conv1d (compiles against installed torch) =="
pip install --quiet ninja packaging wheel setuptools
# --no-build-isolation is ESSENTIAL: a plain `pip install causal-conv1d` builds in an
# isolated env against a different torch and fails. Force a source build against ours.
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" \
    pip install --quiet causal-conv1d --no-build-isolation

echo "== re-pin checkpoint-compat versions (fla may tug transformers) =="
pip install --quiet transformers==5.5.0 trl==1.7.1 peft==0.19.1

python -c "import fla, causal_conv1d, transformers; print('kernels OK | transformers', transformers.__version__)"
