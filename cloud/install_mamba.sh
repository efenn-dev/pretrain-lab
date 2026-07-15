#!/usr/bin/env bash
# Install the Mamba SSM kernels for the Phase-1G SSM-vs-transformer experiment
# (train.py --arch mamba). Triton/CUDA — Linux GPU box only (RunPod), NOT
# Windows. Idempotent. Mirrors install_kernels.sh's proven causal-conv1d build.
set -euo pipefail

echo "== build prerequisites =="
pip install --quiet ninja packaging wheel setuptools

echo "== causal-conv1d (source build against the installed torch) =="
# --no-build-isolation is ESSENTIAL — a plain install builds against a different
# torch and fails at import. Same lesson as the GRPO kernel kit.
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" \
    pip install --quiet causal-conv1d --no-build-isolation

echo "== mamba-ssm (the selective-scan kernel; ~3-8 min compile) =="
MAMBA_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" \
    pip install --quiet mamba-ssm --no-build-isolation

python - <<'PY'
import torch, mamba_ssm, causal_conv1d
from mamba_ssm import Mamba
m = Mamba(d_model=384, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 64, 384, device="cuda")
assert m(x).shape == x.shape
print("mamba kernels OK | torch", torch.__version__, "| a Mamba layer runs on GPU")
PY
