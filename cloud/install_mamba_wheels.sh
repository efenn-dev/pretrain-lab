#!/usr/bin/env bash
# Install the Mamba SSM kernels from PREBUILT WHEELS (no compile). This is the
# robust path — source-building (install_mamba.sh) fails when the box's nvcc is
# newer than the torch CUDA build (e.g. RunPod's stock image: nvcc 12.8 vs a
# torch cu121 wheel -> "Error compiling objects for extension"). Prebuilt wheels
# sidestep nvcc entirely. Verified on RunPod RTX 4090, Ubuntu, Python 3.12.
set -euo pipefail

# 1. Pin torch to a version the mamba wheels are built against. torch 2.8 is too
#    new for mamba-ssm 2.2.2 (undefined c10::cuda symbol at import, unfixable by
#    rebuild) — 2.4.1 is the known-good floor.
pip install --quiet torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install --quiet "numpy<2" transformers==4.44.2 einops

# 2. Prebuilt wheels. Pick the asset matching your (python tag, cxx11 ABI):
#      python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"   # -> abiFALSE/TRUE
#      python --version                                                    # -> cpXY
#    torch 2.4.x Linux pip wheels are cxx11abiFALSE. The cu122 wheels run fine on
#    a cu121 torch runtime (CUDA 12.x minor versions are compatible). Browse tags:
#      https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0
#      https://github.com/state-spaces/mamba/releases/tag/v2.2.2
CC1D="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0%2Bcu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
MAMBA="https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2%2Bcu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
pip install --quiet "$CC1D"
pip install --quiet "$MAMBA"

# 3. Prove the fused selective-scan kernel loads and runs on the GPU.
python -c "import torch; from mamba_ssm import Mamba; m=Mamba(d_model=384,d_state=16,d_conv=4,expand=2).cuda(); x=torch.randn(1,8,384,device='cuda'); print('FUSED_OK', tuple(m(x).shape), torch.__version__)"
