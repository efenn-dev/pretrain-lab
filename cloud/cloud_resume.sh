#!/usr/bin/env bash
# Resume the GRPO run on a rented GPU (A100), then eval. Run this INSIDE tmux:
#   tmux new -s grpo ; bash cloud/cloud_resume.sh
# Detach with Ctrl-b d ; reattach with `tmux attach -t grpo`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."                       # pretrain-lab root

# Cache HF weights/datasets on the pod volume so a restart doesn't re-download.
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TOKENIZERS_PARALLELISM=false

echo "== installing torch (cu128) =="
pip install --quiet torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
echo "== installing pinned deps =="
pip install --quiet -r "$SCRIPT_DIR/requirements-cloud.txt"
echo "== installing hybrid-attention kernels (REQUIRED for Qwen3.5-4B GPU speed) =="
bash "$SCRIPT_DIR/install_kernels.sh"

echo "== GPU check =="
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

mkdir -p logs
echo "== resuming training (auto-detects latest checkpoint under out-dir) =="
python grpo_qwen.py --out-dir runs/qwen35-4b-gsm8k-grpo --max-steps 150 2>&1 | tee -a logs/grpo_qwen.log

echo "== after-RL eval =="
python eval_gsm8k.py --model Qwen/Qwen3.5-4B --adapter runs/qwen35-4b-gsm8k-grpo --tag after-rl 2>&1 | tee -a logs/grpo_qwen.log

echo "== DONE. Pull back: runs/qwen35-4b-gsm8k-grpo/ (final adapter) and logs/grpo_qwen.log =="
