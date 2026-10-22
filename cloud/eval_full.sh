#!/usr/bin/env bash
# Full GSM8K (1319-problem test split) eval: base Qwen3.5-4B vs the GRPO adapter,
# both 4-bit, greedy, identical harness — the apples-to-apples confirmation of the
# +5pt gain seen on n=200. Runs on a Linux GPU pod (needs the hybrid-attention kernels).
#   tmux new -s eval ; bash cloud/eval_full.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TOKENIZERS_PARALLELISM=false
BATCH="${BATCH:-32}"     # 48-64 fine on A100 80GB; 32 safe on 40GB

echo "== installing runtime deps (bitsandbytes/datasets/accelerate/transformers — pod base lacks bitsandbytes) =="
pip install -q -r "$SCRIPT_DIR/requirements-cloud.txt"
echo "== installing hybrid-attention kernels (Qwen3.5 needs these; slow CPU fallback without) =="
bash "$SCRIPT_DIR/install_kernels.sh"

mkdir -p logs
echo "== BASE model, full 1319 =="
python eval_gsm8k.py --model Qwen/Qwen3.5-4B --tag base-full \
    --max-examples 1319 --batch-size "$BATCH" 2>&1 | tee -a logs/eval_full.log

echo "== GRPO ADAPTER, full 1319 =="
python eval_gsm8k.py --model Qwen/Qwen3.5-4B --adapter runs/qwen35-4b-gsm8k-grpo --tag afterrl-full \
    --max-examples 1319 --batch-size "$BATCH" 2>&1 | tee -a logs/eval_full.log

echo "== RESULTS =="
echo -n "BASE:    "; cat runs/gsm8k-evals/base-full.json; echo
echo -n "ADAPTER: "; cat runs/gsm8k-evals/afterrl-full.json; echo
echo "== DONE =="
