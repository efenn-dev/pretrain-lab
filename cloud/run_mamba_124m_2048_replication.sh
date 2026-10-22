#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mamba-124m-2048-replication-v1

experiment_id="mamba-124m-2048-replication-v1"
run_root="runs/${experiment_id}"
mkdir -p "${run_root}"

exec > >(tee -a "${run_root}/orchestration.log") 2>&1

echo "experiment=${experiment_id}"
echo "started_at=$(date --iso-8601=seconds)"
python --version
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
sha256sum data/fineweb/train.bin data/fineweb/val.bin

for seed in 1337 2027 7919; do
  seed_root="${run_root}/seed-${seed}"
  mkdir -p "${seed_root}"

  echo "pair_start seed=${seed} at=$(date --iso-8601=seconds)"

  python train.py \
    --data data/fineweb \
    --out-dir "${seed_root}/transformer" \
    --preset gpt2 \
    --arch modern \
    --block-size 2048 \
    --batch-size 6 \
    --max-iters 3000 \
    --eval-interval 500 \
    --eval-iters 20 \
    --seed "${seed}" \
    2>&1 | tee "${seed_root}/transformer.log"

  python train.py \
    --data data/fineweb \
    --out-dir "${seed_root}/mamba" \
    --preset gpt2 \
    --arch mamba \
    --n-layer 22 \
    --block-size 2048 \
    --batch-size 6 \
    --max-iters 3000 \
    --eval-interval 500 \
    --eval-iters 20 \
    --seed "${seed}" \
    2>&1 | tee "${seed_root}/mamba.log"

  echo "pair_complete seed=${seed} at=$(date --iso-8601=seconds)"
done

echo "training_complete_at=$(date --iso-8601=seconds)"
