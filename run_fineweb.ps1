# Detached overnight run: FineWeb-Edu prep (2B tokens) then 124M GPT-2 pretraining.
# Launched via Start-Process so it survives the Claude session ending.
# Progress:  Get-Content logs\fineweb_prep.log -Tail 20
#            Get-Content logs\fw124m_train.log -Tail 20
# Stop:      Get-Content logs\run.pid | Stop-Process -Id { $_ }
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$PID | Out-File logs\run.pid

if (-not (Test-Path data\fineweb\train.bin)) {
    python prepare_fineweb.py --out-dir data/fineweb --max-tokens 2000000000 *>> logs\fineweb_prep.log
    if ($LASTEXITCODE -ne 0) { "PREP FAILED exit $LASTEXITCODE" *>> logs\fineweb_prep.log; exit 1 }
}

# micro-batch 8 (not 16): 16 hit the 16GB VRAM ceiling and WDDM paged to system
# RAM, stalling the run at ~4x slowdown. 8x1024x64 = same 0.5M tokens/step.
python train.py --data data/fineweb --out-dir runs/fw-124m --preset gpt2 `
    --block-size 1024 --batch-size 8 --grad-accum 64 --lr 6e-4 `
    --max-iters 4800 --warmup-iters 200 --eval-interval 200 --eval-iters 10 `
    *>> logs\fw124m_train.log
