# Detached GRPO run on Qwen3.5-4B / GSM8K (survives closed sessions).
# Progress:  Get-Content logs\grpo_qwen.log -Tail 20
# After it finishes:
#   python eval_gsm8k.py --model Qwen/Qwen3.5-4B --adapter runs/qwen35-4b-gsm8k-grpo --tag after-rl
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
# 100 steps: lands same-day at healthy pace; ckpt every 25 steps caps interruption loss
python grpo_qwen.py --out-dir runs/qwen35-4b-gsm8k-grpo --max-steps 100 *>> logs\grpo_qwen.log
if ($LASTEXITCODE -eq 0) {
    python eval_gsm8k.py --model Qwen/Qwen3.5-4B --adapter runs/qwen35-4b-gsm8k-grpo --tag after-rl *>> logs\grpo_qwen.log
}
