# Bundles the latest checkpoint + code + cloud scripts into one uploadable tarball.
# Run this AFTER pausing training (.\pause-grpo.ps1) so the latest checkpoint is final.
Set-Location "$PSScriptRoot\.."                     # pretrain-lab root
$ckpt = Get-ChildItem runs\qwen35-4b-gsm8k-grpo -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue |
    Sort-Object { [int]($_.Name -replace '\D','') } | Select-Object -Last 1
if (-not $ckpt) { Write-Error "No checkpoint found under runs\qwen35-4b-gsm8k-grpo"; exit 1 }
$rel = "runs/qwen35-4b-gsm8k-grpo/$($ckpt.Name)"
$out = "migrate-grpo.tar.gz"
Write-Host "Packing latest checkpoint: $($ckpt.Name)"
tar -czf $out grpo_qwen.py eval_gsm8k.py cloud/requirements-cloud.txt cloud/cloud_resume.sh $rel
$size = "{0:N0} MB" -f ((Get-Item $out).Length / 1MB)
Write-Host "Created $out ($size)"
Write-Host "Send to the pod (install runpodctl first):  runpodctl send $out"
