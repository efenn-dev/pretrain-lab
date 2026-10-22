# Resume GRPO training from the last checkpoint, detached (survives closing this window).
# grpo_qwen.py auto-detects the latest checkpoint in the out-dir; run_grpo_qwen.ps1 also
# fires the after-RL eval once training finishes.
# Watch progress:  Get-Content logs\grpo_qwen.log -Tail 20 -Wait
$already = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*grpo_qwen.py*' }
if ($already) {
    Write-Host "GRPO is already running (PID $($already.ProcessId)). Nothing to do."
    return
}
Start-Process -WindowStyle Hidden powershell.exe `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','run_grpo_qwen.ps1' `
    -WorkingDirectory $PSScriptRoot
Write-Host "GRPO resuming from the last checkpoint (detached)."
Write-Host "Tail the log:  Get-Content logs\grpo_qwen.log -Tail 20 -Wait"
