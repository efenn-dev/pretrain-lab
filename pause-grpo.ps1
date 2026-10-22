# Pause GRPO training and free the GPU (e.g. to play a game).
# Progress is safe: it resumes from the last checkpoint (every 25 steps).
# You lose only the steps since the last checkpoint. Restart with: .\resume-grpo.ps1
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*grpo_qwen.py*' }
if (-not $procs) {
    Write-Host "No GRPO training process is running."
    return
}
foreach ($p in $procs) {
    Write-Host "Stopping GRPO (PID $($p.ProcessId))..."
    Stop-Process -Id $p.ProcessId -Force
}
Start-Sleep -Seconds 3
$free = (nvidia-smi --query-gpu=memory.free --format=csv,noheader) -join ', '
Write-Host "GRPO paused. GPU free memory: $free"
Write-Host "When you're done gaming, run:  .\resume-grpo.ps1"
