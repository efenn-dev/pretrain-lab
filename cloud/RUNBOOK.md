# Migrate the GRPO run to a rented A100 (RunPod)

Goal: free the local 5070 Ti (for Minecraft) while the Qwen3.5-4B GSM8K GRPO run
finishes faster in the cloud, then pull the trained adapter back. ~5 h, ~$6–8.

The smart ordering: **set up the pod while training keeps running locally**, then
pause + upload the latest checkpoint. Zero idle time, minimal lost steps.

---

## 1. RunPod account (do this while training runs locally)
1. Sign up at runpod.io, add **$10** credit (Billing).
2. Install the CLI locally for painless transfers (no winget needed — download the exe):
   ```powershell
   Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile "$env:USERPROFILE\runpodctl.exe"
   ```
   Then call it by full path, e.g. `& "$env:USERPROFILE\runpodctl.exe" send migrate-grpo.tar.gz`.
   (Alternatives if that 404s: the RunPod web UI "Upload" button, or scp over the pod's SSH.)

## 2. Deploy the pod
- **Deploy → GPU Pod → Community Cloud**.
- GPU: **A100 (40 GB)** — plenty for a 4B/4-bit job; don't pay for 80 GB.
- Template: an official **PyTorch / CUDA 12.x** template.
- Container disk ~20 GB, add a ~20 GB volume (holds the HF weight cache).
- Deploy, then open the **web terminal** (or SSH).

## 3. Pause locally + pack the checkpoint
Back on your PC:
```powershell
cd C:\projects\pretrain-lab
.\pause-grpo.ps1              # frees the local GPU — Minecraft is a go after this
.\cloud\pack_for_cloud.ps1   # -> migrate-grpo.tar.gz (~220 MB, latest checkpoint)
runpodctl send migrate-grpo.tar.gz    # prints a one-time code
```

## 4. On the pod
```bash
cd /workspace
runpodctl receive <code-from-step-3>
mkdir -p pretrain-lab && tar -xzf migrate-grpo.tar.gz -C pretrain-lab
cd pretrain-lab
tmux new -s grpo
bash cloud/cloud_resume.sh          # installs deps, resumes, then evals
# detach: Ctrl-b then d   |   reattach: tmux attach -t grpo
```
Watch it: `tail -f logs/grpo_qwen.log`. It resumes from the checkpoint's step and
runs to step 150, then prints the after-RL GSM8K score (baseline 86.0%).

## 5. Pull results back + SHUT IT DOWN
When the log shows the eval score:
```bash
# on the pod
cd /workspace/pretrain-lab
tar -czf done-grpo.tar.gz runs/qwen35-4b-gsm8k-grpo logs/grpo_qwen.log
runpodctl send done-grpo.tar.gz
```
```powershell
# on your PC
cd C:\projects\pretrain-lab
runpodctl receive <code>
tar -xzf done-grpo.tar.gz     # merges the final adapter + log back in
```
Then **Terminate the pod in the RunPod console** — billing stops only on terminate,
not on "stop". This is the step people forget and wake up to a drained balance.

---

### Gotchas
- Match versions — that's what `requirements-cloud.txt` is for. A mismatched
  transformers/trl/peft is the usual cause of a failed resume.
- If `torch==2.11.0+cu128` won't install, the pod's driver is too old — pick a
  newer CUDA 12.x template.
- Keep an eye on spend: A100 Community ~$1.2–1.6/hr. $10 = ~6–8 h of runway.
