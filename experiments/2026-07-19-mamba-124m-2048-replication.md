# Mamba 124M / 2048 Replication v1

Status: complete; frozen gate passed (`replicated`)
Date: 2026-07-19

## Experiment identity

- Experiment ID: `mamba-124m-2048-replication-v1`
- Dataset: the same FineWeb GPT-2 uint16 corpus used by the prior 124M result
- Seeds: `1337`, `2027`, `7919`
- Transformer: `preset=gpt2`, `arch=modern`, default 12 layers
- Mamba: `preset=gpt2`, `arch=mamba`, `n_layer=22`
- Context: `2048`
- Batch: `6`
- Iterations: `3000`
- Evaluation interval: `500`
- Downstream: HellaSwag, 1000 validation examples per checkpoint
- Hard spend cap: USD 12
- Stop threshold: USD 10 projected or actual spend

## Why this experiment

The prior matched runs found that the transformer won at 30M parameters and 512
context, while Mamba won at 2048 context at both roughly 30M and 124M parameters.
Those results used one seed and validation loss only. This replication tests the
largest remaining credibility gaps without opening a new architecture lane:
seed sensitivity and downstream utility.

The kernel route-history application remains a separate, data-gated question.
It is not trained as part of this experiment.

## Kernel readiness snapshot

Read-only `build_sequences.py --stats` on 2026-07-19 reported:

- 43 callers with at least 3 events
- 3,488 events in qualifying sequences
- 2,851 `local_reason` events and 637 events on other routes
- 25 synthetic callers with 380 tagged synthetic events
- 18 non-synthetic callers among the qualifying caller set

Kernel readiness verdict: `false`. Non-`local_reason` events are now in the
hundreds, but qualifying callers remain far below the existing "into the
hundreds" gate and organic qualifying callers are only 18. Re-entry trigger:
at least 100 qualifying callers with a non-trivial route mix, reported with
organic and synthetic counts separated.

## Frozen success gate

Verdict is `replicated` only when all three requirements pass:

1. Mamba final validation loss beats the matched transformer in at least two of
   three seeds.
2. Median paired `mamba_val_loss - transformer_val_loss` is below `0.000`.
3. Median HellaSwag `acc_norm` is no worse than transformer by more than `0.010`,
   and Mamba wins at least one seed.

If some but not all requirements pass, verdict is `partial`. If none of the
directional requirements pass on complete valid pairs, verdict is `null`.

## Frozen stop conditions

Stop and record `stopped` for any affected pair when any condition occurs:

- NaN loss
- Parameter mismatch above 5%
- Unequal token budget between a seed pair
- Missing partner for a seed
- Mamba kernel failure after one clean reinstall
- Projected or actual spend reaches USD 10

A stopped or invalid pair is not counted as a win. Metrics and thresholds are
not changed after results are visible.

## Exact training configuration

For each seed, run the transformer and Mamba with the same data, context, batch,
iteration count, evaluation interval, and seed. Each model receives 36,864,000
training tokens before any internal accumulation adjustment:

`6 batch * 2048 tokens * 3000 iterations`.

## Environment and data evidence

- Local syntax validation: passed for `train.py` and `eval_hellaswag.py` using
  Python 3.12 `py_compile`
- Local PyTorch execution: unavailable in the bundled Windows Python; no new
  local training environment will be installed for this experiment
- Upload archive: `C:\projects\review-artifacts\mamba-124m-2048-replication-v1.tgz`
- Upload archive SHA-256: `F8091A0EB8F41F0AAD3A19EE17D833CC90398E59BADE03350A4B68CB99448A79`
- Upload manifest: `train.py`, `eval_hellaswag.py`, `prepare_fineweb.py`,
  `cloud/install_mamba_wheels.sh`, `cloud/RUNBOOK-ssm.md`, and this brief only
- Remote environment: RunPod RTX 4090, Python `3.11.10`, PyTorch `2.4.1+cu124`, NVIDIA driver `550.127.05`; fused Mamba kernel smoke passed with `mamba-ssm 2.2.2` and `causal-conv1d 1.4.0`; downstream evaluation used `tiktoken 0.13.0`
- FineWeb train SHA-256: `4a042405b212f04882d137e91f851270127e8d2bcb1f44af027262c4d279e432` (local/remote match)
- FineWeb validation SHA-256: `b0244a836383dafccbdd2760653c6c5ce54ba10592492abdc8e73da3a11d1b8f` (local/remote match)
- Parameter smoke: transformer `123.59M`; Mamba `121.61M`; mismatch approximately `1.6%`, within the frozen `5%` limit
- Training launched at `2026-07-20T03:14:43+00:00`; orchestrator PID `2773`
- Evidence archive: `C:\projects\review-artifacts\dual-track\2026-07-19\agi-mamba-replication\mamba-124m-2048-replication-v1-evidence.tgz`
- Evidence archive SHA-256: `FC80B8C56D665A86EA7E5462757F8E7E8575EDEA360520B94ED5915F3B653D99`
- Evidence contents: source, run configuration, complete logs, six metadata files, six HellaSwag JSON results, environment freeze, data hashes, and checkpoint hashes. Checkpoint binaries were intentionally not exported.
- Replacement pod runtime: about `1h 4m` at `$0.70/hr`; observed balance change was approximately `$0.74`, below the frozen spend cap. After the verified evidence export, both the replacement pod and the earlier incompatible-image pod were terminated with their attached volumes on 2026-07-19. The RunPod pod list was verified empty afterward.

## Results

All three matched pairs completed with the frozen configuration. Lower validation
loss and higher HellaSwag `acc_norm` are better.

| seed | transformer val | mamba val | val delta (M-T) | transformer HellaSwag | mamba HellaSwag | HellaSwag delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 4.5042 | **4.4165** | **-0.0877** | 0.272 | **0.286** | **+0.014** |
| 2027 | 4.4936 | **4.4242** | **-0.0694** | 0.282 | **0.284** | **+0.002** |
| 7919 | 4.5294 | **4.4456** | **-0.0838** | **0.286** | 0.277 | -0.009 |
| median | 4.5042 | **4.4242** | **-0.0838** | 0.282 | **0.284** | **+0.002** |

## Frozen-gate verdict

`replicated` — all three predeclared requirements passed:

1. Mamba won validation loss in `3/3` seeds (required at least `2/3`).
2. Median paired validation-loss delta was `-0.0838` (required below `0.000`).
3. Median HellaSwag delta was `+0.002`, and Mamba won `2/3` seeds (required no
   worse than `-0.010` and at least one win).

The validation-loss result is consistent and materially stronger than the prior
single-seed result. The HellaSwag result only clears the non-regression gate: all
scores are close to chance, the evaluation uses 1,000 examples per checkpoint,
and the paired advantage changes sign in one seed. Treat downstream utility as
promising but unresolved, not as a strong capability claim.

The first evaluation attempt completed model loading but failed because
`tiktoken` was absent. The environment was repaired by installing only
`tiktoken==0.13.0`; the exact frozen checkpoints and evaluation command were then
rerun successfully with shell fail-fast enabled. Both logs are retained.
