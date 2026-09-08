# CAFD Mistral Support-Corrected Final Report

## Experiment identity

- Protocol: `mistral_cafd_support_full_v9`
- Seed: 2027
- Student: `mistralai/Ministral-3-3B-Instruct-2512-BF16`
- Teacher: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Benchmark: DELTA Manufactoria-HAS
- Training budget: 200 updates, 4 prompts x 8 rollouts per update
- Slurm job: 41049657 on `c0910a-s5`
- Code commit: `13211088a56bbad63fd4cb781a0d3819ac1c140a`

This run is CAFD-v1 with support-corrected, phase-stationary next-Teacher replay.
The exact FP32 full-vocabulary target and completion-position forward KL remained:

`q_m = stopgrad[softmax(z_phase_ref + z_T(m+1) - z_T(m))]`

It must not be reported as the failed raw Student on-policy-prefix variant.

## Development selection

| Step | Correct | Accuracy |
|---:|---:|---:|
| 0 | 0/64 | 0.0000 |
| 10 | 4/64 | 0.0625 |
| 40 | 23/64 | 0.3594 |
| 80 | 36/64 | 0.5625 |
| 120 | 32/64 | 0.5000 |
| 160 | 35/64 | 0.5469 |
| 200 | 32/64 | 0.5000 |

The development-only selection froze `step80`.

## Frozen held-out result

- Correct: 63/132
- Official full-pass/pass@1: 0.4772727
- Wilson 95% interval: [0.3939181, 0.5619128]
- Frozen evaluation count: one

## Compute and token cost

- Training-generated tokens: 2,465,328
- Teacher-scored tokens: 4,930,656
- Held-out evaluation tokens: 67,955
- Training GPU-hours: 2.8658
- End-to-end Slurm elapsed: 03:13:17
- Resources: 1 x NVIDIA B200, 96G requested host memory
- MaxRSS: approximately 9.55 GiB

## Trajectory audit

There were exactly 6,400 training rollouts: 32 per update for all 200 updates.
Each phase contained 1,280 rollouts, with sources R1, R2, R3, R4, and R5
respectively. No ERROR, Traceback, OOM, NCCL, or fatal error was found.

## Retained checkpoints

The storage compaction retains the minimal reproducible checkpoint set plus two
small auxiliary route checkpoints needed to keep the frozen RL-only route valid:

1. Student S0.
2. Teacher raw Instruct TBase.
3. Teacher verified-SFT endpoint SFT125.
4. Teacher RL checkpoints T40, T60, T80, and T100.
5. Development-selected CAFD Student step80.
6. Auxiliary Teacher RL checkpoints T0 and T20.

All configurations, code, gates, curves, final score/cost files, state metadata,
and logs are retained. Optimizer resumes, phase-reference copies, raw rollouts,
unselected checkpoints, the v8 probe weights, and obsolete Qwen runs are removed.

## Storage compaction audit

- Compaction date: 2026-09-04 (America/New_York)
- Explicitly authorized permanent deletion: 4,030 GiB
- CAFD checkpoint tree after compaction: 140 GiB
- Filesystem before: 7.0 TiB total, 6.2 TiB used, 834 GiB available (89%)
- Filesystem after: 7.0 TiB total, 2.3 TiB used, 4.8 TiB available (33%)
- Retained checkpoint verification: 10/10 present
- Broken retained symlinks: 0
- Primary Teacher route status: frozen and complete
- Auxiliary RL-only route status: frozen and complete
- Selected CAFD checkpoint: step80, present and readable

The cleanup is irreversible from Git. Recovery of deleted weights requires an
external filesystem snapshot or rerunning the corresponding experiment.
