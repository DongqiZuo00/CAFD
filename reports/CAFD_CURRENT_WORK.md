# Current authorized work: residual-anchored CAFD v14

Updated 2026-09-05. Remote work root: /blue/du.j/jinjiaguo/CAFD.
User authorized implementing and launching the correction. No additional approval
is needed for this pilot. Qwen is banned; use the existing Ministral 3B/8B pair.

## State and evidence

Pilot job submitted: 41175009; training code commit 51e8f3c.
Initial Slurm state: PENDING (Priority), no node assigned.
Read logs/cafd/anchored-v14-41175009.out and .err.
Monitor: cafd-anchored-v14-monitor (10-minute checks).

Audit job 41173510 completed on c1006a-s5, exit 0, elapsed 5m31s.
Audit commit 17724bd; artifacts/cafd/experiments/mistral_cafd_target_audit_v13/.
Audit sampled only two Teacher rollouts per phase and 64 positions per rollout.
It suggests route imbalance and residual mismatch; it does not establish causal
attribution, explain Student-generated prefixes, or prove the fix will help.
Uniform full-vocabulary RMS also includes very low-probability tokens.

Same selection split: CAFD [0,2,20,32,37,38,38], Progressive [0,1,14,26,38,49,55],
Endpoint [0,1,19,46,39,43,49], steps [0,10,40,80,120,160,200].
Do not mix CAFD confirmation 33/64 with other methods' selection scores.

## Frozen pilot definition

Run: mistral_cafd_anchored_v14. Derive the resolved configuration from the
validated v10 mixed configuration. Preserve model revisions, seed 2027,
fit/selection data, renderer, generation, prompt stream prefix, optimizer,
learning rate, Teacher route, phase length 40, and support schedule.

For each completion prediction position:
d = RMS(center(z_next - z_previous))
r = RMS(center(z_phase_ref - z_previous))
alpha = d/(d+r), with alpha=0 when d+r=0.
q = (1-alpha)*softmax_FP32(z_next)
    + alpha*softmax_FP32(z_phase_ref+z_next-z_previous).
Use detached targets/weights and exact full-vocabulary forward KL.
phase_ref is the independent frozen phase-start Student and is never refreshed
within a phase. Log alpha and RMS summaries for every optimizer update.
Use logaddexp for the probability mixture. No top-k KL or ratio clipping.

Pilot budget: 80 updates, 32 slots/update, milestones 0/10/40/80.
1 B200 and 96G RAM, Slurm time limit 6h. CAFD maximum 4 B200 / 192G.
Preserve all old checkpoints and resumes; write into the new run only.
Selection split only; no confirmation or frozen-test evaluation.

Pilot pass: step40 >=20, step80 >=33, step80 > step40.
step80 >=40 is aspirational, not a post-hoc threshold.
A passing pilot is eligible for the previously discussed extension to 200;
this launch itself stops at 80 and records pilot_gate.json.

## Entry points and outputs

cafd/mistral_anchored_v14.py; scripts/cafd/slurm/mistral_anchored_v14.sbatch.
Runs output: runs/cafd/experiments/mistral_cafd_anchored_v14/.
Metrics: cafd/alpha_metrics.jsonl under the run.
Scores and pilot gate: artifacts/cafd/experiments/mistral_cafd_anchored_v14/.
Tests cover mixture correctness, detached gradients, offset invariance, numerical
edge cases, full-vocabulary block/dense loss and gradients, and prompt stream.
