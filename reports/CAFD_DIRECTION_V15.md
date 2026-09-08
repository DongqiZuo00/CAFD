# CAFD direction diagnostic v15
Authorized 2026-09-05: diagnose increment direction before any new training.
Root /blue/du.j/jinjiaguo/CAFD. Job 41214438.
Entry cafd/mistral_direction_v15.py. Slurm scripts/cafd/slurm/mistral_direction_v15.sbatch.
1 B200, 96G, 1 hour maximum. No optimizer, generation, selection, confirmation, or frozen-test reads.
Two phases, first 8 distinct fit prompts with persisted Student rollouts per phase.
Compare canonical verified gold and persisted Student prefixes separately.
Phase 1 compares original and anchored step40 references.
Up to 128 uniformly spaced prediction positions per sequence, exact full-vocabulary distributions.
Canonical correctness is verified with real contract/verifier; Student labels are NOT gold.
Do not infer correctness under a divergent Student prefix from canonical-prefix NLL.
Wrapper proxy vs body is descriptive, not ground-truth semantic attribution.
Outputs artifacts/cafd/experiments/mistral_cafd_direction_v15/{progress,diagnosis}.json.
Logs logs/cafd/direction-v15-41214438.{out,err}.
Tests tests/cafd/test_direction_v15.py: 2 passed before submission.
Previous v14 completed with PILOT_NOT_PASSED, 11/64 at40 and31/64 at80.
No permission inferred for full training or budget expansion from this diagnostic.
