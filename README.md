# Capability Acquisition Gap validation

This repository implements the single-seed candidate-gap experiment described in
`configs/gap_validation/`. It intentionally contains no CAFD mechanism, relative
policy targets, adjacent-checkpoint ratios, or transverse restoration.

All compute is launched from `/blue/du.j/jinjiaguo/CAFD`. The scheduler enforces a
hard ceiling of four allocated or pending B200 GPUs for this experiment.

The experiment is contract-gated: official benchmark access, deterministic splits,
overlap removal, tokenizer identity, verifier tests, exact forward-KL numerical
alignment, checkpoint/resume tests, and a hardware/NCCL probe must pass before a
claim-critical training job is eligible for submission.

Useful entry points:

```bash
scripts/gap_validation/status.sh
scripts/gap_validation/run_all.sh
scripts/gap_validation/resume_all.sh
scripts/gap_validation/summarize.sh
```

