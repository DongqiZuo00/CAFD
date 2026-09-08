# Continue this authorized experiment
All remote work stays in /blue/du.j/jinjiaguo/CAFD. SSH: wsl.exe -e ssh -o BatchMode=yes HiPerGator.
Read execution_prompt.md and execution_state.json here first.
The formal Slurm job is 41361573; smoke 41361075 passed two updates with an explicit round1 resume.
The formal script runs kd_retention_train, kd_retention_finalize twice (idempotency), then kd_retention_report.
Do not submit another job while it is active. Resume only after a recoverable failure and a confirmed terminal state.
Scripts: scripts/cafd/slurm/kd_retention_formal.sbatch (auto-detects resume).
Run: runs/cafd/experiments/mistral_cafd_kd_retention_replay_s2027.
Artifacts: artifacts/cafd/experiments/mistral_cafd_kd_retention_replay_s2027.
Reports and audits: reports/kd_retention_replay_s2027.
Heartbeat cafd-kd checks every 10 minutes and must continue through final delivery, then be stopped.
After Slurm finishes, rerun only the read-only report to include terminal allocation cost:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m cafd.kd_retention_report
Final checks: 200 unique rounds and optimizer updates, 800 groups/6400 answers, exact replay u and IDs,
7 development milestones with 64 records each, frozen best selection earliest tie, exactly132 test IDs,
no test regeneration on repeated finalize, Slurm terminal resource/cost and source provenance.
All old data/source/model archives remain unchanged. New code is isolated cafd/kd_retention_*.py.
The report is honest about absent old round0 development mean reward and absent explicit old repetition definition.
Apply the same post-hoc exact extracted-program identity rule to both old and new saved outputs.
No extra variants, baseline reruns, test selections or budget expansion.

Report recovery audit completed: report_engineering_20260908 contains an additional report-only patch and 34 passing tests. Training sources unchanged. Cost accounting merges test-only resumed jobs; missing costs stay unknown. Existing132-output finalize repairs progress without importing torch or regenerating. The formal-source snapshot is preserved; report-only later changes have separate before/after SHA manifests.

LATEST USER AUTHORIZATION (2026-09-08): retain200 rounds/800 groups/6400 answers and truthfully199 actual updates because round194 all32 full-pass skipped. Do not add training. Formal training41361573 exited FAILED1:0 only at strict completion guard after fullresume200 saved. CPU kd_retention_completion sealed metadata under explicit user_authorization.json. Selectedround12031/64 (round200 ties). Test-onlyjob41375988 runs scripts/cafd/slurm/kd_retention_test.sbatch. Never resubmit training. Finalize27 tests and report36 tests passed under narrow199-update clarification. Run final_output_audit.py after132test outputs complete, then kd_retention_report after terminalSlurm to count BOTH training41361573 and test41375988. Existing top-level200-update requirements are superseded only by this explicit user authorization.

Audit correction: cafd package __init__ imports torch through exact_forward_kl even on metadata-only entry. Do not equate library import with model loading. final_output_audit.py now blocks model/runtime imports, weight reads, torch model loading and CUDA initialization during repeated completed-output finalization, and compares all frozen hashes. Training/package source remains unchanged.
