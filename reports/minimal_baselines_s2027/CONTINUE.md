# 已完成：最小对照最终交付

全部授权实验、冻结测试、独立审计与归档已完成；原生 heartbeat cafd-kd 已暂停。无需继续提交、重训或重测。

主摘要：FINAL_SUMMARY_ZH.md；交付包：CAFD_MINIMAL_BASELINES_DELIVERY.tar.gz；556个文件校验通过。ABS 88/132，relative CAFD 74/132，GRPO按预定规则选回S0后测试0/132。

以下为保留的执行历史，不能视为新的执行指令。

# Active authorized minimal comparison

User: 做最小方法。 Scope is the previously proposed minimum comparison: original S0 evaluation, direct GRPO, and absolute-target KD+RL. Existing relative-retention and old baseline are completed historical references, read-only. No new variants or extra seeds.

All work stays in /blue/du.j/jinjiaguo/CAFD. Use .venv/bin/python and PYTHONPATH=root. GPU jobs run serially with 1 B200, 192G memory, max 1 active job in this experiment set. Keep all cache/temp directories inside CAFD.

Read execution_state.json and protocol.json here for exact IDs and current status. CPU validation artifacts: training_validation, finalization_validation, report_validation. Training source admission hashes are immutable. Never rerun running/pending work or regenerate completed test rows. Source and test hashes from historical_source_sha256_before.json must remain unchanged.

Historical smoke submission (both now COMPLETED): GRPO41412450 and absolute41412451 (afterok). They each perform round1 save and a new-process resume to round2. Legitimate GRPO smoke may have zero updates; do not confuse empty optimizer-state validation with nonempty momentum validation. CPU dense tests cover the latter. The completion budget_audit must agree with recomputed group routing and round flags.

CPU validation and real smoke validation are complete. S0 predeclaration, source archive and all three formal/evaluation submissions are complete. See execution_state.json for authoritative IDs. Do not submit initial jobs again. Read validated_pipeline_sha256.json, gpu_smoke_audit.json and validation folders for evidence. Every formal condition has a separate allocation; dependencies enforce S0 then GRPO then absolute KD.

Each trained condition runs 200 rollout rounds (4 prompts x 8 answers per round), not a forced number of optimizer steps. The round budget is 6400 answers and is never extended for legitimate skips. Selection is development full-pass best, earliest tie across rounds0,10,40,80,120,160,200. Finalizer runs twice: the second call must not load model weights or regenerate any test output. No tests before selection freezes except evaluation-only S0.

If recoverable failure occurs, wait until job is terminal, preserve attempts/cost ledger and source-fix evidence, then resume from saved complete block. Finalizer journal continues only unseen rows. A training complete.json may precede test completion. CPU report --partial is safe at any point; --final requires completed terminal allocations and all five test summaries. Never invoke --final in an active GPU allocation, because it deliberately returns2 until allocations are terminal.

Final result must cover all three new results and both historical references, by-family full-pass, dev curves and frozen selections, actual update counts, costs and paired bootstrap against relative CAFD. Use Slurm train/test job union including failed attempts; smoke as separate engineering cost union. GRPO retains historical teacher/control diagnostics and this overhead must be disclosed. The test has historical usage and the dev split was exposed to Teacher training; this is a single-seed matched comparison, not a clean pipeline-held-out multi-seed study.

Run final source/test integrity and output-journal audit after terminal jobs. Deliver a concise Chinese report in this CAFD report directory and mark execution_state COMPLETE only after all work succeeds. Existing Codex heartbeat cafd-kd has been reactivated for this experiment; keep unchanged progress silent, notify milestones/errors/completion, pause after final delivery.

## Current submission update

CPU suites and both real smoke jobs passed. S0 predeclared and submitted as 41413618. GRPO formal 41413796 waits afterok S0; absolute formal 41413797 waits afterok GRPO. Do not repeat the earlier Next submission instructions: all initial jobs are now submitted. validated_pipeline_sha256.json and validated_source_snapshot.tar.gz are frozen; validate_readiness.py passed before formal submission and runs again in each job. gpu_smoke_audit.json records honest per-condition update counts and verification scope.

## Reporting schema correction after S0 completion

S0 passed the complete real-output audit: 0/132; terminal Slurm592s (frozen in-job snapshot579s is retained). Report admission had used selection.rule instead of the real selection.selection_rule. Only that report field was corrected, with24 CPU tests and real S0/historical compatibility checks. Training and frozen outputs unchanged. Pipeline manifest v1 and v2 plus exact change record are retained in pipeline_manifest_history; validated_pipeline_sha256.json now records the validated report v2 hash so the waiting ABS job passes unchanged readiness.py. Source snapshots v1/v2 both retained. Readiness gate passed again. Do not restore v1 report hash or rerun any completed generation.

## GRPO completed and audited

GRPO41413796 finished COMPLETED0:0,4175s total train+test (frozen in-job snapshot4161s retained). Training200rounds/6400answers,29actual updates171legal skips; grpo_training_audit.json passed. All seven dev checkpoints0/64, earliest-tie selected round0 original S0. Frozen test0/132; grpo_final_audit.json passed49 checks including132 identical token/program/verifier records versus independent S0 and blocked-model CPU idempotency. This is the selected S0 endpoint, not a round200 test. ABS41413797 is now running; do not submit/retest/retrain GRPO or S0. Four of five comparison conditions have final results. Keep following ABS through its200rounds, frozen test, terminal cost and final combined report.

## Required final interpretation

The primary Chinese final summary must explicitly say that GRPO selected round0 original S0 because all seven development full-pass scores tied at0/64. Its0/132 frozen result therefore evaluates selected S0, not the round200 trained checkpoint. Do not claim a measured round200 test score. Final comparison is matched200 rollout rounds, not matched optimizer updates (GRPO29, relative199, old200, ABS actual count pending). Preserve disclosure of teacher/control diagnostics, Teacher-exposed development and single-seed/item-bootstrap limitations. The existing generated report shows selected round0 in its table but lacks the explicit GRPO explanation; add this to the primary final summary without modifying validated report code. See final_interpretation_review.json when present.

## Final delivery preparation

ABS reached200rounds/200updates; dev20045/64 is best. Check current complete/selected and test progress before any action. build_final_delivery.py is prepared but must run ONLY after comparison.json is complete and S0/GRPO/ABS final audits, both new training audits, and final_comparison_audit.json all report status passed. It generates FINAL_SUMMARY_ZH.md with explicit GRPO selected-S0 meaning, model/runtime index, checksum manifest and verified evidence archive; it excludes model weights. Validated training/finalization/report source has not changed. Final source/data hash check, terminal-cost report --final and independent comparison audit are still required. Existing heartbeat should be paused only after final delivery.

ABS final training audit is now passed:200rounds/200updates, unique best development round20045/64, frozen checkpoint round200. Final test proceeds in41413797. After132 outputs and COMPLETED job: execute absolute_kd_final_audit.py with PYTHONPATH set to CAFD root, run cafd.minimal_baseline_report --final, independently audit final comparisons and write final_comparison_audit.json(status passed), then execute build_final_delivery.py. Review final summary and verified archive, mark execution_state COMPLETE, pause native heartbeat cafd-kd and give user the concise five-condition result. Do not regenerate existing test rows or add training.
