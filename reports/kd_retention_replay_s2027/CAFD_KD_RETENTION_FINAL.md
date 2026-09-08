# KD coverage modification + old schedule replay

Completed 200 rounds and 199 optimizer updates; 800 prompt groups and 6400 Student answers.
Historical baseline completed 200 rounds, 200 optimizer updates and 6400 Student answers; this run completed 200 rounds, 199 optimizer updates and 6400 Student answers. The comparison shares the 200-round and 6400-answer budgets; optimizer-update counts differ.
User-approved protocol clarification: all 32 answers in round194 fully passed, so every group used the unchanged all-success skip rule. The user approved retaining 200 rounds with 199 actual optimizer updates and proceeding to the single frozen test. No extra training was authorized or added.
Development selected round120; one frozen test: 74/132. Historical baseline selected round120, test 15/132.
This run used the old realized teacher schedule, not adaptive MPC. Control observations and MPC recommendations remain diagnostic.

| round | old correct/64 | new correct/64 | old mean reward | new mean reward | old cumulative GPU h | new cumulative GPU h |
|---|---:|---:|---:|---:|---:|---:|
|0|0/64|0/64|missing|0.067401|0.135825|0.091944|
|10|0/64|0/64|0.013068|0.000000|0.770355|0.713611|
|40|0/64|13/64|0.751171|0.763226|1.223965|1.258889|
|80|2/64|26/64|0.808818|0.825360|1.810430|1.953611|
|120|5/64|31/64|0.846734|0.889333|2.347625|2.645556|
|160|4/64|30/64|0.853732|0.896772|2.804800|3.219167|
|200|2/64|31/64|0.832590|0.917642|3.250610|3.810556|

Old round0 per-task development outputs were not saved; its mean reward is unavailable.
Old milestone allocated times reconstructed from the last of each 64 development-generation events and UTC Slurm allocation start. New milestone times are recorded directly.

| test family | old correct/total | new correct/total |
|---|---:|---:|
|contains_count|6/31|8/31|
|contains_ordered|9/48|34/48|
|contains_substring|0/53|32/53|

Test difference (new-old): 44.697 percentage points; item-paired bootstrap 95% interval [34.848, 54.545] points (20,000 resamples). This interval excludes training-seed uncertainty.

| coverage | old | new |
|---|---:|---:|
|groups|800|800|
|tokens|1553276|2212434|
|actual_kd_groups|101|413|
|actual_kd_tokens|191373|1314082|
|actual_kd_group_fraction|0.12625|0.51625|
|actual_kd_token_fraction|0.12320604966535245|0.5939530851541787|
|all_fail_variable_groups|635|370|
|all_fail_variable_tokens|1237705|1214980|

Routing counterfactuals use each run's own outputs; they do not substitute for the old policy's training results.

Allocated Student run costs: old 3.446111 GPU h; new 4.043611 GPU h. Existing Teacher trajectory cost excluded.
Engineering smoke allocation: 0.101667 GPU h; separate from the formal run.
Common observed total budget: 3.446111 GPU h. See matched_compute.json for only measured, retained checkpoints; no interpolated performance.
New costs.json separately reports rollout, frozen scoring, teacher loading, optimization, control, development, probe setup and resume saving. Wall-clock operation bins are not exclusive hardware kernel timings: target vocabulary projections/recomputed backward remain in optimization; initialization_inclusive_seconds overlaps startup operations. GPU reservation totals are authoritative.

Test failure counts: old {"passed": 15, "semantic": 117}; new {"passed": 74, "semantic": 58}.
Repeated-program statistics use exact extracted program equality for both sets; historical formal artifacts did not define a repetition metric. See repeated_programs.json for top programs and per-family counts.

Single-seed historical comparison, not a multi-seed causal validation. Test tasks have historical evaluations; all 64 development tasks were exposed to Teacher training (44 SFT, 20 RL).
Any improvement first supports the supervision-coverage hypothesis; relative-versus-absolute target superiority remains untested. No additional variants or expanded budget were run.

## Locations
- Code: /blue/du.j/jinjiaguo/CAFD/cafd/kd_retention_*.py
- Minimal differences and validation: /blue/du.j/jinjiaguo/CAFD/reports/kd_retention_replay_s2027
- Initial CPU validation: /blue/du.j/jinjiaguo/CAFD/reports/kd_retention_replay_s2027/cpu_tests.xml
- Report cost/recovery validation: /blue/du.j/jinjiaguo/CAFD/reports/kd_retention_replay_s2027/report_engineering_20260908/pytest.xml
- Authorized round/update boundary validation: /blue/du.j/jinjiaguo/CAFD/reports/kd_retention_replay_s2027/protocol_clarification_20260908/report_pytest.xml
- Authorized report-only changes: /blue/du.j/jinjiaguo/CAFD/reports/kd_retention_replay_s2027/protocol_clarification_20260908/report.patch
- Resume: /blue/du.j/jinjiaguo/CAFD/runs/cafd/experiments/mistral_cafd_kd_retention_replay_s2027/resume.pt
- Selected model: /blue/du.j/jinjiaguo/CAFD/runs/cafd/experiments/mistral_cafd_kd_retention_replay_s2027/round120
- Training groups/control/development outputs: /blue/du.j/jinjiaguo/CAFD/runs/cafd/experiments/mistral_cafd_kd_retention_replay_s2027
- Training curve/costs/test outputs: /blue/du.j/jinjiaguo/CAFD/artifacts/cafd/experiments/mistral_cafd_kd_retention_replay_s2027

Known CSV field-order finalization issue fixed in the isolated finalization module. Existing 132 saved baseline outputs re-sealed twice in a scratch fixture without changes; formal finalization is invoked twice and must not regenerate completed test outputs.
