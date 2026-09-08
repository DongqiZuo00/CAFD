# CAFD-MPC 完整主实验启动记录

## 范围与状态

用户在 2026-09-06 授权启动完整 CAFD 实验。本次按 CAFD_Method_MPC_ZH.md 的完整方法执行，不是上一轮 T1/T2 诊断，也不是两条纯 KD 目标对照。

- 正式 run_id：mistral_cafd_mpc_v1_formal
- 首个 Slurm job：41249364
- 启动：2026-09-06 14:11:23 America/New_York
- 节点：c1001a-s5
- 请求：1×B200、8 CPU、192G、24小时单次作业上限
- 项目约束：最多2×B200，内存预留不得超过192G
- 工作目录：/blue/du.j/jinjiaguo/CAFD
- 不启动两条纯 KD、其他 baseline、Teacher 重训或调度搜索。
- 其他项目作业不改动；旧 checkpoint、resume 与 T1/T2 结果保留。

## 冻结协议

Student：原始 Ministral-3-3B-Instruct-2512-BF16 S0（不是 capacity-SFT）。
Teacher：Ministral-3-8B-Instruct-2512-BF16。
真实路线：原始 Instruct → SFT125 → RL40 → RL60 → RL80 → RL100。

目标为永久冻结 S0 基准的累计相对目标：
q_u = stopgrad(softmax_FP32(z_S0 + (1-alpha) z_Tm + alpha z_Tnext - z_T0))。

只使用当前 Student 生成前缀。同奖全失败组进行 exact full-vocabulary forward KL；
奖励有差异的组进行组相对 clipped RL；全成功组零贡献；分母包含全批有效 completion tokens。
全零贡献轮次不执行 optimizer step，但仍计入200轮采样预算。

seed=2027；200 rollout-update rounds；每轮4 prompts×8 rollouts；
max_new_tokens=2048；temperature=1、top_p=1、top_k=0；
FP32-master AdamW、恒定lr=1e-6、weight_decay=0、grad_clip=1。
K=10、H=2、六个初始化块、epsilon=.1、gamma=.95、ridge=1。
控制器只使用控制题观测，不读取development分数。
预算结束允许u<M，不因为对照方法表现或中途成绩扩大预算。

## 数据与结果门槛

602道优化题、12道固定控制题、64道development题，按ID分离。
development milestones：0、10、40、80、120、160、200；完整通过数最高者入选，平分保留较早checkpoint。
只有200轮正式训练完成、selected.json已经冻结且与complete.json一致，才执行132题最终greedy评测。
最终评测沿用相同chat renderer、2048生成上限、停止规则与官方verifier。
不得根据最终test改变checkpoint、配置、路线或后续结论。

development64包含Teacher SFT44/RL20训练暴露，不能作为全流程独立泛化证据。
历史实验已观察过132题test；本次冻结后评测不使该测试集重新成为研究过程从未见过的数据。
T1目标策略成功率不是KD Student的硬天花板；旧CAFD结果不属于本MPC版本。

## 实际执行与审计

提交命令：sbatch --parsable scripts/cafd/slurm/mpc_v1_formal.sbatch
训练入口：.venv/bin/python -u -m cafd.mpc_train --profile formal --run-id mistral_cafd_mpc_v1_formal
末端入口：.venv/bin/python -u -m cafd.mpc_finalize --run-id mistral_cafd_mpc_v1_formal

提交前147项CPU测试通过（28.30秒）；节点启动后再次147项通过（22.46秒）。
已有独立64-token集成smoke通过，但不作为本次正式效果结果。
本次训练源快照：runs/cafd/experiments/mistral_cafd_mpc_v1_formal/source_training_41249364.tar.gz。
实际Slurm脚本：同目录executed_job_41249364.sbatch。

运行信息：
- runs/cafd/experiments/mistral_cafd_mpc_v1_formal/
- artifacts/cafd/experiments/mistral_cafd_mpc_v1_formal/
- state/cafd/experiments/mistral_cafd_mpc_v1_formal/state.json
- logs/cafd/mpc-v1-formal-41249364.out
- logs/cafd/mpc-v1-formal-41249364.err

完整resume在初始化后及每个完整block边界保存。
physical_costs.jsonl跨attempt累计，不把回滚前物理计算记为零。
24小时是作业上限而非预计完成时间；不能用T1纯推理吞吐直接当作本次训练ETA。

后台跟进每15分钟检查，只在完成、失败或需要处理时通知。
正式200轮未完成时，本文件仅为启动记录，不是完成报告或方法有效性结论。
