# CAFD 当前结果与实验说明书

记录日期：2026-09-09  
用途：内部工作记录、结果核对与复现。本文汇总已完成实验，未新增训练或测试。

## 1. 当前结论

在 DELTA Manufactoria-HAS 的固定 132 道官方测试题、seed=2027 和当前检查点选择规则下，绝对目标保留式 KD+RL 得到 **88/132（66.67%）**，相对目标 CAFD 得到 **74/132（56.06%）**，旧版 CAFD-MPC 得到 **15/132（11.36%）**。

绝对目标比相对目标多通过 14 道题，提升 **10.61 个百分点**。当前结果支持把绝对目标 KD+RL 作为后续工作的最小方法；本次实验没有显示相对目标优于绝对目标。该结论限于本数据集、单个训练种子和现有验证协议，不能据此声称跨数据集优势或 SOTA。

直接 GRPO 的开发集检查点均为 0/64，按预先规定的“通过数最高、同分取最早”规则选中 round0，即原始 S0。因此表中的 GRPO 0/132 是被选中的 S0 的测试结果，**不是训练到 round200 的 GRPO 检查点测试结果**。

## 2. 实验问题与比较条件

| 条件 | 实际定义 | 主要用途 |
|---|---|---|
| S0 | 原始 Student，不训练 | 衡量初始能力 |
| GRPO | 仅在组内奖励有差异时进行 RL 更新，关闭训练 KD | 检查直接奖励学习 |
| 旧版 CAFD-MPC | 在线选择 Teacher 路径位置；仅“全部失败且同奖励”组使用 KD | 历史参照 |
| 相对目标 CAFD | 保留全部失败组的 KD，并使用相对目标；回放旧 MPC 路径 | 检查扩大 KD 覆盖后的表现 |
| 绝对目标 KD+RL | 与相对目标采用相同覆盖规则和回放路径，改用绝对 Teacher 目标 | 隔离目标构造的影响 |

绝对目标与相对目标是本轮最直接的方法对照。旧 MPC 与后两者还存在 KD 路由、在线控制与路径回放的区别，不能把它们的差值全部归因于目标公式。GRPO 实现保留了 Teacher/control 诊断计算，不能将其耗时理解为精简、完全不加载 Teacher 的 GRPO 成本。

## 3. 数据与模型

### 3.1 数据划分

数据集为 DELTA Manufactoria-HAS，任务族为 contains_count、contains_ordered 和 contains_substring。

| 分区 | Count | Ordered | Substring | 总数 | 用途 |
|---|---:|---:|---:|---:|---|
| Optimization | 128 | 239 | 235 | 602 | Student 训练采样 |
| Control | 4 | 4 | 4 | 12 | 控制与诊断 |
| Development | 14 | 25 | 25 | 64 | 检查点选择 |
| Official test | 31 | 48 | 53 | 132 | 冻结后的最终评估 |

源训练集有 742 题；除 602+12+64 外，另有 64 道历史 confirmation 题，本轮未使用。Student 分区按问题 ID 分离。

**暴露历史：**64 道 Development 题全部曾被 Teacher 训练使用，其中 44 道用于 SFT、20 道用于 RL。开发集和官方测试集也有历史评估记录。本轮冻结评估可以核查选定检查点的表现，但不构成从未接触过的全新盲测；Student 分区互斥也不等于 Teacher 未见过开发题。

划分依据：`data/cafd/mpc_v1/manifest.json`、`optimization.jsonl`、`control.jsonl`、`development.jsonl`、`test_ids.json`。测试内容位于 `data/cafd/test.jsonl`。准备阶段 manifest 中的 evaluated=false 是当时状态，不能作为当前测试未完成的依据。

### 3.2 Backbone 与 Teacher 路径

- Student：`mistralai/Ministral-3-3B-Instruct-2512-BF16`
- Student revision：`b6d637bef2393152b3da2b2fde72eecdee30557e`
- Teacher：`mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Teacher revision：`f6fae9795746f63c9be8344932f01275f3c63734`

S0 为原始 Student 初始化，不是额外做过 capacity SFT 的模型。逻辑 S0 引用解析到 `runs/cafd/experiments/mistral_cafd_only_v6/student_base/S0`。

复用已有 Teacher 轨迹：

`T0=Base → T1=SFT125 → T2=RL40 → T3=RL60 → T4=RL80 → T5=RL100`

这是 **SFT+RL 轨迹**，不是纯 RL 轨迹。本轮未重新训练 Teacher。路由文件为 `runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/route.json`。

## 4. 方法定义

### 4.1 两种蒸馏目标

所有模型在同一条 Student 采样前缀上计算 logits，使用相同 token-ID 映射，词表大小 131072。设路径位置 u∈[0,5]，m=min(floor(u),4)，α=u−m。softmax 温度为 1，目标停止梯度。

**相对目标：**

`q_rel = stopgrad softmax_FP32[z_S0 + (1−α)z_Tm + αz_T(m+1) − z_T0]`

S0 始终固定，不随 Student 更新。该构造把 Teacher 相对初始 Teacher 的 logit 变化加到固定 S0 上。

**绝对目标：**

`q_abs = stopgrad softmax_FP32[(1−α)z_Tm + αz_T(m+1)]`

绝对目标直接使用路径上相邻 Teacher logits 的线性插值，不含 S0 锚点或减去 T0 的项。插值发生在 logits 上，不是两个概率分布的线性混合。

### 4.2 KD / RL 路由

每轮 4 个 prompt，每个 prompt 采样 8 个回答，每组独立判断：

- RL：组内 max(reward)−min(reward)>0。
- 保留式 KD：8 个回答均未全通过 verifier，无论奖励是否相同。
- 因而，“全部失败但部分奖励不同”组同时产生 KD 与 RL 损失。
- 全部成功组跳过；若整轮没有任何有效损失，跳过 optimizer.step、动量和 weight decay 更新，但仍计为一个 rollout 轮次。
- 旧 MPC 的 KD 仅覆盖“全部失败且奖励相同”组，因此没有 KD/RL 重叠。
- GRPO 关闭训练 KD，仅保留 RL 路由。

### 4.3 损失与数值实现

组内优势使用总体标准差：

`A_gj = (r_gj − mean_j(r_gj)) / (population_std_j(r_gj) + 1e−6)`

Z 是完整 4×8 batch 中所有有效 completion token 数，包含跳过组的 token。KD 与 RL 均使用这个分母：

`L_KD = (1/Z) Σ_KD位置 KL(q || p_θ)`

`ρ = exp(log p_θ(y_t|prefix) − log p_behavior(y_t|prefix))`

`L_RL = −(1/Z) Σ_RL位置 min[ρA, clip(ρ,0.8,1.2)A]`

`L = L_KD + L_RL`，KD 条件的两个系数均为 1；GRPO 的训练 KD 为 0。

每轮使用当前 Student 的独立冻结副本作为 behavior model，并记录实际采样 token 的 log probability。KD 为全词表精确 forward KL，采用 token_chunk=64 的流式 LM head 投影，不使用 top-k KL 近似。模型权重使用 BF16，目标 softmax、相关投影及优化器 master 参数/状态使用 FP32；梯度仅更新 Student。

Completion mask 包含首个回答 token 和实际生成的 EOS，排除 prompt、padding 和 EOS 后的位置。生成按外层闭合代码围栏或实际 EOS 停止，不补造 EOS。每个有贡献的轮次只进行一次梯度裁剪与优化器更新。

### 4.4 训练奖励与最终指标

| Verifier 状态 | 奖励 |
|---|---:|
| 无法提取符合约定的程序 | 0 |
| 已提取程序，但 verifier 判定无效 | 0.05 |
| 有效程序，但测试用例通过率为 0 | 0.10 |
| 部分测试用例通过 | 0.10 + 0.90 × pass_rate |
| 全部测试用例通过 | 1 |

训练使用上述分级奖励。最终主指标是“完整通过全部 verifier 测试的题数 / 132”，不能用平均奖励代替。fullpass 标记与 reward=1 的一致性已审计。

## 5. 固定预算与训练配置

| 参数 | 配置 |
|---|---|
| 随机种子 | 2027 |
| 训练预算 | 200 rollout 轮 |
| 每轮采样 | 4 prompts × 8 answers |
| 每个训练条件总回答数 | 6400 |
| Prompt / completion 上限 | 4096 / 2048 tokens |
| 训练采样 | temperature=1，top_p=1，top_k=0 |
| 优化器 | FP32-master AdamW |
| 学习率 / weight decay | 固定 1e−6 / 0 |
| 梯度裁剪 | 1.0 |
| RL ratio clip | 0.2 |
| KD / RL 系数 | 1 / 1 |
| Control | 每族 4 题，每题 2 rollouts |
| 路径块长度 | 10 轮 |
| 开发集检查点 | 0、10、40、80、120、160、200 |
| 新增作业资源 | 单作业 1 B200、192G，串行执行 |

相对目标、绝对目标和 GRPO 回放旧 MPC 的 20 个块位置：

`[0.5,1,1.5,1.5,1.5,2,2.5,2.5,3,3.5,4,4.5,5,5,5,5,5,5,5,5]`

保留的诊断不能修改回放 u。旧 MPC 则在线运行控制器（ridge、horizon=2、block=10）。这些配置的区别应在复现实验中保留。

各训练条件对齐有序的 800 次 prompt 呈现与显式 rollout seeds。训练后模型不同，生成长度、输出内容和有效更新次数随之不同。因此这是固定 rollout 预算比较，**不是 token 数或 optimizer 更新次数完全匹配的比较**。

新增基线先经过独立两轮 smoke 和保存/恢复检查；正式训练重新从 S0 开始，不继承 smoke 权重。

## 6. 检查点选择与冻结评估

检查点按 Development 全通过题数最大选择，同分取最早轮次，不按平均奖励或测试结果选择。确定检查点后，官方 132 题各生成一次 greedy 回答，max_new_tokens=2048，使用完整 verifier 判定。

逐题输出写入可恢复日志，已完成题不重新生成。后续 CPU 汇总与幂等性检查不增加模型测试采样。这里的“冻结”指本轮选定检查点和测试输出被冻结，并不抹去数据的历史使用记录。

## 7. 当前结果

### 7.1 主结果

| 方法 | 训练轮数 | 实际更新 | 选中轮次 | 测试通过 | 准确率 | 分配 GPU 小时 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 S0 | 0 | 0 | 0 | 0/132 | 0.00% | 0.1644 |
| GRPO（选中 S0） | 200 | 29 | 0 | 0/132 | 0.00% | 1.1597 |
| 旧 CAFD-MPC | 200 | 200 | 120 | 15/132 | 11.36% | 3.4461 |
| 相对目标 CAFD | 200 | 199 | 120 | 74/132 | 56.06% | 4.0436 |
| 绝对目标 KD+RL | 200 | 200 | 200 | 88/132 | 66.67% | 3.9239 |

GPU 小时按已记录作业分配时间统计，包含对应训练/评估流程开销；S0 只有评估。表中不含已有 Teacher 轨迹的训练成本。

### 7.2 按任务族

| 方法 | Count（31） | Ordered（48） | Substring（53） |
|---|---:|---:|---:|
| S0 | 0 | 0 | 0 |
| GRPO（选中 S0） | 0 | 0 | 0 |
| 旧 CAFD-MPC | 6 | 9 | 0 |
| 相对目标 CAFD | 8 | 34 | 32 |
| 绝对目标 KD+RL | 4 | 43 | 41 |

相对目标换为绝对目标后，Ordered 和 Substring 各增加 9 题，Count 减少 4 题。提升并非覆盖所有任务族。绝对目标剩余 44 道失败均归为语义失败，没有解析、格式或截断失败；测试共生成 45,283 个回答 token。

### 7.3 开发集曲线：全通过题数 / 64

| 方法 | 0 | 10 | 40 | 80 | 120 | 160 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 旧 CAFD-MPC | 0 | 0 | 0 | 2 | 5 | 4 | 2 |
| 相对目标 CAFD | 0 | 0 | 13 | 26 | 31 | 30 | 31 |
| 绝对目标 KD+RL | 0 | 0 | 11 | 27 | 28 | 36 | 45 |

旧 MPC 的 round0 保留了通过数 0，但没有保存该轮逐题输出和平均奖励，不能声称其 round0 具备与后续轮次相同的审计粒度。相对目标 round120 与 round200 同为 31，故选择 round120。

### 7.4 绝对目标与相对目标的逐题配对比较

| 结果组合 | 题数 |
|---|---:|
| 两者都通过 | 62 |
| 仅绝对目标通过 | 26 |
| 仅相对目标通过 | 12 |
| 两者都失败 | 32 |

差值为 (26−12)/132，即 **+10.61 个百分点**。按冻结 row_id 配对重采样，bootstrap 20,000 次、seed=2027，95% 区间为 **[+1.52，+19.70] 个百分点**。该区间只反映当前测试题的重采样变化，不衡量训练种子间的不确定性。

## 8. 训练覆盖与异常记录

| 方法 | 总组数 | 回答 token | KD 组 | KD token | RL 组 | RL token | 重叠组 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 800 | 257,924 | 0 | 0 | 45 | 71,061 | 0 |
| 旧 CAFD-MPC | 800 | 1,553,276 | 101 | 191,373 | 696 | 1,357,788 | 0 |
| 相对目标 CAFD | 800 | 2,212,434 | 413 | 1,314,082 | 714 | 2,033,799 | 370 |
| 绝对目标 KD+RL | 800 | 2,351,850 | 368 | 1,293,807 | 680 | 2,087,128 | 332 |

KD 与 RL 可以覆盖相同 token，两列不能相加当作独立生成总量。旧 MPC 有 635 个“全部失败但奖励不同”的组，它们未进入旧版 KD 路由。

**GRPO：**6400 个训练回答中，6346 个奖励为 0.05（format_only），54 个为 0（invalid_format），无全通过回答。只有 45 组满足奖励变化条件，分布在 29 个实际更新轮次；另外 171 轮合法跳过。路由与原始奖励审计未发现不一致。独立保存的 GRPO 选中模型与 S0 的 132 道测试输出、token 序列和 verifier 字段一致。

**相对目标 round194：**当轮 32 个回答全部成功，按路由跳过更新。因此实际为 200 轮、199 次更新。用户已批准保留原 200 轮预算，不补一轮。训练作业在完成并保存 200 轮后，因旧收尾逻辑强制要求 200 次更新而返回 FAILED 1:0；之后通过 CPU 如实封存，再进行冻结测试，没有为满足计数而重训。

## 9. 运行与成本记录

| 作业 | Job ID | 分配秒数 |
|---|---|---:|
| GRPO smoke | 41412450 | 191 |
| 绝对目标 smoke | 41412451 | 264 |
| S0 冻结测试 | 41413618 | 592 |
| GRPO 正式训练及测试 | 41413796 | 4175 |
| 绝对目标正式训练及测试 | 41413797 | 14126 |
| 相对目标训练 | 41361573 | 13848 |
| 相对目标冻结测试 | 41375988 | 709 |
| 旧 MPC | 41249364 | 12406 |

新增三个正式条件合计 18,893 秒（5.2481 GPU 小时），新增 smoke 合计 455 秒（0.1264 GPU 小时）。绝对目标冻结快照中的 14,110 秒为当时不可变快照；最终成本采用作业终态的 14,126 秒。

工作根目录为 `/blue/du.j/jinjiaguo/CAFD`。用户资源上限为 192G RAM、最多 4 B200；上述新增作业实际使用 1 B200 串行运行。

## 10. 记录索引与复现入口

### 10.1 Run ID

| 条件 | Run ID |
|---|---|
| S0 | mistral_cafd_minimal_s0_s2027 |
| GRPO | mistral_cafd_minimal_grpo_s2027 |
| 绝对目标 | mistral_cafd_minimal_absolute_kd_s2027 |
| 相对目标 | mistral_cafd_kd_retention_replay_s2027 |
| 旧 MPC | mistral_cafd_mpc_v1_formal |

每个 Run ID 对应根目录下的 `runs/cafd/experiments/`、`artifacts/cafd/experiments/` 和 `state/cafd/experiments/`。具体模型和检查点引用以 model_index.json 为准，避免只凭目录名称猜测。

最终证据目录：`reports/minimal_baselines_s2027/`。

| 文件 | 内容 |
|---|---|
| protocol.json | 固定实验协议 |
| conditions.csv、test_by_family.csv | 主结果与分族结果 |
| development.csv、coverage.csv | 开发曲线和训练覆盖 |
| paired_bootstrap.json | 配对比较 |
| allocations.csv | 成本和作业记录 |
| comparison.json | 汇总状态；complete，blockers 为空 |
| model_index.json、runtime_versions.json | 模型引用与环境版本 |
| source_proof.json、validated_pipeline_sha256.json | 源码及流水线校验 |
| final_comparison_audit.json | 最终比较审计，838 项检查通过 |
| s0_final_audit.json | S0 审计，36 项通过 |
| grpo_final_audit.json | GRPO 最终审计，49 项通过 |
| absolute_kd_final_audit.json | 绝对目标最终审计，41 项通过 |
| delivery_archive_audit.json | 交付包 556 个文件的校验记录 |

原始相对目标报告在 `reports/kd_retention_replay_s2027/`。旧 MPC 结果在 `reports/CAFD_MPC_FINAL_41249364.md`；方法历史在 `reports/CAFD_MPC_IMPLEMENTATION_20260906.md` 与 `docs/methods/CAFD_MPC_20260906/`。历史说明中的“尚未提交”等描述是当时状态，以最终报告为准。

### 10.2 代码与操作入口

核心源码包括：

- `cafd/minimal_baseline_objective.py`、`minimal_baseline_train.py`：最小基线目标与训练。
- `cafd/kd_retention_objective.py`、`kd_retention_train.py`：相对目标与保留式 KD。
- `cafd/mpc_objective.py`、`mpc_train.py`、`mpc_controller.py`：旧 MPC。
- `cafd/verifier.py`：奖励与通过判定。
- 对应的 runtime、finalize、report 模块：运行、冻结与汇总。

训练入口为 `.venv/bin/python -m cafd.minimal_baseline_train --condition grpo|absolute_kd --profile formal`；这里的竖线表示两个可选条件，不是可直接复制的 shell 命令。精确复现应先恢复归档、核对依赖/模型 revision、数据及流水线哈希，再核对归档中的配置和 Slurm 脚本。已有 Run ID 和冻结文件应保留；新实验使用新 Run ID。本说明书没有启动这些命令。

已有结果可通过 `PYTHONPATH=$PWD .venv/bin/python -m cafd.minimal_baseline_report --final` 汇总，不需要重新生成测试答案。模型权重等二进制文件未包含在 GitHub 工作备份中，恢复文档和日志不等于已经恢复全部可执行模型资产。

### 10.3 GitHub 备份

仓库：[DongqiZuo00/CAFD](https://github.com/DongqiZuo00/CAFD)。

本说明书之前的完整工作备份提交为 `789b5b02f5cc7d6dda43ddd18b403104f168cc41`。备份包含 9,163 个工作文件，原始大小 1,929,653,192 字节；压缩流 386,255,808 字节，拆分为 62 段。压缩流 SHA256：

`3e2973a7560d6ba3e9b989c0379f3c7951e3fa59dbc0b8a678e951d76030e126`

完整公开仓库下载、恢复和逐文件校验已通过。恢复方式见仓库根目录 [BACKUP_RESTORE.md](../BACKUP_RESTORE.md) 和 [restore_backup.py](../restore_backup.py)，校验记录见 [github_roundtrip_audit.json](../backup/2026-09-08/github_roundtrip_audit.json)。

备份包含代码、方法、数据划分、逐题输出、日志和工作记录；排除了模型/优化器/探针二进制、环境和缓存，排除项有索引。本文作为后续新增文件单独提交，不修改旧归档或伪造旧备份时间。

## 11. 后续实验建议（尚未执行）

1. 以绝对目标保留式 KD+RL 为最小方法，先对绝对/相对目标做多个独立训练种子重复，保留一致的选择规则，报告均值与种子间变化。
2. 在下一轮训练前固定 Teacher 未见过的验证协议；若要得出更强的泛化结论，增加新任务或新数据集，避免继续依赖历史测试结果调参。
3. 核查 Count 的成对失败案例：绝对目标只有 4/31，相对目标为 8/31。先确认具体语义错误，再决定是否修改目标。
4. 若比较效率，另设明确的 token 或计算预算匹配实验，并单独测量去掉 Teacher 诊断的 GRPO。当前固定轮次成本表不能替代该实验。

当前完成的是单种子、固定预算、已冻结输出的内部方法比较。公开 baseline 的外部数值尚未纳入这份实验表；没有核对一致的数据版本、backbone、训练暴露和评估协议之前，不与文献数值直接排名。
