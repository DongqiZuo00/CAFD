# CAFD-MPC 正式结果：41249364

## 结论

完整200轮、200次optimizer更新已完成。Development-only选中round120（5/64=7.8125%），冻结后在132题测试集得到15/132=11.3636%。
本次结果显示学生获得了有限的完整通过能力，但整体表现较弱；没有运行同协议对照，不能宣称CAFD-MPC优于GKD、提高蒸馏效率或达到SOTA。
旧CAFD与T1目标策略结果均不是本版本Student的结果，目标策略成功率不是Student硬天花板。

## 配置与选择

- Benchmark：DELTA Manufactoria-HAS。
- Student：mistralai/Ministral-3-3B-Instruct-2512-BF16；Teacher：Ministral-3-8B-Instruct-2512-BF16。
- seed2027，200 rollout-update rounds，每轮4 prompts×8 rollouts，生成上限2048。
- 永久S0累计相对目标，当前Student前缀，精确全词表forward KL；奖励差异组走RL、同奖全失败组走KD、全成功组跳过。
- K10/H2 MPC；冻结六个真实Teacher节点；本轮没有重训Teacher。
- checkpoint：/blue/du.j/jinjiaguo/CAFD/runs/cafd/experiments/mistral_cafd_mpc_v1_formal/round120
- 200轮预算没有因为选择round120而缩减。第200轮development为2/64，并非最终测试的checkpoint。
- 选择冻结时间早于最终评测；没有根据test重新选择checkpoint。

| development轮次 | 正确数 | 准确率 |
|---|---:|---:|
| 0 | 0/64 | 0% |
| 10 | 0/64 | 0% |
| 40 | 0/64 | 0% |
| 80 | 2/64 | 3.125% |
| 120（选中） | 5/64 | 7.8125% |
| 160 | 4/64 | 6.25% |
| 200 | 2/64 | 3.125% |

## 最终测试：官方full-pass，greedy，每题一次

| 题型 | 正确数 | 准确率 |
|---|---:|---:|
| contains_count | 6/31 | 19.3548% |
| contains_ordered | 9/48 | 18.75% |
| contains_substring | 0/53 | 0% |
| 合计 | 15/132 | 11.3636% |

132个唯一题目ID及顺序与冻结清单完全一致，逐题结果汇总与final_scores一致。
失败类型：117条语义失败；没有格式失败、解析失败或基础设施故障。
本次失败主要体现为程序没有完整通过测试，而非输出包装损坏。
这只是失败类型描述，尚不能从此确定算法的因果失败机制。

## 调度与训练分流

20个完整block：6个初始化、1个随机探索、6个多动作MPC决策、7个终点保持。
6次MPC均选择推进；第121轮开始进入u=5，最终u=5。
共800个prompt groups：KD101（12.625%）、RL696（87%）、全成功跳过3（0.375%）。
因此大部分题组实际由RL分支更新；这是已记录的路由事实，不证明它是低成功率的原因。
预测器总拟合与搜索CPU计时约0.718秒，不含控制探针GPU成本。
完整MPC数据保存在controller_decisions.jsonl、observations.jsonl和transitions.jsonl。

## 实际成本（没有新增训练作业）

作业41249364：COMPLETED，exit0:0，节点c1001a-s5，1×B200、192G、8CPU。
Slurm终态elapsed=12406秒=3小时26分46秒，总预约GPU-hours=3.446111。
final_scores生成时账本为3.445556 GPU-hours，比Slurm真正终态少2秒；以终态为总资源数，不覆盖原始账本。

- 训练runner墙钟11797.888秒（约3小时16分38秒），包括控制、development和保存。
- 最终评测操作526.235秒，加模型加载17.154秒，约9分3秒；其余为进程与作业收尾等开销。
- 训练采样生成：1,553,276 tokens。
- 控制生成：117,739 tokens。
- development生成：220,289 tokens。
- 固定probe初始化生成：3,170 tokens。
- 最终test生成：32,633 tokens。
- 全部生成合计：1,927,107 tokens。
- 训练Teacher completion评分：409,155 tokens；固定probe Teacher评分6,340；合计415,495。
- 训练S0 completion评分191,373；固定probe S0评分3,170；当前Student控制评分66,570，不能混记为Teacher评分。
- Teacher训练侧prompt评分1,701,248 tokens，probe prompt评分23,122，另列于原始costs。
- 固定目标缓存1,662,082,849 bytes；训练峰值GPU allocated=123,823,680,512 bytes。
- 训练与评测的combined_gpu_hours不是可再相加的独立成本。
- 以上为复用已有Teacher路线的本次Student增量成本，不是包含Teacher训练的端到端成本。

## 验证与限制

- 校验200条唯一训练轮次、200次参数更新、132条唯一最终输出、正确数15、最终生成tokens32633均一致。
- 日志未发现Traceback、CUDA OOM等致命错误，Slurm终态0:0。
- 原训练147项CPU测试通过；末端评测CPU测试22项通过、1项重复收尾幂等测试失败（恢复JSON造成CSV列顺序变化）。
  本次首次评测与收尾实际成功，没有触发重复写出分支；该已知恢复问题不应被描述为全部测试通过，本轮没有修改代码或重跑结果。
- development64有Teacher SFT44/RL20暴露。132题test在历史研究中已被评测，不能称为整个研究过程从未观察过的测试集。
- 新训练的优化/控制/development按ID隔离；控制题只影响调度而不反向传播。
- 单seed、没有同协议对照。不能将与旧版本结果的差异归因于MPC，不能据本轮扩大训练预算或调参。

## 交付与保存

便携包包含逐题最终输出、训练/开发曲线、成本账本、冻结选择、控制器记录、配置和源代码。
模型权重、48GB完整resume及固定目标大缓存仍保留在HiPerGator原run目录，不随便携包重复下载。
两条纯KD配置未提交，其他baseline与Teacher没有启动。后台监控在交付后停用。
