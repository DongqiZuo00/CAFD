# 最小对照最终结果

全部三项新增实验及两项历史参考已完成核验。
数据集：DELTA Manufactoria-HAS，固定132题；Student：Ministral-3-3B-Instruct-2512-BF16；Teacher：同系列8B。
同一 seed2027、相同生成/验证规则，测试采用每题一次 greedy 输出、最多2048新token、完整通过率。

| 条件 | 测试完整通过 | 训练轮次 | 实际更新 | 选中轮次 | 正式GPU小时 |
|---|---:|---:|---:|---:|---:|
| 原始 S0 | 0/132 (0.00%) | 0 | 0 | 0 | 0.1644 |
| GRPO（开发集选回 S0） | 0/132 (0.00%) | 200 | 29 | 0 | 1.1597 |
| 旧 MPC | 15/132 (11.36%) | 200 | 200 | 120 | 3.4461 |
| CAFD 相对目标 KD+RL | 74/132 (56.06%) | 200 | 199 | 120 | 4.0436 |
| 绝对目标 KD+RL | 88/132 (66.67%) | 200 | 200 | 200 | 3.9239 |

绝对目标减相对目标：+10.61个百分点；按相同132题配对、20,000次bootstrap的95%区间为[+1.52, +19.70]个百分点。
仅绝对目标通过26题，仅相对目标通过12题。
这个区间只反映本次固定模型和题目的重采样差异；单个训练seed不足以确认跨训练运行的稳定优势。

GRPO的七个开发集检查点均为0/64，按预定的同分取最早规则选中round0原始S0。因此GRPO行的0/132是所选S0的冻结测试，不能解释为round200训练后模型的测试分数。独立S0与GRPO的132题保存输出逐项一致。
各训练条件都使用200轮、800次题目呈现、6400条回答；合法无损失轮次跳过更新，所以这是相同采样轮次的对照，实际优化器更新数不同。
GRPO只关闭训练KD，保留了历史Teacher/control/q5诊断和保存流程。表中按Slurm终态统计每个条件训练与测试作业的并集，包含该开销及失败重试；不能把它视为精简GRPO实现的最低成本。Teacher轨迹预训练成本不计入表中。
64道开发题均曾用于Teacher训练（44道SFT、20道RL），测试集也有历史评测记录。本结果是单seed、与既有协议匹配的对照；题目bootstrap不包含训练随机性，也不能当作整个训练流程未见数据上的独立确认。

## 题型结果

| 条件 | Count (31) | Ordered (48) | Substring (53) |
|---|---:|---:|---:|
| 原始 S0 | 0/31 | 0/48 | 0/53 |
| GRPO（开发集选回 S0） | 0/31 | 0/48 | 0/53 |
| 旧 MPC | 6/31 | 9/48 | 0/53 |
| CAFD 相对目标 KD+RL | 8/31 | 34/48 | 32/53 |
| 绝对目标 KD+RL | 4/31 | 43/48 | 41/53 |

新增三项正式实验合计5.2481 GPU小时；两项smoke合计0.1264 GPU小时，独立列账。
所有新增GPU作业串行使用1张B200、192G内存，工作文件均位于CAFD目录。

## 证据

详细表与统计：MINIMAL_BASELINES_FINAL.md、comparison.json、paired_bootstrap.json、conditions.csv、development.csv、coverage.csv、test_by_family.csv、allocations.csv。
训练、冻结输出、终态成本与幂等检查：各条件的training_audit/final_audit及final_comparison_audit.json。
源码、验证、修复记录及配置：validated_source_snapshot*.tar.gz、pipeline_manifest_history、各validation目录及protocol.json。
模型权重与resume保持原位置，见model_index.json；交付包仅含证据、代码、配置及日志，未复制权重。
