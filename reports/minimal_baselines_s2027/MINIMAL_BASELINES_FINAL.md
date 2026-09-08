# S0、GRPO 与绝对目标 KD 最小对照

状态：完整结果及终态成本已核验。

| 条件 | 训练轮数 | 实际更新 | 训练回答 | 训练 tokens | 选中轮次 | test | GPU 小时 |
|---|---:|---:|---:|---:|---:|---:|---:|
|原始 S0|0|0|0|0|预指定 S0|0/132|0.164444|
|纯 GRPO（保留诊断）|200|29|6400|257924|0|0/132|1.159722|
|绝对目标 KD+RL|200|200|6400|2351850|200|88/132|3.923889|
|相对目标 CAFD|200|199|6400|2212434|120|74/132|4.043611|
|历史 CAFD-MPC|200|200|6400|1553276|120|15/132|3.446111|

S0 在评测前固定为原始 checkpoint，0 轮训练、0 次更新，无 development 选模。其他训练条件使用 200 轮、800 组、6400 条回答预算；实际更新次数随合法 skip 如实单列，不将它们称为同更新数比较。
GRPO 关闭训练 KD，但保留 control/q5 与 Teacher 诊断流程。成本包含这些诊断开销，不能解释为移除全部 Teacher 流程的最精简 GRPO 成本。
绝对目标条件仅将训练 KD 目标改为 softmax((1-alpha) z_Tm + alpha z_Tnext)，保留相同 u 日程、全失败 retention 路由、全局 token 分母和 RL 项。相对 CAFD 使用永久 S0 的相对目标。

| 新条件相对已完成 relative CAFD | test 差（百分点） | 题级配对 bootstrap 95% 区间 |
|---|---:|---:|
|s0|-56.061|[-64.394, -47.727]|
|grpo|-56.061|[-64.394, -47.727]|
|absolute_kd|10.606|[1.515, 19.697]|

仅比较已经冻结并保存的 132 题 test；不会重新选模、插值或补做评测。各条件与 relative CAFD 按题 ID 配对，20,000 次 bootstrap，seed=2027。单训练 seed 的题级区间不覆盖训练随机性。
这些 test 已有历史评测；development 有 Teacher 训练暴露（44/64 用于 Teacher SFT，其余 20 用于 Teacher RL）。本次是单 seed 条件比较，不能称为多 seed 因果验证。
GPU 成本按各条件训练与 test 的 attempt job ID 并集去重，取 Slurm 预留终态时间；包括失败或恢复尝试。未知账目保持未知，共享作业不虚构条件分摊。已有 Teacher 轨迹训练成本不包含。Smoke 单列，不计入条件成本。

| 独立 smoke | GPU 小时 |
|---|---:|
|minimal_baselines_engineering_smoke|0.126389|

逐题型结果、development 曲线、KD/RL 覆盖、物理计数和作业终态见同目录 CSV/JSON；源码与冻结配置摘要见 source_proof.json。历史 MPC 的 round0 development 逐题输出和平均奖励未保存，保持缺失。
