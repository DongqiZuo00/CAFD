# CAFD-MPC 实现记录（2026-09-06）

## 版本与范围

实现依据是用户提供的 CAFD_Method_MPC.pdf 及 CAFD_Method_MPC_ZH.md，
已原样留档于 /blue/du.j/jinjiaguo/CAFD/docs/methods/CAFD_MPC_20260906/。
本版本与旧 phase-reference、mixed-support 和 endpoint-anchored CAFD 分开。
旧结果不属于本算法的实验结果；旧代码、实验和恢复文件未改动。

实现前仅清理四个被禁用 Qwen 模型的可重新下载缓存，共29.33 GiB。
详细目录与字节数见 CAFD_MPC_CLEANUP_20260906.md。没有清理 Teacher trajectory、
Mistral 模型、实验 checkpoint、resume、raw rollout 或报告。

## 算法到代码的对应

| 规范 | 实现 |
|---|---|
| 永久初始 S0、相邻 Teacher 累计变化、连续 u | cafd/mpc_objective.py |
| 完整词表 FP32 forward KL 与精确流式反向 | cafd/mpc_objective.py |
| 同奖全失败 KD / 奖励变化 RL / 全成功跳过 | cafd/mpc_objective.py、mpc_train.py |
| 冻结行为模型、真实采样 log-prob、输出停止 | cafd/mpc_runtime.py |
| D+17 特征、遗忘 ridge、H=2 MPC、探索、缺失处理 | cafd/mpc_controller.py |
| 602梯度训练 / 12控制 / 64 development、ID隔离 | cafd/mpc_data.py |
| S0固定前缀与全词表 qM 缓存、每块新控制生成 | cafd/mpc_train.py |
| 数据路径/模型分片存在检查（无大文件hash） | cafd/mpc_data.py |
| Slurm、B200及内存强制校验 | cafd/mpc_allocation.py |
| 跨attempt物理成本、日志尾部归档、逻辑恢复 | cafd/mpc_ledger.py |

核心目标严格为：

    m = min(floor(u), M-1)
    alpha = u-m
    q = stopgrad(softmax_FP32(
        z_S0 + (1-alpha)*z_Tm + alpha*z_T(m+1) - z_T0
    ))

S0全程不刷新。不混入 Teacher replay，不使用 endpoint anchor mixing。
同一目标中的所有模型在相同 Student token prefix 上评分。
同一个 Teacher 路径的系数会合并，零系数模型不加载或评分。

## 训练伪代码

    freeze(S0, real_teacher_route)
    behavior = independent_frozen_copy(Student)
    split optimization / control / development / test by problem ID
    fixed_prefixes = generate_once(S0, control)
    cache q_M on every valid fixed-prefix completion position, FP32 full vocabulary
    s_0 = fresh_control_reward_and_success(Student) + fixed_endpoint_KL(Student)
    controller = MPC(s_0, initial_u=min(delta, M))

    for each full block, then optional final partial block:
        action = initialization / exploration / MPC_first_action
        u = controller's shared coordinate, constant throughout this block

        for each rollout-update round:
            copy Student weights into independent behavior model
            sample R current-behavior completions per optimization prompt
            record actual sampled-token behavior log probabilities
            score fixed hierarchical reward and full-pass verifier

            D = groups with identical reward and no successful response
            R = groups with different rewards
            Z = all valid completion tokens, including zero-contribution groups
            skip = groups whose rewards are all 1

            if Z == 0 or both contributing groups are empty:
                skip entire optimizer step, including momentum/weight decay
            else:
                zero Student gradients
                for each completion in D:
                    evaluate frozen S0 and nonzero Teacher sources
                    backward exact_full_vocab_KL(q_u || Student) / Z
                for each completion in R:
                    backward clipped_group_relative_policy_loss / Z
                clip Student gradient norm
                FP32-master AdamW step

            count the round even when optimizer step was skipped
            count real generation, verification and scoring costs

        observe fresh control responses plus fixed-prefix q_M fitting error
        append only valid complete-block transitions
        refit weighted ridge; record controller state and costs
        save formal block-boundary resume
        development-only selection at declared milestones, earlier checkpoint wins ties

    freeze development-selected checkpoint
    do not evaluate frozen test in this training runner

## 数值与采样实现

- 学生、参考和 Teacher 使用相同 token-ID 映射，启动时核对词表及 chat renderer。
- 模型主干 BF16；目标构造、LM-head投影/归一化及 exact KL 使用 FP32，
  显式禁用相关算子的 autocast。
- 全词表保留131072个token；没有top-k KL、反向KL、词表重排或目标裁剪。
- token分块前向和反向重新计算完整softmax，返回精确 p-q 梯度，不保存 L×V 激活。
- 训练采样 temperature=1、top_p=1、top_k=0；无概率截断。
- 从真实generate过程逐步收集被采样token的概率，仅保留一块B×V概率缓冲。
- closing fence 是外部停止条件，不强行生成EOS；长度上限不捏造EOS。
- 首completion与实际生成EOS纳入mask；prompt、padding、EOS后的内容排除。
- optimizer使用FP32 master/moments；恢复时覆盖通用load_state_dict可能造成的BF16舍入。
- 单GPU入口拒绝默默启动多进程DDP，因而全局token分母明确对应整个batch。

## 数据与Teacher来源

Benchmark保持DELTA Manufactoria-HAS；Student为Ministral-3-3B-Instruct-2512-BF16，
Teacher为Ministral-3-8B-Instruct-2512-BF16。Qwen不参与新版本。

真实Teacher路线复用：

    TBase -> SFT125 -> RL40 -> RL60 -> RL80 -> RL100

这是明确记录的SFT+RL acquisition trajectory，不称为纯RL路线。
原始S0初始化不是capacity SFT checkpoint。

从原fit614按seed2027每family留出4题控制，共12题，梯度训练余602题。
原selection64作为本版本development。Teacher历史SFT见过其中44题，
RL见过另外20题；它只在Student侧held-out，不是全pipeline未见数据。
原development64作为unused confirmation，不进入本控制器。

测试集只流式提取132个问题ID用于交叉核对，不使用题目/答案训练或评测。
旧实验历史已经评测过该test；本版本不宣称它从未被研究过程访问。
新的控制器不读取development/test的分数。任何最终泛化主张仍需遵守这一历史限制。

## 配置

正式入口预置：
seed2027，200 rollout-update rounds，4 prompts×8 rollouts，
generation cap2048，K10/H2，delta=min(.5,M/8)，六个初始化block，
epsilon=.1，gamma=.95，ridge=1，RL系数1，优势epsilon=1e-6，
对称ratio clip=.2，lr1e-6 constant，FP32 AdamW，weight decay0，grad clip1。
development milestones为0/10/40/80/120/160/200，平分选较早checkpoint。
实际optimizer updates单独计数，不要求每轮都更新。
控制预算允许结束于u<M，没有隐藏的强制终点推进或额外gate。

smoke为独立工程配置：8 rounds、2×2、64-token cap、K1，
每family控制题1个，不进行development/test评测。
它只检验真实Mistral生成、反传、固定probe与MPC模块联通，不能报告为准确率实验。

## 恢复与成本

正式运行在初始化后和每个block边界保存完整resume。
只支持已提交block边界恢复；不将pending controller状态当成已完成轮次。
保存Student、FP32 optimizer、RNG、controller、best selection和逻辑日志偏移。
未提交尾日志先归档再回滚；未提交的checkpoint单独保留，不静默复用。

physical_costs.jsonl只追加，不参与逻辑回滚。任何恢复均保守标记
lower_bound_after_interruption：不能假定被中断算子的部分token工作已完全计量。
GPU-hours尽可能来自所有attempt的Slurm allocated elapsed；
Slurm查询不可用则标记worker_elapsed_lower_bound，不伪造0。

报告生成tokens、Teacher/S0评分tokens、prompt处理量、control成本、
verifier calls、CPU控制时间、模型加载、缓存空间、峰值显存、实际更新和耗时。
Teacher既有路线构建成本与本次Student增量成本分开；本次没有重新训练Teacher。

## 验证记录与入口

CPU测试首轮124项通过；补充成本台账后，Slurm节点全套147项通过（28.43秒）。
测试包含dense参考梯度/单步参数一致、真实tiny-Transformer采样概率回放、
全成功组不动动量、固定终点cache、控制器递推/RNG恢复、数据隔离与资源边界。

单卡集成验证job：41221171，1×B200、192G、最长1小时。
终态COMPLETED、exit 0:0，节点c0901a-s15，Slurm最终耗时309秒（5分9秒）。
8个rollout rounds与8个optimizer updates完成，实际执行1次多动作MPC预测决策，
末尾u=2.5；另外一次初始化后的动作来自规定的随机探索。
峰值GPU allocated显存123615446528 bytes，固定FP32终点cache约96 MiB。
这只是64-token工程smoke，未提交正式200轮作业，也未评测development/test。
真实结果读取：

    artifacts/cafd/experiments/mistral_cafd_mpc_v1_smoke/result.json
    runs/cafd/experiments/mistral_cafd_mpc_v1_smoke/complete.json
    state/cafd/experiments/mistral_cafd_mpc_v1_smoke/state.json

只读预检查：
    .venv/bin/python -m cafd.mpc_train --prepare-only

正式脚本（已实现但未提交）：
    scripts/cafd/slurm/mpc_v1_formal.sbatch

本次没有实现/重跑全部baseline和机制消融，也没有新版本的正式效果、
蒸馏效率提升或SOTA结论。算法一致性测试不等于任务能力获得。
