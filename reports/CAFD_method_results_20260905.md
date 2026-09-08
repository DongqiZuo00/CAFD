# CAFD 方法实现与实验结果

研究技术记录  2026年9月5日

## 1 概要与结论

我们研究如何将 Teacher 训练过程中的分布变化迁移给较小的 Student。CAFD 不直接把阶段 Teacher 的概率分布作为唯一目标，而是把相邻 Teacher checkpoint 在同一因果前缀上的 logits 差，施加到阶段起点 Student 的冻结分布上，再用全词表正向 KL 训练当前 Student。当前主方法采用 Teacher 与 Student 混合生成的训练前缀。本文中的 CAFD 均指该主方法；纯 Teacher replay 仅作为历史实现记录，不作为另一项核心贡献。

目前已实现跨模型规模的增量目标构造、独立冻结的阶段参考、FP32 全词表 KL、混合前缀训练、完整 checkpoint 与恢复机制。当前 backbone 为 Ministral 3B Student 和 Ministral 8B Teacher，任务为 DELTA Manufactoria-HAS 程序生成。我们拥有可运行的方法和实验，但尚无证据支持 SOTA 或整体蒸馏效率领先。

同一 Mistral selection split 上，CAFD 在 step40 和 step80 分别得到 20/64 和 32/64，高于同期 Progressive GKD 的 14/64 和 26/64；到 step200，CAFD 为38/64，低于 Progressive GKD 的55/64和 Endpoint OPD 的49/64。80步残差锚定修正版得到31/64，未通过预设 pilot 门槛。早期学习优势只对特定 baseline 和预算区间成立，不能扩展为整条曲线或最终能力优势。

本文以远端仓库 /blue/du.j/jinjiaguo/CAFD 的代码、配置和落盘结果为依据。代码映射与证据路径见第12节。所有数值均为本项目实验结果，不是模型厂商公布的 benchmark 成绩。

## 2 问题定义与实验对象

### 2.1 我们要迁移什么

Endpoint OPD 学习最终 Teacher 已有的分布；Progressive GKD 随阶段更换 Teacher，学习各阶段的绝对分布。CAFD 要迁移的是相邻 Teacher 之间的变化：在相同前缀上，Teacher 刚刚把哪些候选 token 的相对概率提高或降低。

这是一种可检验的建模假设，而不是“logits 差就是能力”的定理。差分里也可能包含格式适应、概率校准、遗忘或不相关变化。方法的实际价值需要通过准确率随步数、生成 tokens、Teacher 评分 tokens 和总计算成本的变化来验证。

### 2.2 模型和白盒要求

| 角色 | 当前模型 | 规模 |
|---|---|---|
| Student | mistralai/Ministral-3-3B-Instruct-2512-BF16 | 3B |
| Teacher | mistralai/Ministral-3-8B-Instruct-2512-BF16 | 8B |

Student revision 为 b6d637bef2393152b3da2b2fde72eecdee30557e；Teacher revision 为 f6fae9795746f63c9be8344932f01275f3c63734。模型采用 BF16 前向与全参数训练，优化器另保留 FP32 master weights。

CAFD 是白盒蒸馏。训练时必须访问两份 Teacher 的全词表 logits，以及 Student 阶段参考的 logits。当前运行时检查双方 tokenizer 的131072个 token 映射及特殊 token 一致；不同隐藏层宽度无需对齐，但词表维度和 token 语义必须一致。没有词表投影、top-k 近似或黑盒 API 替代路径。

### 2.3 任务与输出合同

Benchmark 为 DELTA Manufactoria-HAS。模型读入自然语言任务，输出 Manufactoria DSL 程序，由 verifier 解析并执行题目测试。项目数据包含 contains_count、contains_ordered、contains_substring 三类问题。

系统提示固定为 Return exactly one Manufactoria DSL code block and nothing else。输入通过 ministral3_instruct_fixed_system renderer 构造 Instruct 模板。生成上限为2048 tokens，训练采样 temperature=1.0、top_p=0.95；development 与 test 使用单次贪心生成。ClosingFenceEOSProcessor 在 closing fence 后强制下一个 token 为 EOS，随后裁掉 padding 与 EOS 后的内容。每题只有全部测试通过才计为正确，最终指标为 full-pass/pass@1；部分测试奖励只用于 RLVR 训练，不代替准确率。

### 2.4 数据划分的边界

| 数据部分 | 题数 | 用途 |
|---|---|---|
| 原 train | 678 | 早期训练与 Teacher 轨迹来源 |
| 后续 fit | 614 | 当前 Student 参数更新 |
| 后续 selection | 64 | 当前 Student milestone 选择 |
| 原 development 后称 confirmation | 64 | 选择后复核及后续观察 |
| frozen test | 132 | 历史正式 held-out 评测 |

fit 与 selection 是原678题 train 的不重叠划分，拆分 seed=12027；selection 的三类题数分别为14、25、25。Student 训练 seed=2027。后续复用的 Teacher 来自拆分前的训练流程，selection中44/64题直接出现在其SFT监督记录里，因此不能把selection描述为“Teacher和Student都未见过”的独立数据。原development也曾参与Teacher选择，而且之后已被多次观察；confirmation不是从未使用过的最终盲测。

当前 v10、v11、v12、v14、v15 没有新的 frozen-test 成绩。历史 v9 test 成绩不能直接代表当前主方法。观察过的 test 也不能在后续反复调参中继续充当完全独立的证据。

## 3 CAFD 的数学定义

### 3.1 相同前缀上的有序差分

设阶段 m 使用相邻 Teacher T_m 和 T_(m+1)。h 表示由 prompt 与 completion 前缀组成的同一 token 序列，v 表示词表元素。z_T(h) 是模型在 h 后预测下一 token 的 logits。S_ref,m 是阶段开始时对当前 Student 的独立冻结复制。

```math
Delta_m(h) = z_T(m+1)(h) - z_T(m)(h)
u_m(h) = z_Sref,m(h) + Delta_m(h)
q_m(.|h) = stopgrad(softmax_FP32(u_m(h)))
```

Teacher 两端与 Student 参考必须看到完全相同的 input_ids、attention_mask 与预测位置。不能分别对各模型自由生成的不同前缀做 logits 相减。冻结参考也不是 Student Base 永久不变，而是在每个新阶段重新复制一次，此后阶段内不刷新。

当前 gamma=1、目标 temperature=1。代码支持显式缩放参数，但主实验没有使用额外差分裁剪、概率比裁剪或词表筛选。

### 3.2 概率比解释

在温度为1且 token 对齐的条件下，softmax 对位置相关的加性常数不敏感，因此同一个目标也可写为：

```math
q_m(v|h) ∝ p_Sref,m(v|h) * p_T(m+1)(v|h) / p_T(m)(v|h)
```

这个等价关系说明：Teacher 将某个候选相对其他候选提高多少，CAFD 尝试把该相对变化转移到 Student 阶段起点分布中。实现仍在 FP32 logits 与 log-softmax 空间完成，不直接对极小概率做除法。

该结构不能保证增量在 Student 的错误前缀上同样有效，也不能保证两个不同架构的 logits 残差有相同的功能意义。另一个必要性质是：如果阶段起点 Student 与前一 Teacher 分布完全一致，CAFD 目标就等于后一 Teacher 分布；CAFD 与 Progressive 的差异来自这项参考残差，而非一种独立保证更优的监督。

### 3.3 精确正向 KL

令 M 为有效 completion 预测位置集合，N 为该 update 的 completion token 数。目标为：

```math
L = (1/N) * sum_(h in M) sum_(v in V)
        q_m(v|h) * (log q_m(v|h) - log p_Student(v|h))
```

KL 的方向是 KL(q || p_Student)，不是反向 KL。Teacher、阶段参考、q 及 log q 均不接收梯度；当前 Student 的 log-softmax 保留梯度，包括全词表归一化项。温度为1时，单位置 Student logits 梯度为 p_Student-q，再乘归一化系数。

mask 使用因果移位：logits 的位置 t 预测 input_ids 的 t+1。包括第一个 completion token 和第一个 EOS，排除 prompt targets、padding 和首个 EOS 后内容。训练生成已由 trim_completion 截断，因此 rollout completion 长度之和应等于有效 mask 数；这一不变量是全 update token 归一化成立的前提。

## 4 真实 Teacher acquisition trajectory

当前route有六个节点、五个相邻迁移阶段。它是raw Instruct到Verified-Solution SFT再到RLVR的真实训练轨迹，不是六份纯RL checkpoint。Teacher把原678题按seed2027分成512题SFT与166题RLVR，两池不重叠且并集覆盖原train；这与后来Student的614加64划分不同。

| 路由节点 | 来源 | 在 Student 中的作用 |
|---|---|---|
| R0 | 原始 Instruct TBase | 阶段0的差分起点 |
| R1 | Verified-Solution SFT125 | 阶段0终点 |
| R2 | RL40 | 阶段1终点 |
| R3 | RL60 | 阶段2终点 |
| R4 | RL80 | 阶段3终点 |
| R5 | RL100 | 阶段4终点及 Endpoint 目标 |

Teacher 的原 development 曲线为：原始模型0/64；SFT25为1/64，SFT50为5/64，SFT75为30/64，SFT100为46/64，SFT125为60/64；随后 RL50为62/64，RL100为63/64。第一段轨迹跨越最大的能力变化，后面的 RL 阶段变化较小。不能据此把五个阶段视为等量能力增量。

route.json 状态为 frozen。CAFD 使用 R_m 与 R_(m+1)，Progressive 使用 R_(m+1)，Endpoint 始终使用 R5。Teacher 轨迹构建耗时属于共享上游成本，应与单个 Student 的训练成本分开统计；端到端效率论证还需将其纳入。

## 5 我们如何实现主方法

### 5.1 混合前缀与训练预算

每个 update 取4个 prompt，每个 prompt 生成8条 completion，共32 slots。阶段长度为40 updates，总共200 updates；五阶段的 Teacher 条数为 [6,4,4,2,2]，Student 条数为 [2,4,4,6,6]。

| 阶段 | updates | Teacher来源 | 每题Teacher条数 | 每题Student条数 |
|---|---|---|---|---|
| 0 | 1至40 | R1 | 6 | 2 |
| 1 | 41至80 | R2 | 4 | 4 |
| 2 | 81至120 | R3 | 4 | 4 |
| 3 | 121至160 | R4 | 2 | 6 |
| 4 | 161至200 | R5 | 2 | 6 |

Teacher completion 提供较成熟程序的前缀；Student completion 让监督覆盖自身生成分布。两类 rollout 都进入相同 CAFD 目标和 KL，不因答案失败而自动过滤；rollout token 是被评分的前缀来源，不是 token-level one-hot 蒸馏标签。

共享 prompt stream、seed、slots 和采样配置不等于共享实际生成文本。Student 参数不同后，Student rollout 必然可能不同。Endpoint 和 Progressive 对照使用相同的混合来源调度；其中 Endpoint 的 Teacher rollout 仍来自当前阶段 R_(m+1)，但其监督目标固定为 R5。Direct RLVR 则保持32条 Student on-policy rollout，不使用 Teacher 生成。

### 5.2 阶段参考的冻结与恢复

阶段开始调用 clone_frozen_model，deepcopy 当前 Student，设为 eval，并 requires_grad_(False)。实现逐参数检查 data_ptr，防止参考与可训练 Student 共享存储。每40步结束检查参考身份及无梯度状态，阶段内不得刷新参考。

当前 Mistral 训练使用 NoHashPhaseController，避免反复扫描全模型计算哈希。它保留阶段编号、起始 update、reference_id 与原子保存的参考权重；旧字段名仍可能含 hash，但当前检查是身份和冻结状态，不应写成每步全权重加密校验。

resume 包含 Student、FP32 optimizer 状态、scheduler、RNG、global update、phase-local update、prompt cursor 和冻结参考定位信息。恢复阶段时加载当时保存的参考，不能用恢复后的当前 Student 重新复制替代。

### 5.3 数值计算与内存控制

HiddenCausalLM 先取得各模型隐藏状态，选出有效 completion positions，再用各自 LM head 投影到共享完整词表。Student、phase_ref、前后 Teacher 的隐藏宽度可以不同。head 权重与投影计算转为 FP32，target softmax、log-softmax 与 KL 求和均为 FP32。

每个 sequence microbatch 为4条。有效位置按64个 positions 分块，但每个块覆盖131072维完整词表。这是位置分块，不是词表 top-k，也不是在词表碎片上分别归一化。当前代码将一个 microbatch 内各块 loss_sum 累加后再 backward；因此分块限制单次投影张量大小，但不能宣称计算图和所有中间量都在每块后立即释放。

一个 update 中所有 microbatch 梯度累积后，执行全局梯度范数裁剪1.0，再更新 Student。优化器为 FP32AdamW：lr=1e-6，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，constant scheduler。master weights 与一阶、二阶矩以 FP32 保存，更新后拷回 BF16 参数，以避免小步长更新在 BF16 中反复舍入丢失。没有 LoRA，也没有将 Teacher 放入 optimizer。

出现非有限 loss 时保存首个问题 batch 并报错，不用静默 clamp 掩盖。主方法路径没有额外 RL reward loss 与 KL 相加；verifier 在 CAFD 中主要用于 checkpoint 评测和诊断。

### 5.4 Checkpoint 选择

在 steps [0,10,40,80,120,160,200] 保存和评测。完整预算结束后取 selection accuracy 最大的 checkpoint；并列时选较早 step。选择step160不代表只训练160步，CAFD v10实际完成200步。模型权重与完整 optimizer resume 分别保存；后者主要对齐40步阶段边界。

v10含step10格式存活检查，例如64题中至少48题符合输出合同、EOS正常，以及非零 loss/gradient；它不是52/64能力门槛。v14另有独立 pilot 规则，见第9节。两者不能混称为同一 gate。

## 6 CAFD 训练伪代码

以下伪代码对应当前 v10 主方法。T[0..5] 为已冻结真实轨迹，S 从原始 Ministral Student S0 开始，而非从 capacity SFT checkpoint 开始。

```python
T = load_frozen_teacher_route(R0, R1, R2, R3, R4, R5)
S = load_student(S0, trainable=True)
opt = FP32AdamW(S, lr=1e-6, betas=(0.9, 0.999))
teacher_counts = [6, 4, 4, 2, 2]
stream = frozen_prompt_stream(fit, seed=2027)

for m in range(5):
    Sref = deepcopy(S).eval().requires_grad_(False)
    assert_independent_storage(S, Sref)
    save_phase_reference(Sref, phase=m)

    for j in range(40):
        step = 40*m + j + 1
        prompts = stream[step-1]       # 4 prompts
        B = []
        for x in prompts:
            B += sample(T[m+1], x, n=teacher_counts[m])
            B += sample(S, x, n=8-teacher_counts[m])
        persist_raw_rollouts(B, source_tags=True)
        N = sum(len(y.completion_ids) for y in B)
        opt.zero_grad()

        for microbatch in chunks(B, 4):
            ids, attention, M = causal_completion_batch(microbatch)
            hs = S.hidden(ids, attention)       # with gradient
            with no_grad():
                hr = Sref.hidden(ids, attention)
                hp = T[m].hidden(ids, attention)
                hn = T[m+1].hidden(ids, attention)
            loss_sum = 0
            for positions in chunks(valid_positions(M), 64):
                zs = full_vocab_head_FP32(S, hs, positions)
                with no_grad():
                    zr = full_vocab_head_FP32(Sref, hr, positions)
                    zp = full_vocab_head_FP32(T[m], hp, positions)
                    zn = full_vocab_head_FP32(T[m+1], hn, positions)
                    logq = log_softmax(zr + zn - zp, dim=vocab)
                    q = exp(logq).detach()
                logp = log_softmax(zs, dim=vocab)
                loss_sum += sum(q * (logq - logp))
            backward(loss_sum / N)

        clip_grad_norm(S, 1.0)
        opt.step()                    # FP32 masters -> BF16 weights
        if step in [10, 40, 80, 120, 160, 200]:
            save_checkpoint(S, step)
            evaluate_greedy(S, selection)
        if step % 40 == 0:
            save_resume(S, opt, RNG, Sref, phase=m, cursor=step*4)

selected = argmax(accuracy, tie_break="earliest_step")
freeze_selection(selected)
# v10 workflow: confirmation only; no new frozen-test evaluation
```

伪代码中的 sample 均使用固定 renderer、temperature=1、top_p=0.95、max_new_tokens=2048和closing-fence停止规则。所有目标模型在相同微批次位置上前向；模型输出不同，并不改变这些位置的 token 对齐。

## 7 已实现的比较方法

| 方法 | 前缀来源 | 每位置目标 | Teacher评分次数 |
|---|---|---|---|
| Direct RLVR | 当前Student | verifier奖励的组相对优势 | 0 |
| Endpoint OPD | 匹配混合调度 | 最终R5绝对分布 | 1 |
| Progressive GKD | 匹配混合调度 | 当前R_(m+1)绝对分布 | 1 |
| CAFD 主方法 | 匹配混合调度 | Sref加相邻Teacher差分 | 2 |
| Residual Anchored v14 | 匹配混合调度 | 阶段Teacher与CAFD概率混合 | 2 |

Endpoint 与 Progressive 使用相同的 exact full-vocabulary forward KL 实现，区别是 target checkpoint。这里的名称指项目内实现的 baseline，不代表复现某篇论文全部工程细节或厂商官方评分。

Direct RLVR 对每题8个 reward 做组内标准化，advantage=(reward-mean)/(std+1e-4)，采用带ratio clip的策略损失。层级奖励为：格式无效0；有单代码块但解析失败0.05；解析通过但零测试通过0.10；部分通过0.10+0.90×pass_rate；全部通过1.0。最终 full-pass 指标始终是0或1。若组内所有奖励相同，优势为零，训练就可能没有有效更新；这需要结合实际日志判断，不能仅由最终0分倒推出原因。

Ministral Student已有Verified-Solution SFT capacity control：原development在SFT175达到57/64=89.0625%，超过52/64门槛。其曲线依次为step0/25/50/75/100/125/150/175对应0/0/10/40/30/48/48/57。该结果由v6曲线和v7保留清单交叉确认；原capacity gate文件已不在当前目录，不能声称旧checkpoint仍可恢复。

该监督控制说明Student具备表达目标能力的可行性，但CAFD从S0开始，不从SFT175继续。它使用旧train/development，不是当前614题fit与64题selection下重新完成的matched Oracle baseline。旧Qwen capacity也单独作为历史记录，不混入当前Mistral横向比较。

## 8 当前同设置的实验结果

### 8.1 Mistral selection 曲线

下表各分数分母均为64，使用同一 fit/selection、模型对与32-slot预算。Direct 的200步尚未完成，不能填成正式0/64。

| 方法 | 0 | 10 | 40 | 80 | 120 | 160 | 200 |
|---|---|---|---|---|---|---|---|
| CAFD | 0 | 2 | 20 | 32 | 37 | 38 | 38 |
| Progressive GKD | 0 | 1 | 14 | 26 | 38 | 49 | 55 |
| Endpoint OPD | 0 | 1 | 19 | 46 | 39 | 43 | 49 |
| Direct RLVR | 0 | 0 | 0 | 0 | 0 | 0 | 未完成 |

CAFD：selected step160，38/64=59.375%，actual updates=200。Progressive：selected step200，55/64=85.9375%，actual updates=200。Endpoint：selected step200，49/64=76.5625%，actual updates=200。Direct state 停在180/200，其最新已评测 milestone 是160步0/64，不能把180步状态当作已评测结果。

CAFD 在40和80步比 Progressive 多答对6题，但从120步起被反超；相对 Endpoint，仅40步领先1题，80步已落后14题。以0至200步的分段线性梯形积分除以200得到平均 accuracy AUC：CAFD=0.451563，Progressive=0.478906，Endpoint=0.530078。这个 AUC 只按updates归一化，不是GPU成本效率；Direct与80步pilot不纳入200步AUC。

### 8.2 当前主方法的 confirmation

CAFD v10 selected step160 在 confirmation 得到33/64=51.5625%，比selection少5题。分题型：contains_count为1/17，contains_ordered为15/24，contains_substring为17/23。64题中61题格式及解析达到要求，28题部分通过，3题格式无效且达到token上限。

这说明该 checkpoint 的主要剩余问题不只是输出合同：count族的 full-pass 尤其低，且许多程序可执行但不能通过全部测试。不过这是已被观察的数据，只能用于描述错误，不应再作为未经调参影响的最终泛化证明。本文不把33/64与其他方法的selection分数比较。

### 8.3 Tokens 与耗时

| 方法 | 实际生成tokens | 实际Teacher评分tokens | 训练计时记录 GPU h |
|---|---|---|---|
| CAFD v10 | 2,433,625 | 4,867,250 | 3.9971 |
| Progressive v11 | 2,852,135 | 2,852,135 | 1.4545 仅末次恢复段 |
| Endpoint v12 | 2,449,130 | 2,449,130 | 3.8813 |
| Direct v12 | 7,521,840 状态累计 | 0 | 未形成完整选择账本 |

Progressive 的逻辑生成tokens为2,486,091，另有366,044个已生成但未计入最终逻辑轨迹的tokens；其selected.json标明resume_start_update=120、prior_gpu_hours=0。因此1.4545小时不能当作完整200步耗时，不能据此计算完整训练加速比。

Teacher-scored tokens 是completion长度乘Teacher评分份数，未包括prompt prefill、phase_ref前向、Teacher生成rollout的实际计算或轨迹训练。CAFD每个位置两份Teacher评分，另需Student参考前向；相同updates与slots不是相同FLOPs或GPU-hours。当前结果不支持宣称“大幅提高整体蒸馏效率”。

### 8.4 历史 frozen test 与归属纠正

Ministral 历史 v9 纯Teacher-replay CAFD 完成200步，development在step80为36/64=56.25%，冻结该checkpoint后test为63/132=47.7273%。训练生成2,465,328 tokens、Teacher评分4,930,656 tokens，训练2.8658 GPU h，Slurm耗时3:13:17。这是旧版支持分布与旧训练划分的成绩，不是当前混合前缀主方法的test成绩。

此前经常引用的 Direct 0/132、Endpoint 48/132、Progressive 65/132，实际来自 chat_hier_v4b_sft25。该配置的 Teacher 是 Qwen3-4B-Instruct-2507，Student 是 Qwen3-1.7B。它们必须与Ministral结果分开；不能称为Ministral官方baseline。

| 旧Qwen实验条件 | test正确数 | test准确率 | updates |
|---|---|---|---|
| Teacher Base 即SFT warmstart | 97/132 | 73.4848% | RL 0 |
| Teacher RLVR | 120/132 | 90.9091% | RL 100 |
| Student Base | 0/132 | 0% | 0 |
| Capacity Control | 119/132 | 90.1515% | 100 |
| Direct RLVR | 0/132 | 0% | 200 |
| Endpoint OPD | 48/132 | 36.3636% | 200 |
| Progressive GKD | 65/132 | 49.2424% | 200 |
| 旧CAFD | 0/132 | 0% | 200 |

此表只保留历史事实。当前项目已停用Qwen；不能将旧Qwen容量通过54/64、旧Qwen frozen-test对照、Ministral v9 test与当前Ministral selection合并成一个正式矩阵。

## 9 残差锚定修正版的实现与结果

v14 保留主方法的模型、数据、prompt流、32slots、阶段长度、学习率与exact KL，只替换目标。对每个预测位置，在词表维度做中心化RMS：

```math
d = RMS(center(z_next - z_previous))
r = RMS(center(z_phase_ref - z_previous))
alpha = d / (d+r)                 # zero/zero -> 0
q_mix = (1-alpha)*softmax_FP32(z_next) + alpha*q_CAFD
```

这是概率分布的混合，不是logits线性插值。实现用logaddexp计算混合log概率，alpha与两项target全部detach。阶段参考仍在阶段内冻结。

pilot只跑80步，事前门槛为step40≥20、step80≥33且step80>step40；step80≥40仅是理想目标。结果为step10=3/64、step40=11/64、step80=31/64=48.4375%，状态PILOT_NOT_PASSED，没有延长至200步。

alpha由update1约0.5115降至update80约0.0449。末步Teacher delta RMS约0.0574，reference residual RMS约1.2267。末期目标约95.5%的混合权重来自当前阶段Teacher，但混合权重不等于梯度贡献比例。作业41175009在c1010a-s5正常退出0:0，用时1:41:53；训练记录1.6209 GPU h。实际生成975,059 tokens，Teacher评分1,950,118 tokens；selected checkpoint为step80。

第一阶段11/64低于原CAFD的20/64，而末期31对32仅差一题。该pilot不支持此修正提升，但单seed小样本不能证明统计显著退化，更不能仅由alpha变化确定因果。

## 10 方向诊断与尚未解决的问题

### 10.1 v15 怎样做诊断

作业41214438已完成，节点c0901a-s15，Slurm耗时4分7秒，exit code=0:0；前向诊断主体208.36秒，使用1张B200、96G。没有训练、重新生成、selection、confirmation或frozen-test读取。

每个前两阶段，从已有Student rollout日志中取最先出现的8个不同fit题目，复用其prompt tokens。对每题构造真实包装的canonical solution，经verifier确认full-pass及无截断；另保留原Student completion。每条最多均匀抽取128个预测位置，按完整词表计算不同target的NLL，先对题内位置平均再对题目等权平均。

phase0以S0为reference；phase1分别使用原CAFD step40与v14 step40作为reference。Teacher前后端与最终endpoint始终在相同输入序列上评分。每个比较都是逐前缀配对的，但Student自己生成的token不当作gold标签。

### 10.2 Gold前缀上的结果

下表是已验证gold token的抽样平均NLL，单位为nats/token，越低代表这些指定token的条件概率越高，并不等同于自由生成的full-pass。

| 阶段与参考 | reference NLL | 原CAFD目标 | v14混合目标 | 当前Teacher | 最终Teacher |
|---|---|---|---|---|---|
| phase0 S0 | 0.297668 | 0.003031 | 0.001408 | 0.000164 | 0.000048 |
| phase1 原step40 | 0.006094 | 0.003451 | 0.000211 | 0.000048 | 0.000004 |
| phase1 v14 step40 | 0.010056 | 0.004392 | 0.000296 | 0.000048 | 0.000004 |

phase0八题的原CAFD目标均优于reference；phase1两种reference下均有7/8题改善。三组中混合target在8/8题上均比原CAFD target的gold NLL更低。这与“混合修正必然破坏gold增量方向”的简单解释不一致。

但训练后的selection结果更差，因此gold前缀上的概率拟合不是充分条件。样本来自Teacher训练池；phase0的7/8题和phase1的5/8题直接出现在Teacher SFT记录中，其余属于RLVR池。每阶段仅8题，不能推断未见题或偏离gold后的恢复能力。

### 10.3 Student前缀的解释限制

phase0原Student token在reference下NLL约0.4908，在原CAFD目标下约3.7393，在混合目标下约3.5374。这只表示目标不再偏好这些旧Student token，不能称为“损害正确答案”；若这些token本来错误，降低概率可能是合理的。

phase1原reference对应的Student token NLL从0.01018变为原CAFD的0.01332、混合的0.02383，同样不能直接转换为能力变化。当前诊断没有为偏离canonical的Student前缀建立可验证的后续正确动作集，也没有测量完整闭环恢复。因此它尚未区分“增量在错误前缀上无用”与“Student未学会执行有用增量”。

脚本记录wrapper_proxy与body_or_other，但这只是字符串启发式，不是语义关键token标注。不能据此声称已证明格式学习与语义学习的精确分解。此前v13仅抽样每阶段两条Teacher rollout，更不能作为完整归因证据。

### 10.4 下一候选尚未实现为训练结果

曾提出用 q_beta=softmax(z_phase_ref+beta×Delta_T) 控制迁移幅度，并用概率分布距离约束beta。它目前是候选设计，不是已验证方法；没有确定预算规则，也没有正式训练结果。v15并不构成自动启动该版本的充分依据。

## 11 我们现在可以怎样表述贡献

可以陈述：我们实现了一种利用真实Teacher训练轨迹、把相邻logits变化迁移到冻结Student阶段参考的白盒蒸馏方法，并完成了Ministral 8B到3B的程序生成实验。其实现包含混合前缀、完整词表KL和可恢复阶段参考。在固定selection上，CAFD相对Progressive在40至80步有早期准确率优势，但最终分数与200步AUC落后。

不应陈述：当前方法已达SOTA、显著提高最终泛化、整体蒸馏成本更低、已经因果解决Teacher-replay分布偏移，或已发现通用跨模型能力向量。也不应把历史56.25% development解释成56.25% frozen test。

下一轮有意义的验证应预先冻结对照、目标预算与评价规则；除updates外，同时统计Teacher评分、参考模型前向、轨迹构建与失败重试成本。核心可检验目标仍然是：在可比计算成本下，Student是否更早获得可验证能力，并在最终独立评测中保留这些能力。

## 12 实现与证据索引

所有下述相对路径均以 /blue/du.j/jinjiaguo/CAFD 为根。模型路径较长时用run_id加子路径定位，避免将旧版文件误当成新结果。

### 12.1 核心代码

- cafd/relative_target.py：relative_target_logits与detached FP32目标。
- cafd/exact_forward_kl.py：causal completion mask、全词表投影、KL位置分块。
- cafd/train_student.py：32-slot更新、microbatch归一化、阶段切换、checkpoint选择。
- cafd/optimizer.py：FP32AdamW master weights与状态。
- cafd/mistral_cafd_generalization_v10.py：混合前缀与固定fit/selection。
- cafd/mistral_cafd_support_full.py：NoHashPhaseController和参考持久化。
- cafd/mistral_runtime.py、cafd/prompting.py：固定模型、tokenizer与renderer。
- cafd/training_common.py、cafd/verifier.py：采样、EOS合同、RLVR和官方执行评分封装。
- cafd/mistral_progressive_observation_v11.py、cafd/mistral_baselines_v12.py：当前matched-support baselines。
- cafd/anchored_target.py、cafd/mistral_anchored_v14.py：残差锚定pilot。
- cafd/mistral_direction_v15.py：fit-only方向诊断。

### 12.2 主要run与结果文件

| run_id | 关键文件及用途 |
|---|---|
| mistral_cafd_disjoint_v7 | teacher/route.json与Teacher曲线 |
| mistral_cafd_support_full_v9 | final_scores.csv、costs.csv、final_summary.json |
| mistral_cafd_gen_v10_mixed | student_curves.csv；cafd/selected.json和confirmation.json |
| mistral_progressive_gkd_observation_v11 | student_curves.csv；progressive/selected.json |
| mistral_endpoint_observation_v12 | student_curves.csv；endpoint/selected.json |
| mistral_direct_observation_v12 | student_curves.csv；state下direct.json |
| mistral_cafd_anchored_v14 | pilot_gate.json；cafd/alpha_metrics.jsonl和selected.json |
| mistral_cafd_direction_v15 | diagnosis.json和progress.json |
| chat_hier_v4b_sft25 | 旧Qwen final_scores.csv及对应配置 |

曲线与汇总一般位于artifacts/cafd/experiments/<run_id>/，checkpoint和selected一般位于runs/cafd/experiments/<run_id>/，运行状态位于state/cafd/experiments/<run_id>/。CAFD v10选中checkpoint为cafd/step160，v14为cafd/step80，旧v9为cafd/step80；它们属于不同run，不可互换。

v14实现提交51e8f3c，仓库最近记录提交518157e；v15为本轮新增诊断脚本，不应把它归入此前提交。资源约束为CAFD最多4张B200、预留内存上限192G。本次整理仅读取实验记录，不提交或修改训练。
