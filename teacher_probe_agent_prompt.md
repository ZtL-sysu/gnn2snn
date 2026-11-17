# 给本地 Agent 的任务提示词：ANN-to-SNN 图转换 —— teacher 归因 + 强 teacher 全量实验

> 用法：把下面整段（从「你是我的实验执行代理」到结尾）复制给你的本地编码 agent。
> 前置条件：仓库里已有论文那套 2 层 GCN + SNN 转换的代码，能跑出 Table 1/2/3 的数字。

---

你是我的实验执行代理，负责一篇 CVPR 投稿（ANN-to-SNN 转换，图结构视觉分类，superpixel 图）的全部补充实验。目标不是"再刷点"，而是**把一个反常现象归因清楚，并把论文从"弱 teacher 上有效"升级到"强 teacher 上同样有效"**。

## 0. 铁律（违反任何一条即作废，不要变通）

1. **先复现、后扩展。** 第一步必须复现论文现值：MNIST teacher 34.02±1.60%，CIFAR10 teacher 27.27±1.67%；conversion loss 3.79±2.26pp / 1.57±0.67pp；Table 2 的 Ours 62.60/65.79（MNIST T=4/T=8）、32.76/34.10（CIFAR10 T=4/T=8）；Table 3 消融 65.79/34.10（full）、53.65/23.46（β⋆ restored）、16.54/11.36（count readout）、45.97/18.36（random second phase）。**若复现偏差超过 ±2pp，立刻停止扩展，先写 `REPRO_BLOCKER.md` 说明差异来源**（配置不同？seed 不同？mask 不同？）再问我。
2. **不许在 test split 上选任何东西。** 所有 (λ, k, θ, T, 宽度, lr) 只在 validation 上选；test 只报一次。
3. **报全部 seed，不报 best。** 每个配置 5 个 seed（0–4），输出 mean ± std 和 95% CI。禁止 `max over seeds`。
4. **不静默丢 run。** 任何 OOM / NaN / 发散都记录进 `RUNLOG.md` 并降 batch size 重跑，不换 seed、不删点。
5. **口径写死。** conversion loss := `teacher_top1 − converted_top1`，**两者必须在完全相同的配置下**（同一 k、同一宽度、同一训练集、同一 A、同一 R）。这正是论文现在最大的隐患：Table 1 的 teacher(34.02%) 与 Table 2 的 Ours(65.79%) 不可能自洽，你要用同配置对照把它钉死。
6. 不要为了得到"好看的数"改数据管道、改 split、改评测。宁可得到负数结论并如实报告。

## 1. 复刻当前设置（唯一事实来源，不得改）

数据：PyG `SuperPixel MNIST` / `SuperPixel CIFAR10`（Dwivedi 版，SLIC superpixel + 8-NN 高斯加权边，MNIST 40–75 节点、CIFAR10 85–150 节点，官方 split 55k/5k/10k 与 45k/5k/10k）。

ANN（teacher）：
```
B1 = A(X W1) + b1
B  = R( A( ReLU(B1) W2 ) + b2 )
```
- `A`：固定（不学习）、对称归一化、非负、block-diagonal（同 batch 内多图拼成 N×N 稀疏阵）。
- `R`：固定线性 sum pooling，R∈R^{G×N}。
- 现有窄网配置：训练集 **2,000** 图、800 epochs（论文 §4.3 只提到这些，其余超参论文没写 —— 你去代码里找，并写进 `CONFIG_AUDIT.md`）。

SNN（转换侧，用来算 fidelity / energy）：
- 确定性 two-phase 计数：`s[t] = round(t*n/T) − round((t−1)*n/T)`，`n = round(ReLU(B1+β)·T/θ)` clipped 到 [0,T]，于是 `Σ_t s[t] = n` 精确成立。
- 主臂 `β = 0`；消融臂保留 `β⋆ = θ/(2T)`。
- 读出：膜电位解码 `argmax B̂`，`B̂ = (1/T) Σ_t inst_t`，`inst_t = R(A(s_t W2)) + b2`。
- 宽度控制：unit mask 把除 k 个 hidden unit 外的其余置零（按训练集激活幅度排序，转换前固定）。
- margin 正则：`λ·E[ReLU(1−Δ)]`，`Δ = B_y − max_{c≠y} B_c`。

能量（两个口径都要报，op-count 估计，45nm：MAC 4.6pJ / AC 0.9pJ）：
- ANN MACs = `E·F + N·F·H + E·H + N·H·C + N·C`（N 节点数、E 非零边、F 输入维、H hidden、C 类数）。
- 主口径 `R_prim = MAC·4.6 / (AC·0.9)`，`AC = S·(H + d̄·C)`，S = hidden spike 总数，d̄ = E/N。
- 保守口径 `R_cons = MAC·4.6 / (0.9·S·H + 4.6·S·d̄·C)`，即把实值传播也按 MAC 计价。
- **先验证这两个式子能复现 Table 1 的 9.103×/6.086×（MNIST）和 7.603×/5.938×（CIFAR10）**；如果复现不了，说明代码里的计价口径与论文描述不一致，先报告再动。

## 2. 实验 A：teacher 归因（2×2×2 因子设计，最关键）

固定其他一切，只动三个因子：训练集规模 {2k, 全量}、hidden 宽度 k {16, 128}、edge feature {无, 有}。5 seeds，两个数据集。

| ID | train | k | edge feat | 目的 |
|---|---|---|---|---|
| T0 | 2k | 16 | 无 | 复现现值（应 ≈34% / 27%）|
| T1 | 全量 | 16 | 无 | 数据规模主效应 |
| T2 | 2k | 128 | 无 | 宽度主效应 |
| T3 | 全量 | 128 | 无 | 两者交互 |
| T4 | 全量 | 128 | 有 | edge feature 效应 |
| T5 | 全量 | 128 | 有 + BN/dropout + cosine lr | 上限探针 |

输出 `results/teacher_probe.csv`（列：dataset, exp_id, train_size, width_k, edge_feat, seed, teacher_top1, wall_clock_s）以及一张 LaTeX 表。

**必须回答**（写进 `FINDINGS.md` 的 Q1）：34.02% / 27.27% 里，分别有多少个点来自"只用了 2k 图"和"宽度只有 16"？（用 T0→T1 和 T0→T2 的差值给出点估计和 std。）

## 3. 实验 B：强 teacher 上的转换（论文升级的关键）

取 T5（或 T4，哪个高用哪个，说明理由）作为强 teacher，跑转换：

- T ∈ {2, 4, 8, 16}，β ∈ {0, β⋆}，5 seeds。
- 每个 (T, β) 在 validation 上选 (λ, k, θ)，**不许碰 test**。
- 报：teacher_top1、converted_top1、**同配置 conversion loss**、float64 恒等式残差。
- 额外跑一条"宽度扫描"：k ∈ {8, 16, 32, 64, 128} × T ∈ {2,4,8}，得到 fidelity–width 曲线（这是新 Figure 的最佳候选）。

**必须回答**（Q2）：teacher 提到 ~90%（MNIST）/ ~75%（CIFAR10）后，conversion loss 是否仍 ≤2pp？如果显著变大，说明什么（是不是 margin/量化在高精度区更受限）？
**必须回答**（Q3）：去掉 β⋆ 的收益是绝对量还是相对量？在强 teacher 上还成立吗（预期：仍成立，因为机制是"双重校正"）？

## 4. 实验 C：能量口径在强 teacher 上是否还活

强 teacher 通常更宽 → spike 更多 → 5× 优势可能消失。这正是要测的。

- 对实验 B 的每个工作点算 R_prim / R_cons，两个数据集，per-seed 记录 min/max。
- 明确标注哪些点满足"teacher > 2× chance"的可采用条件，哪些不满足（论文 §3.5 的规则），**不要偷偷放松这条规则**。
- **必须回答**（Q4）：能量优势与保真度之间的真实 frontier 长什么样？如果强 teacher 上 5× 消失，那条边界在哪（给出 (teacher_acc, R_cons) 的散点 + 拐点）？

## 5. 实验 D：把 Table 1 的自洽性钉死

- 用**同一配置**（同 k、同宽度、同训练集）重算：teacher_full_top1、teacher_at_working_point_top1（masked）、converted_top1、loss vs 后者。
- 输出"反超审计"：`converted_top1 − teacher_at_working_point_top1`，若为正，逐 seed 列出并检查是否存在信息泄露（mask 是否在 train 之前生成？校准集是否与 test 重叠？`R` 是否见过标签？）。
- 这是**整个任务里优先级最高的一件事**：论文现值出现了 converted(65.79) 比 teacher(34.02) 高 31.77pp 的组合，而 loss 却是正的 3.79pp。三者在同一配置下不可能同时成立。查清它，写进 `FINDINGS.md` 的 Q5，附最小可复现脚本。

## 6. 交付物（全部放进仓库，路径写死）

```
teacher_probe/
  CONFIG_AUDIT.md          # 从代码里提取的全部超参（含论文未写的：optimizer/lr/batch/H/dropout/feature dim）
  REPRO.md                 # 复现表：论文值 vs 本次复现值 vs 偏差
  REPRO_BLOCKER.md         # 仅当复现失败时存在
  RUNLOG.md                # 每条命令、耗时、硬件、异常
  FINDINGS.md              # Q1–Q5 的答案，每条 ≤ 5 句，必须带数字和 CI
  results/
    teacher_probe.csv
    strong_teacher_conversion.csv
    energy.csv
    fidelity_width.csv
  tables/
    tab_teacher_probe.tex
    tab_conversion_strong.tex
    tab_energy.tex
    tab_table1_fixed.tex    # 四列版：Teacher(full) / Teacher@working-point / Converted / Loss
  figures/
    fig_fidelity_width.pdf
    fig_frontier_strong.pdf
  run_all.sh                # 一键复现全部实验，含固定 seed
```

## 7. 工程要求

- 单一 `config.yaml` 驱动，禁止把超参散落在脚本里；每行结果必须能反查到 config hash。
- 每个实验幂等、可断点续跑（已经存在的 (config_hash, seed) 直接跳过）。
- 固定 `torch.manual_seed`、`numpy`、`random`、`torch.use_deterministic_algorithms(True)`；CUDA sparse reduction 非确定，所以每个点 5 seeds，并记录 std（论文已声明这一点，保持一致）。
- 恒等式检查必须用**同一个 float64 forward pass**，报 max|B̂−B − R A(E1 W2)|，预期 ~1e-14；同时记录 fp32 与 fp64 重算的差距（论文报 3.7e-7）。
- 所有图用 matplotlib，不加标题、字号 ≥9pt，直接能进 LaTeX（`\usepackage{graphicx}`，宽度 `\linewidth`）。
- 显存不够时降 batch，不降 seed 数、不降 train 规模（train 规模是实验变量，改了答案就废了）。

## 8. 预算与终止条件

- 预估：实验 A 12 runs、实验 B ~40 runs、实验 C 复用 B 的 spike 统计、实验 D 少量。单 run 在单卡上应 ≤10 分钟，总计 ≤ 4 小时。若某配置超过 30 分钟，停下来告诉我，不要硬跑。
- **终止条件**：`REPRO.md` 任一行的偏差 > ±2pp → 停。`teacher_probe` 里 T5 的 teacher 反而低于 T0 → 停（说明数据管道有问题）。反超审计发现泄露 → 停并优先报告。

## 9. 最后一步

跑完写 `FINDINGS.md`，并在最上面用三句话总结：**34% 的来源是什么、强 teacher 上转换是否仍然成立、论文该按"双轨报告"怎么改**。不要写客套话，直接给数字和结论。如果我需要做取舍，给出你推荐的报告方案（单轨/双轨/三轨）和理由。
