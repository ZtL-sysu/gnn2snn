# 创新方向与"做扎实"建议（Claude 审阅）

- 日期：2026-11-14
- 审阅对象：`ZtL-sysu/gnn2snn` 的 `main`，提交 `f19a2f2`
- 审阅方式：通读 `docs/`、`research/visual-graphs/`、`paper/historical-draft/`、`review-suggestions/grok-bot/`；未运行训练或推理，未展开 `artifacts/verified-evidence.zip`
- 目标调整：项目所有者 2026-11-14 说明 CVPR 2027 不再赶，目标是同等级顶会，重点是创新和把工作做扎实

本文只新增在 `review-suggestions/claude/` 下，不改动原有研究代码、证据和历史稿。文中的数字分三类，请勿混用：

- **仓库数字**：来自仓库已归档文件，标明出处。
- **本文推算**：由仓库数字按仓库自己的记账约定算出，可用同目录 `energy_ratio_check.py` 复核。
- **建议与假设**：尚未做的实验、提议的判据，不是结果。

---

## 1. 一句话结论

现在的创新押在"静态图像、单次推理内 T 个 tick 的编码效率"这条轴上，它不是图特有的，收益上限也低。建议把创新移到两处图结构真正改变转换问题的地方：**离散结构决策的稳健性**（创新点 B）和**流式视觉图上的时间冗余**（创新点 C）；已有的残差前缀编码作为组件保留，并用"同比特量化 ANN"对照把能耗叙事做诚实。

---

## 2. 时间

CVPR 2027 的论文注册截止是 2026-11-10，正文截止是 2026-11-16（AOE），官方写明不延期（见参考 [R1]）。这一届不赶。后续场地的截止日期本文未核实，定目标前请以官方页面为准。

---

## 3. 现状判断

### 3.1 站得住的

- **强教师上的低时步保真。** 三个教师种子、完整官方测试集，T=4/8 与教师的差距不超过 0.10pp：MNIST 固定 GCN 教师 91.02±0.27%，CIFAR10 GatedGCN 教师 68.77±0.41%（仓库数字，见 `fixed-gcn-test-three-seed-summary-v1.md`、`gated-three-teacher-seed-summary-v1.md`）。
- **证据纪律。** 读出错误、holdout 重叠、基线冒名、测试集暴露、引用问题都如实记录在 `docs/RESEARCH_LOG.md`。

### 3.2 还撑不起方法论文的

- 保真结果来自 DCGS 的多阈值差分单元 [R2] 的图适配，不是本项目提出的方法；仓库自己也这样表述。
- 本项目的候选（affine-base / centered 残差前缀共享编码）只有一个教师（CIFAR10 GatedGCN，seed 41）在 45,000 张训练图上的教师一致率，没有验证集或测试集精度（`affine-six-profile-audit-v1.md`）。
- 历史稿（两层 GCN、教师 34.02% / 27.27%、各转换基线接近随机）不能作为新稿基础，仓库已有同样结论。

---

## 4. 四个关键问题

### 4.1 缺"同比特量化 ANN"对照

8 级带符号多幅度事件乘以 T=4，信息量相当于一个多比特量化网络，近乎无损并不意外。Yan 等 [R3] 指出每个率编码 SNN 都有一个近似等价的量化 ANN"孪生"，公平的能耗比较应在这二者之间做。审稿人会问：同比特预算的训练后量化 GNN，精度和成本是多少？

`review-suggestions/grok-bot/` 的基线清单里有 QCFS、FTBC、SNN Calibration 等转换基线，没有这一条。本文认为它是必须项。

### 4.2 "5× 节能"在当前设置下算不出来

按仓库最乐观的记账（只计一次性基项、门控已知项、事件投影 ADD），用 `affine-six-profile-audit-v1.md` 的投影 ADD 和 `affine-mode-cost-boundary-v1.md` 的约定推算（本文推算，`energy_ratio_check.py` 可复核）：

| 模式（scale 8） | T | 教师一致率（训练集） | ANN / 候选（算术能耗，上界） |
|---|---:|---:|---:|
| centered | 1 | 83.25% | 5.49× |
| centered | 2 | 96.19% | 3.53× |
| centered | 4 | 98.78% | 2.63× |
| first_analog | 4 | 98.48% | 2.16× |
| generic | 4 | 97.88% | 1.45× |

说明：

- 这是上界。未计源单元状态与比较、1/t 解码、sigmoid、除法、scatter、BN、访存；对照的 ANN 是 FP32 且未做 ReLU 零值跳过。
- 一致率不是精度；只有一个教师。
- 根因是事件不稀疏。旧的分支编码 profile（`gated-dual-arithmetic-screen-v1.md`）里，T=1 时节点位点每标量事件数为 0.45–0.50，边位点为 0.50–0.55；T=4 累计节点位点每标量 1.36–2.00 个事件（本文推算：表中事件数除以 N×70 或 E×70）。而约定下 MAC/ADD 的能耗比只有 5.11。
- centered 真实且可辩护的增量：相对 first_analog，T=4 的投影 ADD 少 23%（scale 8）到 24%（scale 1），一致率不降。

建议：废掉"5×"这个自设门槛，改报精度–成本 Pareto 曲线，并给出 4.1 的量化对照。

### 4.3 视觉分量不够

SuperPixel MNIST / CIFAR10（教师 69%）在视觉顶会会被当成玩具设定。免训练转换的最大红利是不用训练：可以直接拿 Vision GNN（ViG / PyramidViG）的 ImageNet 预训练权重做转换，只需校准和评估。Grok 审阅把这项列为"不作为关键路径"（`cvpr-actionable-guide.md` 的 E6-C）；本文的排序相反，建议升为主实验载体之一。

本次检索没有发现 ViG 的 ANN-to-SNN 转换工作，但检索范围有限，定稿前需要系统查一遍，不能据此写"首次"。

### 4.4 "这还算 SNN 吗"

门的 sigmoid、归一化除法和分类头都是模拟计算，事件是带符号多幅度的（仓库在 `gated-result-to-claim-v1.md` 已如实说明）。两条出路，选一条写清楚：

- 把这些非线性也事件化：输出 f(前缀_t) − f(前缀_{t−1})。脉冲 Transformer 的免训练转换走的是同一类思路 [R4]。
- 如实定位成"事件驱动 / sigma-delta 式推理"，不硬叫 SNN。

---

## 5. 创新方向

出发问题：**图结构在哪里真正改变了转换问题？**

### 5.1 创新点 B：结构先于数值（建议作为核心）

视觉图骨干里有一批离散决策：特征空间 kNN 选邻居、max 聚合取哪个邻居、门是否饱和。它们只在误差超过某个"结构间隔"时才翻转，这是 CNN 和 Transformer 转换里没有的现象。

可以得到三样东西（均为待验证假设）：

- **理论**：类比分类 margin，用"第 k 近与第 k+1 近邻居的距离差"对"特征误差界"，给出拓扑翻转概率的上界；max 聚合同理，用"最大值与次大值之差"。
- **方法**：前几个 tick 定结构后锁存，之后只细化数值；max 聚合只在胜者改变时重算。
- **可检验的规律**：由 ANN 的结构间隔分布预测所需 T。这接回仓库早期的"预测规律"（2026-11-04）和"聚合内时序错位"（2026-11-05）两个分支。

载体：ViG（动态 kNN + max-relative 图卷积，ImageNet 预训练权重）和 DGCNN（EdgeConv，ModelNet40）。两者的出处与官方代码链接见 Grok 审阅 `cvpr-actionable-guide.md` 的 E6 节，本文未重新核对。

### 5.2 创新点 C：让 SNN 的时间等于真实时间（建议作为限时赌注）

差分编码真正省计算的地方是输入本身有时间冗余的流式视觉图：骨架动作识别、事件相机图、点云序列。SNN 的 tick 对齐真实帧或事件，而不是对同一张静态图反复细化。

这里有一个图特有的问题：一个输入变化经过 k 层会扩散到 k 跳邻域，事件稀疏性随深度衰减。带阈值的分级差分可以抑制扩散，代价是有界误差；仓库已有的归一化门界（`stable-normalized-gate-bound-v1.md`）可以用于误差递推。

建议先做骨架：拓扑固定，`docs/METHODS.md` 的线性误差恒等式可以原样成立；预训练模型多，数据小。

### 5.3 组件 A：残差前缀编码 = 沿深度的差分

已有的 centered 编码可以重新表述为"沿深度的差分"：DCGS 沿时间差分，本项目沿恒等残差的深度方向去掉静态基项。作为组件并入 B 或 C，补齐验证集 / 测试集 / 多教师证据即可，不单独当主贡献。适用范围仍限于等宽恒等残差、非负 ReLU 增量。

### 5.4 评测骨架 E：盈亏条件

用同一权重精度比较量化 ANN 与事件驱动推理，每个突触上：

- 量化 ANN（做零值跳过）：ρ × E_MAC，ρ 为激活非零比例
- 事件驱动：k(T) × E_ADD，k(T) 为到第 T 个 tick 的每标量累计事件数

事件驱动占优的条件是 **k(T) < ρ × E_MAC / E_ADD**。在仓库的 FP32 约定下 E_MAC/E_ADD = 5.11。这个条件说明：静态输入下 k(T) 随 T 增长，空间有限；流式输入下相邻帧的 k 可以远小于 1，这正是做 C 的理由。ρ 需要在教师上实测，本文没有这个数字。

### 5.5 与已有工作的区别

| 已有脉络 | 确立了什么 | 对本问题的局限 | 本项目可加的东西 |
|---|---|---|---|
| DCGS 等免训练转换 [R2][R4] | 差分多阈值单元在 CNN / Transformer 上低 T 保真 | 不涉及动态拓扑、max 聚合、图门控 | 新算子类与结构间隔理论（B） |
| 骨架脉冲 GCN：MK-SGN [R5]、Signal-SGN [R6] | MK-SGN 在 NTU-RGB+D 60 上用 4 个时间步取得有竞争力的精度；Signal-SGN 把骨架序列的时间维当作脉冲时间步 | 据摘要判断是专门设计并训练的脉冲网络（含知识蒸馏），不能直接复用现成的强 ANN；需读全文确认 | 免训练转换现成强教师，带误差界（C） |
| AEGNN 类异步图推理 [R7] | 新事件到来时只重算受影响的节点，同步训练、异步部署 | 精确重算，更新范围随层数按 k 跳扩大 | 阈值化差分，用有界误差换稀疏（C） |
| Sigma-Delta / DeltaCNN / DynaGraph / Ingress | 时间差分、线性增量、共享缓存（见仓库 `affine-incremental-prior-art-v1.md`） | 不针对图转换中的结构决策 | 残差前缀只作为组件（A） |

---

## 6. 建议的推进顺序与判决实验

以下判据是提议，需团队在开跑前确认并写死；它们不是结果。

### 6.1 B0：结构稳定性探针（先做，一两天，不需要 SNN）

- **做法**：取预训练 ViG-Ti，把各层特征均匀量化到 b ∈ {2,3,4,6,8} 比特（或注入等方差噪声），逐层测量：kNN 邻居集合与全精度的重合率、max 聚合 argmax 一致率、ImageNet 验证子集 top-1。
- **假设**：结构指标在比数值误差更低的比特数下就接近 1。
- **判读**：
  - 结构很早稳定 → B 成立，方法是"先定结构、后细化数值"，继续做理论和转换。
  - 结构很脆弱 → 这是一个明确的失效模式，论文转为"如何保护结构决策"（例如结构部分用更高精度或模拟计算）。
  - 两种结果都能写，区别只在方法节。

### 6.2 E0：盈亏条件的实测（与 B0 并行）

在已有的 CIFAR10 GatedGCN 教师上实测 ρ，并实现一个同比特训练后量化对照，报告精度与按 5.4 记账的成本。产出是 4.1 要的那一行基线。

### 6.3 C0：骨架上的时间稀疏性探针（限时 3–4 周）

- **做法**：取一个公开的骨架 GCN 预训练教师，测相邻帧各层激活差分的非零比例，以及加阈值后的精度–稀疏曲线。
- **止损**：若阈值化后精度可接受的区间内，k 相对逐帧重算没有明显下降，就把 C 收成一节或放弃，核心不受影响。

### 6.4 A 的补证据

centered / first_analog / generic 三模式，在三个教师种子上做验证集选点、冻结后各跑一次测试集，报告成对差值。这部分 Grok 审阅 `cvpr-actionable-guide.md` 的 E1 已有详细协议，本文不重复。

---

## 7. "做扎实"具体指什么

仓库的审计文化已经很强，缺的是另外几样：

1. **每个创新点配一个可证伪假设和一个判决实验**，判据在跑之前写下。现有的"门"（5×、评审分 ≥7）是工程指标，不是科学假设。
2. **成本从真实执行里数出来。** 至少用一个事件驱动的计数器实际跑一遍，不靠公式外推；有条件再做稀疏实现的墙钟。
3. **换数据集时重建干净协议。** 校准和选点只用训练集划出的固定子集；SuperPixel 的测试集已暴露，新数据集上不要重蹈。
4. **人要掌握核心。** 研究历史显示两周内由多个 AI 代理产出，出过多次严重错误（theta 解码、holdout 重叠、基线冒名）。建议一作亲手整理一个几百行的最小核心加测试；`takeover_20261112/` 里几十个带版本号的脚本和 `C:/code/phd-snn` 绝对路径难以交接。
5. **引用全部重拉。** `docs/REFERENCE_NOTES.md` 记录 42 条里 21 条需更正、5 条未核实。新稿的 bib 应从 DBLP 或出版社页面重新取，不在旧 bib 上修。CVPR 2027 要求 LLM Usage Statement，缺失会不送审直接拒 [R8]；其他会议的同类政策投稿前要查。
6. **匿名风险。** 公开仓库带账号名、稿件和目标会议字样。双盲投稿前应转私有或去掉会议信息。

---

## 8. 与 Grok 审阅的异同

- **一致**：历史稿不能投；5× 不宜作硬目标；需要忠实基线、完整成本、换强教师；残差前缀编码是候选组合而非已证方法。
- **不同**：
  - 本文认为主创新不应是残差前缀编码本身，而应是图特有的结构决策（B）和流式时间冗余（C）。
  - 本文把 ViG / ImageNet 从"非关键路径"升为主载体之一，理由是免训练转换不需要训练成本。
  - 本文增加"同比特量化 ANN"对照和盈亏条件。
  - 本文建议先做不需要 SNN 的廉价探针（B0），再决定方法形态。

---

## 9. 本审阅未核验的事项

- 未运行任何训练、推理或仓库里的检查脚本；仓库数字按其文档引用。
- 4.2 的事件密度来自旧的分支编码 profile，不是 affine 三模式的逐位点密度。
- ViG、DGCNN、骨架 GCN 教师的公开精度与代码未在本次核对；ViG 转换的"未见先例"只是有限检索的结果。
- MK-SGN、Signal-SGN 只读了摘要和检索摘录，未读全文。
- 5.4 的盈亏条件是一阶算术模型，未计访存、稀疏执行开销和激活位宽对乘法器成本的影响。
- 后续会议的截止日期未核实。

---

## 参考

本次核对过出处页面的条目：

- [R1] CVPR 2027 Call for Papers：https://cvpr.thecvf.com/Conferences/2027/CallForPapers
- [R2] Huang 等，Differential Coding for Training-Free ANN-to-SNN Conversion，ICML 2025（PMLR 267，第 25235–25264 页）：https://proceedings.mlr.press/v267/huang25i.html ；arXiv:2503.00301
- [R3] Yan、Bai、Tang、Wong，Reconsidering the Energy Efficiency of Spiking Neural Networks Inference from Analytical Perspectives，arXiv:2409.08290：https://arxiv.org/abs/2409.08290
- [R4] Wang 等，Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformers，AAAI 2026（40(3)，第 2128–2136 页），DOI 10.1609/aaai.v40i3.37195
- [R5] Zheng、Xia、Liang、Chai，MK-SGN: A Spiking Graph Convolutional Network with Multimodal Fusion and Knowledge Distillation for Skeleton-based Action Recognition，arXiv:2404.10210：https://arxiv.org/abs/2404.10210
- [R6] Zheng 等，Signal-SGN: A Spiking Graph Convolutional Network for Skeleton Action Recognition via Learning Temporal-Frequency Dynamics，arXiv:2408.01701：https://arxiv.org/abs/2408.01701
- [R7] Schaefer、Gehrig、Scaramuzza，AEGNN: Asynchronous Event-Based Graph Neural Networks，CVPR 2022，第 12371–12381 页，DOI 10.1109/CVPR52688.2022.01205
- [R8] CVPR 2027 Author Guidelines：https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
