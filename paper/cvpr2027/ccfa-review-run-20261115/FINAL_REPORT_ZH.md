# GNN2SNN 论文冲刺最终中文报告

## 选定的论文故事

最终稿定名为 **A Finite-Event Audit of Decision and Temporal Fidelity in Graph-to-Spike Conversion**。我们选择机制审计路线，而不是强行包装成新方法论文。贯穿全文的中心思想是：在图到脉冲转换中，局部重构指标只能描述数值接近程度，不能替代对下游决策、完整前缀和全局事件活动的评估。

论文用两个受控反例支撑这一中心思想：

1. 在真实硬事件干预中，降低图消息 MSE 仍可能使教师类 margin 变差。T=1 时 adverse 比例分别为 MNIST 332/811（40.94%，95% CI [37.28,44.57]%）和 CIFAR10 296/928（31.90%，[28.66,35.31]%）。分层结果显示 8 个 rule/direction strata 全部保持同方向。
2. 在源重放合法的 two-tick codeword 交换中，局部解码张量逐位相等且局部事件数相同，但下游预测和 whole-network activity 发生变化。halving 改变预测 6/128（MNIST）和 16/128（CIFAR10）；doubling 改变 2/128 和 10/128。

这两项结果共同支持一个有限但清晰的结论：terminal equality plus fixed local event count does not determine downstream trajectory or decisions。论文明确说明 codewords 的第一前缀幅值和绝对幅值并不匹配，因此没有把该实验夸大成“纯时间因果”证据。

## 真正的贡献

- 把整个项目长期积累的不同实验统一为一个可审稿的有限事件审计问题：重构保真、决策保真和活动代价是三个不同量。
- 给出图消息扰动到类别 margin 的精确决策条件，并明确区分充分范数界与实际带符号扰动。
- 设计并执行真实可实现的硬消息干预，提供规则、eligibility、分层和 graph-cluster 统计口径。
- 设计 source-replay 的两 tick codeword 等价性测试，同时控制局部终点和局部事件数，并记录逐图决策变化、agreement 转移和全局事件数。
- 提出面向低 timestep 图转换的评估清单：检查每个所需 prefix 的决策、说明干预规则与 eligible set、报告 whole-network activity。
- 将位置敏感性、calibration、centered residual、历史节点分类和成本分析定位为动机、边界或背景证据，避免把它们拼成虚假的统一方法。

## 最重要的实验依据

- 硬事件 T1：MNIST 332/811 = 40.94%，CIFAR10 296/928 = 31.90%；T2 分别为 48.71% 和 44.07%。
- 分层 adverse rate：MNIST deterministic 30.6%/36.2%、residual 49.0%/48.5%；CIFAR10 deterministic 20.4%/24.7%、residual 42.6%/38.5%。
- 图级：MNIST any adverse 50.2%、any prediction change 7.8%；CIFAR10 为 42.2% 和 12.7%。
- 时间 codeword exchange：halving 的 eligible 数为 84,905（MNIST）和 132,899（CIFAR10）；doubling 为 55,510 和 63,500；所有 endpoint 的 local decode max error 为 0，局部事件数逐图相同。
- 全局事件：halving 增加 3.41%/4.99%；doubling 在 MNIST 增加 0.81%，在 CIFAR10 降低 1.31%。
- 校准对照只支持“在所测试目标中没有一致优势”，不声称最优校准目标。
- 历史全测试与节点分类结果均明确标为暴露数据或上下文证据，不充当新盲测。

## CCFA 评审与 R2 修改

R0 两篇独立评审均为 4/10；R1 提升至科学 5/10、写作 6/10。R2 两位评审一致认为，在“不做大实验”的约束下已经接近收益递减，剩余必改项都是文稿和包级别问题。

R2 已采纳的修改：

- 把错误的 “Pred-Quant” 改为 PD-Quant，并明确它是量化参数优化方法；本文没有优化量化参数。
- 明确 One-Spike 是 single-spike phase-coding converter；本文不是 One-Spike 实现，也没有做匹配的 one-spike baseline，只是在 DCGS-derived adapter 内做诊断性单事件干预。
- 全面收窄 temporal 因果措辞，明确控制的是 source legality、exact terminal decoded tensor、local event count 和 task graph；未控制 first-prefix profile 与 absolute amplitude。
- 在首次出现处定义 whole-network activity 为全网络非零标量事件总数，并全文统一使用。
- 修正 supplement 的复现包声明；把真实存在的硬事件行级 CSV、temporal endpoint NPZ、运行 JSON、源脚本和确定性生成脚本纳入匿名审计包，并明确哪些结果需要 checkpoint/dataset 或只存在于历史审计档案。
- 修正 revision log 中 verification 文件名，并把 LLM Usage Statement 改为真实使用情况，移除不准确的 GPT-6-Luna 披露。
- 重画 Figures 4–5：移除 pooled count overlay，放大标签，把图例移出柱状图区域。
- 压缩结论，突出两个核心反例和评估清单。

未采纳的建议均属于用户明确禁止或明确不划算的大实验：跨教师/未暴露数据复现、匹配 One-Spike 实现、新任务或拓扑、feature-dependent gate 干预、等幅值 temporal family、硬件能耗测量。R2 独立评审确认这些不属于低代价 must-fix。

## 验证结果

- 主 PDF：8 页正文加 1 页 LLM statement 与 references；结论结尾只占第 8 页少量空间，statement 与参考文献从下一页开始。CVPR 的 8 页正文限制按内容与 statement/references 分开核算。
- 补充 PDF：4 页。
- 自动化验证 `audit/verification.json` 全部通过：匿名性、resolved references/citations、嵌入字体、clean logs、LLM statement 位置、核心数字、strata、temporal counts、30 个原始/端点/脚本文件哈希、4 组 temporal endpoint 不变量、PD-Quant/One-Spike/temporal/activity 措辞、LLM 披露和 artifact claim。
- 两张重画图已人工目视检查，按双栏缩放仍可辨认。
- 所有历史原件、原始实验数据和 R1 冻结包均未覆盖。

## 投稿前仍存在的实质性问题

1. 核心硬事件与时间交换证据仍主要依赖 teacher 41、已暴露 validation 或 training graphs；这是当前论文最重要的外部效度限制。
2. 时间 codeword exchange 只证明“局部终点相等且事件数相同不足以决定下游结果”，不能隔离时间顺序与幅值差异。
3. 尚未做匹配的 One-Spike graph baseline，因此论文不能支持转换器之间的性能排序。
4. 历史节点分类、部分校准与成本结果缺少每行级原始数据；它们只作为上下文或辅助证据。
5. 没有硬件能耗测量，centered residual 只支持表示与 partial projection proxy 改善，不支持完整效率优势。
6. CVPR 2027 的行政注册、作者资格、匿名提交系统状态和最终官方模板需在真实投稿时由作者再次核对。当前成稿按 review-mode CVPR 样式准备，不假定已有投稿注册。
