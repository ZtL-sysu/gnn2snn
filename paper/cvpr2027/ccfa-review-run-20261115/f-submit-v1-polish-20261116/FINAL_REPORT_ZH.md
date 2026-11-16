# GNN2SNN 论文 F-Submit V1 精修轮最终报告

## 本轮目标与执行范围

本轮按用户要求使用 `f-submit-v1` 与 `scientific-paper-writing`，参考近年 CVPR/ICCV/ECCV/NeurIPS/ICML 顶会论文的写作和叙事方式，对既有成稿做投稿前精修。所有工作均在本轮独立目录 `f-submit-v1-polish-20261116` 内完成，没有覆盖 R2 原始稿、历史稿件、原始数据或图形文件，也没有新增实验。

执行流水线 S0–S10 完整留痕：稿件与证据清单冻结、模块映射、既有独立评审归档、CVPR 风格 profile、10 篇顶会 comparator、叙事结构摘要、模块评分、两轮 S7 精修、两轮 S8 独立重评、S9 包级审计和 S10 交付。第二轮精修是因为第一轮 S8 判定 Results 与 Figures/Tables 尚未达到冻结的顶会 presentation 阈值；第二轮仅修改这两个模块，重新编译并获独立重评接受。

## 本轮真正改进的内容

- Abstract 改成“proxy gap → 第一项条件性发现 → 第二项精确等价反例 → 机制解释 → reporting consequence”的五步推进，保留所有比例、分母、条件与数据范围。
- Introduction 改用有限事件决策失败作为开头，压缩通用转换背景，并把原先重复的 contribution/checklist 表述合并为一个中心主张和一个报告结论。
- Related Work 按 calibration/readout、event coding/timing、decision-aware objective、graph-dependent approximation 分组，使引用服务于能力定位。PD-Quant 与 One-Spike 的精确区别保持不变。
- Results 的四个关键小节改为 finding-first 主题句，让表格承载密集数值，正文负责方向与解释。第二轮进一步修正校准、图结构对照、runtime 小节的开头。
- Figures/Tables 的 caption 压缩为自包含的“量 + 关键范围 + 必要限制”，移除了防御性口号，同时保留暴露数据、单教师、重复干预非独立、本地终点相等、振幅未控和 training-only 等必要口径。
- Discussion/Conclusion 收回 decision、prefix、activity 三个报告量，避免重复 Results 与 defensive negation。

本轮没有改变科学结论、数字、实验条件、公式、引用、图文件、数据、LLM Usage Statement 或核心边界。

## 10 篇顶会 comparator 的写作手法学习

本轮检索并冻结了 20 篇以上候选，最终选用 10 篇近年顶会论文作为 presentation comparator。它们分别是：Temporal Misalignment in ANN-SNN Conversion；Inference-Scale Complexity in ANN-SNN Conversion；Differential Coding for Training-Free ANN-to-SNN Conversion；A Graph is Worth 1-bit Spikes；Are Conventional SNNs Really Efficient?；PD-Quant；Comparing the Decision-Making Mechanisms by Transformers and CNNs via Explanation Methods；VISTREAM；Efficient Event-Based Object Detection；FTBC。

学到并用于本文的主要叙事模式包括：以精确张力作为 opening hook；把 gap 压缩为“常见指标假设了什么、遗漏了什么”；机制审计使用 `audit/test/show`，只在确有方法时才使用 `propose`；概念图定义问题、主表承载决定性比较、分析图解释异质性；不确定性紧随受影响结果；限制在解释处与 Discussion 中两端放置；结论回收开头命名量而不新增结果。所有 comparator 只用于表达和结构对标，没有用其结论修改本文科学主张。

## 验证结果

- 主文 PDF：8 页；补充 PDF：4 页。
- 编译日志：0 undefined citations、0 undefined references、0 overfull box、0 LaTeX error。
- 匿名性：Anonymous CVPR submission，CVPR 2027 review-mode 样式。
- 数值与口径：核心比例、CI、分母、eligible pool、事件数、prefix/amplitude boundary、暴露数据标签与 baseline 一致。
- 图表：6 个主图、3 个主表；所有图文件与 baseline 字节一致。
- 引用：14 条参考文献；citation key 集合不变；无新增或未定义引用。
- LLM Usage Statement：保留原文，独立审计确认与 baseline 字节一致。
- Source ZIP：clean-extract 后重新编译通过，规范化 PDF 文本与 accepted working PDF 一致。
- f-submit-v1 ledger：S0–S10 全部 passed；S9 manifest 23 项 hash/size 全部 PASS。

## 投稿前仍存在的实质性问题

1. 核心硬事件与时间交换证据仍主要依赖 teacher 41、已暴露 validation/training graphs，外部效度有限。
2. 时间 codeword exchange 同时改变 first-prefix profile 与 absolute amplitude，只能证明 local terminal equality plus fixed local event count 不足，不能隔离纯时间顺序效应。
3. 尚未做匹配 One-Spike graph baseline，不能支持转换器之间的性能排序。
4. 历史节点分类、部分校准与成本证据缺少全部行级原始数据，只能作为上下文或辅助证据。
5. 没有硬件能耗测量；centered residual 只支持表示与 partial projection proxy 改善，不支持完整效率优势。
6. 本轮没有 live CVPR 2027 portal/registration 可验证；作者需在真实投稿时核对注册、作者资格、匿名流程和 portal-specific 字段。当前成稿按 review-mode CVPR 样式准备，不假定已有投稿注册。

## 结论

本轮完成了一篇逻辑一致、证据边界清楚、表达更接近顶会风格的 CVPR 成稿。最值得讲的故事仍然是机制审计：局部重构改善与下游决策、完整前缀、全局活动之间存在可复现的受控失配。它没有包装成新方法，也没有超出证据宣称 conversion-wide frequency 或 converter superiority。
