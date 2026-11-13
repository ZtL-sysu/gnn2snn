# 文献与引用修复

原稿references.bib有42条；旧citation-gate主要检查数量、年份窗口、键与人为inventory的一致性，不验证文献真实存在或支持论点。后续primary-source审计将“论文身份匹配”与“元数据逐项正确”区分开：16条元数据一致，21条需修正，5条尚未核实。需修正包括真实论文的作者名字／名单、题名、来源、年份或BibTeX条目类型，不等于21篇都不存在。

特别问题包括：dualphase条目的题名／作者与实际论文不符，且不支持旧稿借它声称的特定累积电流解码等式；optimal-transport条目未找到匹配primary record；若干2025/2026作品元数据错配；SGNNBench“无转换”、PASCAL机制与“没有已有GCN转换”等表述仍须逐条核查。其中One-Spike已经涉及GCN转换，不能忽略。

原bib不改写为“当时就正确”。候选修正另存：`references-corrections-v6.bib`。完整42条canonical metadata、primary URL、核实状态及当前论点映射见`reference-truth-audit-v6.md`；这仍不是每个引用句都已完整验证的声明。

新方法相关研究还包括DCGS、TWE、Sigma-Delta Quantized Networks、DeltaCNN、DynaGraph和Ingress，以及SEW-ResNet／残差量化／时间编码。差分编码、线性增量更新、共享缓存、bias搬移早有先例；应把项目独特机制当待验证组合而非因“这里没搜到”就宣称首创。见`affine-incremental-prior-art-v1.md`、`twe-primary-audit-v1.md`。

下一稿应按实际主张选择真实、相关、支持该句的文献，再运行数量／年份／重复性检查。不能为凑36条或近期多数而加入无关或未经验证引用。
