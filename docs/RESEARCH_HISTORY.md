# 按时间的研究历史

以下原始目录用于历史定位；精简仓库不逐个保留所有旧运行。

## 2026-10-28：方向建立与图转换基线

WorkBuddy围绕ANN→SNN、概率脉冲、FTBC、低时步转换形成方向卡、guidance与CCFA启动材料。最初的Spiking-GNN-Conversion在Cora/CiteSeer/PubMed上运行两层GCN，研究朴素、偏置修正和校准上界，并进行多seed/T-grid实验。早期修复过test.index排序导致的特征图错位、lam缩放、第二层应累计二进制脉冲以及PubMed稀疏文件读取等问题。

该阶段既产生论文／报告，也有审计和“审计论文 vs 方法论文”的不同交付版本。历史报告的medium/strong结论只对应当时定义，不能移植为当前视觉主线的强结果。入口：archive/workspace/auto-paper/20261028-171500/Spiking-GNN-Conversion/。

## 2026-10-29至11-03：误差、读出和图结构机制

项目继续讨论邻域聚合、rho加权误差、系统偏置与时间错位、阈值／时步、margin、spike数量和成本。每日记忆记录了运行、诊断、反例、参数选择及审稿往返。请把每日记忆与对应results/audits/protocols同时阅读，不能只引用最后一条总结。

## 2026-11-04：predictive-laws-gnn2snn

形成预测规律／理论导向的独立材料目录，延续从转换误差到图结构、margin和所需时步的研究。此分支单独保存，不把理论意图等同于已经用强视觉教师验证的普遍定律。

## 2026-11-05：Intra-Aggregation-Misalignment与评审迭代

聚合内错位分支包含完整f-research运行、协议、结果、论文PDF、LaTeX和图；另有ccfa-startup-packet、red-team、review reports、ideas和smoke-tests。这些文件记录了研究如何收敛与转向，以及哪些论点曾被提出。入口：archive/workspace/auto-paper/20261105-111502/与20261105-204552/。

## 2026-11-06至11-10：相关视觉／时空图分支

WorkBuddy多轮探索果蝇连接组、GNN/SNN视觉机制、膜状态与真实光流，并通过novix任务、远程项目快照和本地pull目录保存实验历史。这里把它们列为相关分支而非混入SuperPixel分类主结果。自动循环和远程运行编号可从原始日志与可见会话追溯；未取得有效执行步骤的任务不能当作实验结果。

## 2026-11-10至11-11：视觉图转换论文与teacher probe

Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs建立SuperPixel MNIST/CIFAR10主目录，产生E11/E12/E17/E18等小样本、两层、强mask实验与文章。随后开展teacher质量／宽度／数据量探针，形成novix_delivery_21591交付。另有f-submit布局与文字修订副本。本机原始材料不被覆盖；精简仓库只保留关键证据与一份可续写稿，其余过程写入日志。

## 2026-11-12：Codex接手与证据纠错

接手核查发现：旧teacher远弱于适用参照，旧运行没有精确checkpoint/IDs，部分“holdout”样本跨seed重叠；theta没有在隐藏读出中正确恢复；gauge与转换theta不一致；离线排程和因果IF、不同readout以及若干被命名为论文基线的近似实现混在一起；原成本账漏掉首次模拟计算与图算子。重新建立完整官方数据、完整图、强teacher与来源绑定的主线。

## 2026-11-12至11-13：强教师、完整端点与方法重构

完成四层固定GCN和强GatedGCN的训练／校准／多种子端点；完整保存每样本预测与原始source/checkpoint/data哈希。独立审计确认近教师的差分适配保真。learned-init路线呈负／不稳定结果；stale-CDE/controller、证书复用和branch-fanout成本路线关闭为负结果。进一步推导并实现raw residual与affine base分离的共享consumer编码，先做CPU恒等／源码控制，再进行完整训练集诊断及预先约定的共享theta控制。

同时修复旧两层次要轨，持久化数学检查，重做成本边界，核查新颖性与引用。全部这些工作都只构成当前证据，不自动构成CVPR接收级文章。

## 2026-11-13：用户暂停与公开归档

用户要求暂停原研究目标，把这个大方向的成果与WorkBuddy/Codex历史分门别类完整整理，发布公开gnn2snn项目。本仓库就是该快照。暂停时没有研究Python/Conda进程运行；计划但未执行的工作保留在恢复指南，不能把旧LIVE文字当成恢复指令。


## 归档精简说明
按用户后续要求，只保留关键代码、已核验成果、一个可续写稿件与必要证据。失败／重复尝试汇总在RESEARCH_LOG.md；不上传原始大对话、巨型snapshot、完整远程镜像或原数据缓存。

## 11月13日后续检查点与再次暂停

归档之后曾按执行方案恢复，完成E0完整5000验证重放、匹配分类头／独立消费者编码合成检查，以及seed41三模式×四T的完整官方验证。结果、代码、协议与预测已同步到公开工作分支，详见[精简索引](ARCHIVE_INDEX.md)。E0历史T8存在一例类别漂移，E1仍缺其余种子、完整对照及成本；不把这批测量称为终稿或方法优势验收。用户随后再次明确暂停并要求精简整理，研究推进现已停止。
