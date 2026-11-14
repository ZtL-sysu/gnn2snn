# 问题、失败与修复台账

## 2026-11-14：强教师图级校准闭环

在独立研究分支完成六教师 × 四时间步 × 六条件的匹配校准与冻结测试；保留一次步长修正、全部负对照和三种子逐项差值。图级目标未通过稳定、实质优势门槛，按路线 C 收束。额外理想 ReLU 干预定位首个节点量化位置的条件影响，未将连续值诊断包装成新 SNN 方法。软件成本、部分运算下界、事件语义和未测量硬件能耗分开报告。完整记录见 [本轮报告](../research/20261114_story/REPORT.md) 和 [决策日志](../research/20261114_story/DECISIONS.md)。

## 历史记录

| 问题／路线 | 发现与边界 | 已做／未做 |
|---|---|---|
| 早期数据索引 | test.index排序与特征／图对应错位 | 早期修复记录与审计保留，不能把有缺陷结果继续引用 |
| 旧theta读出 | round(current*T/theta)之后没有乘theta解码 | 反例与修正版保留；未乘theta的恒等式自一致不等于教师保真 |
| 旧gauge | 训练默认theta1但MNIST转换theta1.5 | 新runner显式统一并检查实际pre/post-gauge logits |
| mask首次更新 | 旧helper在首个optimizer step后才清零，被称为窄网from scratch不准确 | 新train_secondary首forward前清零，每step后clamp；独立检查 |
| 旧所谓holdout | 不同seed测试子样本大量重叠且无实际保存IDs | 只能重构近似，不认证历史独立holdout；新跑保存实际IDs |
| baseline名称 | QCFS/FTBC/TPP等部分旧比较只是近似且读出不同 | 保留旧源码／失效说明；不冒充原作者忠实算法 |
| offline vs causal | 先定总count再排程不是在线IF | 新报告明确命名并检查相同输出；不从同一count发明额外优势 |
| learned-init候选 | MNIST全3seed低于FTBC，CIFAR小且不一致 | 保留负证据，不加seed寻找有利符号 |
| dense hybrid效率 | 每tick重算大部分稠密投影、gate、norm与归约 | 低T保真不等于稀疏执行／低能耗 |
| stale CDE/controller | 固定tau=.01下未取得有用的非零误差复用；现有算术floor不足5× | 路线关闭为特定负结果，不宣称所有未来cache不可能 |
| branch-fanout编码 | 完整train事件量使该3/2/1投影路线在有利计价下仍不满足旧5×边界 | 保留实际profile与成本推导，T1细分不能错称投影单项已越线 |
| affine事件行统计 | flatten后把整个矩阵算成1个非零行 | 独立3×2例RED/GREEN修复，官方affine profile前完成 |
| nested eval | 只检查model/layer.training会漏掉独立training BN/Dropout | 新helper检查所有submodule；实际官方run均eval |
| CUBLAS继承值 | setdefault不会拒绝已存在错误值 | 已核实六profile实际:4096:8、deterministicTrue；新shared driver先于Torch严格拒绝冲突，不改旧绑定源 |
| shared状态打印 | flush误传json.dumps会在写完结果后报错 | 正式shared运行前修复并检查 |
| 引用真实性 | 原格式／计数pass依赖人为assertion，不验证论文真实性与支持论点 | 42条primary-source审计：16metadata一致、21需更正、5未解决；旧bib保留，候选修正另存 |
| 新颖性边界 | 差分传播、共享／CSE、bias/base搬移和增量GNN已有先例 | 不能称首次图转换或把代数变形本身称新；特定组合尚待证明 |

“失败”在这里是研究证据，不是需要隐藏或删除的材料。不得从旧论文里挑仍好看的结论，而把反例、重叠、错误读出、成本未知或较强基线丢掉。

成本至少分开：首层模拟工作、base构建、投影、source/decoder、1/t、graded bank、gate sigmoid、乘积、归一化、图scatter、残差、BN、head、pool、存储与搬运。node BN可在条件下折A/B；edge BN的z同时供sigmoid gate，不能简单折CDE后宣称免费。最终不用的edge输出可以两侧公平prune。数学算术代理不等于3060或神经形态芯片实测能量。

详细记录：`correction-ledger.md`、`historical-track-repair-plan-v2.md`、`baseline-fidelity-audit-v1.md`、`gated-dual-arithmetic-screen-v1.md`、`affine-mode-cost-boundary-v1.md`。
