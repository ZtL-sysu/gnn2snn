# RQ1：消息误差与分类决策为什么失配？

本轮支持“误差方向与竞争类别比单一消息范数更关键”，尚未证明一套完整的结构因果解释。全部新结果：教师41、两任务各512张历史选择暴露验证图；不代表多教师或独立确认。

## 数据与回源

`analyze_cached.py` 从历史两张端点表回算142/142准确率、一致率和原始NPZ哈希；扩展检查187个可回算结果条目全部一致，另9个旧格式条目明确未按同一接口回算。`audit_history.py` 从已暴露存档 logits 回算三教师，未加载官方test数据集或进行新test推理：GCN教师MNIST91.0233%、CIFAR10 54.9400%，原转换T1 71.4333%/46.8067%，历史全位置局部78.5000%/49.9233%，GatedGCN教师68.7700%。这些历史完整测试数字与本轮val512严格分列。

历史真实图残差分配相对局部分配：CIFAR10有491/512张图的首tick消息MSE更低，其中27张T1分类由对变错；MNIST对应440张、17张。详见 `analysis/cached/per_graph_diagnostics.csv`。准确率与教师一致性、教师正误四格、教师top2/真值margin、竞争类别切换均独立统计。补偿本身增加事件，不能当等成本算法。

## 受控硬事件干预

冻结教师和校准参数、保持任务图。对每张图首位置的一个已有非零事件，在原几何字母表内升/降一级，使用确定性坐标控制与最大局部残差坐标两类选择。两方向按各自可移动坐标集合选点，不是同一坐标的严格正负成对试验。标签、教师类别和logits不参与选点；教师只用于离线诊断。

每任务512图×4种干预=2048次T1尝试，同样参数/干预检查T2。v2逐行记录节点/通道及实际改动，全部尝试确实改变一个事件。v1保留，v2补充定位字段并将此前不准确的`random`名称改为确定性控制；两版公共数值字段应逐行一致，不算新增科学筛选。

|任务|消息MSE下降次数|其中T1教师类别margin变差|比例及按图聚类bootstrap95%区间|教师一致丢失/恢复|
|---|---:|---:|---|---:|
|MNIST|811|332|40.94% [37.28%,44.57%]|16/16|
|CIFAR10|928|296|31.90% [28.66%,35.31%]|36/30|

此处消息误差是**首tick下一预激活消息**的真实图均方，T2列不是累计T2消息误差。T2同一首tick消息改善时，margin变差比例为48.71%/44.07%。重复干预按图聚类，不按独立2048样本计算不确定性。

具体反例：MNIST官方val索引437，教师与真值均为5，确定性坐标降一级将首消息MSE从0.00604723降至0.00604511，转换教师类别margin却从1.97242变为−3.74764，预测5→4。教师top2 margin本身高达5.20378，说明“小教师margin”并非所有翻转的充分解释。全部丢失一致性案例均保留在 `analysis/phase3/all_lost_agreement_examples.csv`，没有只选有利图。

机制上，下一消息误差范数没有约束误差在“教师类别−任一竞争类”方向上的投影。后续残差路径、量化边界与图读出会改变方向和竞争类别；因此范数下降可以同时跨越最终分类边界。硬干预证明这种失配确实可发生，但尚未逐模块中介分解残差、读出与门切换的相对贡献，也不是所有图传播条件下的因果定理。以最终margin符号解释翻转本身只是决策恒等式，不能充当算法创新。

## 结构是否额外解释？

本数据使用固定出度近邻图；cached出度均值/方差以及新算出度方差是常数，不能据此说结构无效。CPU重建入度方差、邻域特征差和空间边长后，确有结构变化。控制教师margin后，结构变量对局部→消息配对正确性差的样本内增量R²为CIFAR10 T1/T2 0.00964/0.03547、MNIST0.00954/0.00372；这很弱且不是独立预测验证。扩展设计含常数/相关列，秩与常数列在JSON明确记录。尚不支持“结构提供不可替代判别信息”。

## 时间前缀与目标审计

原差分递推：事件改变量δ在T1后同时导致 `mem -= δ`、`cumulative_out += δ`、`exp_out += δ`；因此它不仅改变当前输出，还改变T2的量化比较与负事件资格。解码为Σevent_t/t；T1改善不能推出T2改善。工程测试验证三个状态更新，基线在T1/2/4/8与来源链接的前向逐位一致。干预没有错误地在T2重置补偿状态；但没有完整中介拆分来证明哪一状态造成多少退化。

另一个重要口径：历史optimizer_check T2是重新拟合T2参数；本轮所有T均共享一个参数向量。两者不能直接作为前缀退化的公平对照。

本轮训练侧审计还证伪了“只保护教师top1与runner-up就足够”的简单假设：top2 ranking硬训练目标在两任务下降，参数均未触及±0.5边界，但T1/T2平均转向教师top2之外类别的比例，MNIST由4.6875%增至22.9492%，CIFAR由9.4727%增至27.3438%。因此不能把其失败简单归为无梯度或边界饱和；也不能推广为所有排序目标无效。

## 下一条可证伪编码假设

候选：保持每位置每tick实际非零支持和原幅值字母表，交换少量已有事件的幅值；离线用**全部竞争类别**和T1/T2共同约束学习轻量选择规则。比较同容量的当前状态控制、随机分组、真实结构条件与结构置换条件，在线只读当前状态和缓存图统计。需要先证明硬事件oracle在固定总事件、幅值成本/投递成本下确实存在跨任务一致的全类margin收益，再训练可部署规则。若真实结构不优于同容量非图控制，去掉Graph-aware主张。

这是待检验假设，尚未实施或证明新颖。当前普通任务目标的Gate2联合准入失败，不以该假设为借口跳过门控或扩大搜索。
# Additional causal temporal-history check

`test_prefix_network.py::test_identical_integrated_input_and_event_counts_do_not_fix_next_layer_decode` executes the frozen scalar source with theta1 and zero initialization. Currents `(0.01,1.58)` and `(1,-0.4)` both integrate under the author's decoder to `u1+u2/2=0.8`, and both emit exactly two nonzero signed/multilevel events. Yet their decoded outputs are respectively0.5078125 and0.75. A fixed class boundary0.6 distinguishes them. This is a controlled source-recurrence counterexample, not an observed dataset accuracy estimate: equal integrated input, alphabet, T and event count do not determine a nonlinear neuron's decoded output. Temporal history matters independently of reconstruction norm.

The first-site two-word diagnostic further enforces bitwise identical **local T2 output** on128 real training graphs per task, using both verified orientations. Downstream costs and classes can still differ. Halving first events reduces local amplitude but increases whole-network events; reversing the word can reduce some CIFAR costs but is inconsistent across tasks. Thus a local prefix guarantee cannot justify a whole-network cost or decision guarantee. These are training-only mechanism diagnostics; learned-selector validation results are in `analysis/temporal/`, separately labeled as exposed validation evidence. T1 local error arrays are placeholders; only T2 equality is asserted.
