# 理论与方法谱系

## 固定图两层转换与线性误差恒等式

早期图模型使用固定非负归一化邻接A和线性图读出R。隐藏激活为H=ReLU(AXW1+b1)，图logits为B=RA(HW2)+b2。在同一读出和偏置约定下，B_hat-B=RA((H_hat-H)W2)是线性代数恒等式。它有助于解释误差传播，但不能单独证明H_hat接近H、分类正确、基线更强或成本更低。

若隐藏事件为二进制脉冲，正确解码H_hat=theta·sum(spike)/T，输出偏置按相同一次／平均等价约定处理。旧稿使用未乘theta的count解码，theta≠1时不能保持其声称的ANN/SNN关系；一个针对错误量的近零恒等残差不修复这个问题。正齐次gauge缩放须和转换theta一致，并核验ANN logits不变。

## 初始电位、偏置、margin与排程

历史探索包括零初始化、半阈值初始化、学习初始化、偏置修正、margin与mask；精简版保留关键成果与其余路线的日志总结。不同网络训练、不同读出、不同校准预算不能伪装成只改变一个机制。旧offline_balanced_count预先决定总count后均匀排布，是离线方法；新修正版明确另列causal_IF_zero和causal_IF_half。若相同T得到相同count/logits，则名字或排程本身不构成独立准确率优势。

## 源码固定的差分编码适配

使用公开DCGS仓库固定commit 5e0e4c73f1d0e8e17d81befa24ff947deec1f4be的MTH_diff/MTH_diff_line。差分输入可写为e_t=t(a_t-a_(t-1))，前缀解码为sum(e_t/t)。需要独立检查zero-latch、active tick、重置、bias进入时点与readout。这里的带符号多幅度事件不是二进制IF脉冲。

项目的固定GCN与GatedGCN适配不是完整作者converter/threshold iteration原样复现。作者架构特定路径与generic路径的放置规则不同；项目采用完整train的histogram校准。源文件、广播、调用数和图算子语义均有专门检查。参见`dcgs-threshold-procedure-source-note-v1.md`、`centered-residual-prior-art-v1.md`。

## 动态GatedGCN的困难

每层含A/B/D/E节点投影与C边投影，edge logit为C(e)+D(h_src)+E(h_dst)，sigmoid后在目标节点按特征归一化，加权聚合B(h_src)。边状态也有BN/ReLU/identity residual更新。门、乘法、分母和归约取决于当前状态，不能直接拿固定线性A的误差恒等式宣称所有动态算子都已覆盖。

## affine-base/shared-consumer候选

把identity residual状态写为h_l=h0+q_l、e_l=e0+r_l，q/r为非负ReLU残差增量的原始累计和。generic编码所有原始layer input；first_analog保持首层精确，再编码后续raw input；centered编码后续q/r。各模式都保留raw graded skip，不能把decoded q/r替换进skip。节点代码流在同层A/B/D/E之间共享，边流供C；分类head保持相同模拟MLP。

该重写可让Wh0+b从低维原始特征融合、静态基项缓存，避免旧branch-fanout的3/2/1重复投影。它仍需要实际cache/backend、成本账与匹配基线。三模式／两尺度及共享theta控制区分首层精确、consumer共享、坐标原点与校准范围，防止把控制差异误归因到新机制。

主要推导：`DERIVATION_PACKAGE_v2.md`；源码与CPU检查：gated_affine_base_v1.py、check_gated_affine_base_v1.py。新颖性界限见`affine-incremental-prior-art-v1.md`：差分传播、仿射搬移和共享缓存已有先例，本项目的特定组合仍需证据。

## normalized-gate误差分析

将epsilon分母项解释为消息为0的dummy质量；利用log-sigmoid的Lipschitz性和总变差界，得到消息误差可由输入消息扰动与消息范围×tanh(eta/2)控制。它避免直接1/epsilon型估计在小门值下的爆炸，但仍是带假设的实数算术界，不是浮点或硬件证书，也不证明有用的缓存命中率。

`stable-normalized-gate-bound-v1.md`和持久化NumPy脚本含固定seed20,000随机例与24边界例检查。误差递推仍须计BN、residual与head，不得把局部gate界当最终分类保证。
