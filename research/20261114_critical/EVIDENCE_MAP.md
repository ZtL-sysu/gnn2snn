# 论文故事与证据映射

|研究发现|权威证据（相对 research/）|状态|论文角色|论点及边界|
|---|---|---|---|---|
|九教师可复用|20261114_story/audit_baselines.md; gated_baseline_audit.md|已验证存档|Setup|训练不是新算法；Gated混合模拟门控|
|T4/T8较高保真|20261114_story/analysis/confirmation/per_seed.csv|已验证|Motivation/Baseline|既有差分编码，不是新贡献|
|T1明显性能缺口|20261114_story/analysis/frozen_test/aggregate.csv|已验证、历史暴露测试|Motivation|MN71.44/CIF46.80是测试均值，不与本轮val混用|
|全位置局部校准有效|同上 node行|已验证|强基线|MN78.50/CIF49.92不归功新方法|
|旧logits目标无稳定优势|20261114_story/REPORT.md及paired汇总|限当前参数与优化器|Ablation/Discussion|不能推及所有图级目标|
|首位置理想恢复较大|20261114_story/analysis/ideal/aggregate.csv|仅理想化诊断|Motivation|54.1%/23.5%是教师预测不一致缺口关闭比例，不是准确率缺口恢复比例；不是可部署方法|
|当前GPU无加速|20261114_story/costs/及COST_DEFINITIONS.md|已验证实现限定|Efficiency|有符号多级事件，密集后端，无硬件能耗结论|
|消息预激活扰动恒等式|本轮objective.py与test_objective.py|理论条件明确/待实现验证|Method设计依据|不宣称新定理或Gated适用|
|消息保持独立分类收益|本轮runs/待生成|待确认|有证据才进Main Results|必须超过局部及图内置换/特征控制|

## 本轮已完成证据的论文判断
|发现|证据|状态|论文角色|含义|
|---|---|---|---|---|
|首位置偏置未超越全位置局部|analysis/endpoints.csv中screen/optimizer_check|两任务各单教师初筛|Ablation|理想敏感性≠当前可校准性；不推广成所有位置方法无效|
|Surrogate可使首位置硬训练MSE上升|各run/*_training.json|已验证实现行为|Optimization limitations|不能把梯度有限当成优化正确|
|硬目标坐标优化仍无结构独立分类优势|runs/hard_probe及paired.csv|有界粗网格诊断|Mechanism|消息MSE改善并不必然提高分类；不声称全局最优|
|八幅值局部码本改善MNIST T1|runs/book_screen/MNIST41|仅单教师筛选|候选Motivation|82.23%相对相同筛选集全局部75.00/78.125，不能混用历史测试78.50|
|码本结构权重无独立增益|runs/book_screen及paired.csv|两任务各单教师|Ablation|传播对角代理尚不足；未测试完整交叉项优化|
|完整消息交叉项修正仍劣于局部|runs/full_book_screen及paired.csv|完整硬目标重算检查、各任务单教师|Ablation/失败边界|T1差值−1.171875/−1.3671875pp；不再把遗漏交叉项当当前失败的充分解释|

当前两条主要方法候选均未满足跨任务、强匹配对照和创新性的联合要求。没有已确认的三项新方法贡献；可保留的三项研究产物是首位置可实现性审查、硬优化对照和字母表容量诊断，尚不能替代算法贡献。

后续固定机会预算补偿：`analysis/rescue_endpoints.csv`、`rescue_paired.csv`及`allocation_mechanism_cost.csv`。实际图消息MSE降低优于局部选择，但分类未通过两任务强基线/非图控制/T2联合门槛，计时增加。适合Mechanism和Efficiency，不支持成功结构算法主结果。源分位数对照为GCN适配，原ImageNet配置不直接承担方法优越性基线，详见SOURCE_BASELINE_AUDIT与P3_BASELINE_REVIEW。

旧弱教师、口径不齐的历史SOTA表不进入主结果；只保留来源可核验、口径一致部分。旧论文完整内容和新文献仍需本轮系统审查。当前仅候选结构，无已证实新算法贡献。
