# 目标死磕状态账本

> 账本路径：本 run root 下 `GOAL_LEDGER.md`（唯一进度载体）。R1 建立于 2026-11-13T14:59:20Z。历史记录只追加不改写。

## 1 目标档案

| 项 | 内容 |
|----|------|
| 目标陈述 | 完成 `PROTOCOL-v3.md` + 分类头补充 + CURRENT_SCOPE 中 E0–E6 全部实验与公平对照、验证集上的书面 Story 判定、六节全新终稿、引用核查与当前稿独立 CCFA≥7/10；并同步到本仓库 agent 分支 |
| 背景与动机 | 用户 2026-11-13 22:52 +0800 指令：读取 GitHub gnn2snn 最新进展与协议，按科研流水线标准推进，代码本机 3060 跑，按 research-github-sync 传仓库，目标为完成协议中的所有工作 |
| 边界 | 不改研究方向、不缩数据子集、不伪造基线/效率、不远程 GPU、不擅自合并 main、不 force push、不发布 Release、不上传凭证/私人对话/大失败镜像；不修改旧证据哈希 |
| 时间尺度 | 心跳期（跨会话续跑；已启用，`FREQ=HOURLY;INTERVAL=1;BYMINUTE=0,15,30,45` = 每 15 分钟，2026-11-14 07:32 用户要求调小后更新） |
| 资源约束 | 本机 RTX 3060 12 GiB，Conda dyg-cold；contract v4 计算上限 E1≤80 / E2≤200 / E4≤80 / E3+E5≤40 / E6≤100 GPU 小时；单 GPU 单作业 |
| 判定者 | 系统指标（冻结协议的 .3pp + 完整成本门槛）、文档产物（六节新稿与审计）、独立评审（CCFA≥7/10） |
| 确认档位 | 授权档：用户已一次性授权「本机跑代码 + 增量 push 到 agent 分支 + 完成协议工作」。清单外操作（merge/force push/远程计算/发布）仍需逐项确认 |
| 运行模式 | 会话内死磕 + 每 15 分钟心跳续跑（ACTIVE；GPU 空闲即必须启动真实作业，见 GPU_QUEUE.md） |

## 2 手段弹药库

| 条目 | 路目 | 能力描述 | 可用状态 | 调用路径 | 风险 | 白名单 |
|------|------|---------|---------|---------|------|--------|
| A1 | 内置能力 | 本地文件读写、Bash/PowerShell、Git worktree/分支 | 可用 | 本会话工具 | 低 | PASS |
| A2 | 内置能力 | Conda `dyg-cold` Python 执行（torch 2.11+cu128, PyG 2.8, CUDA） | 可用（已实测 CUDA=True） | `conda run -n dyg-cold python -X utf8`（项目本机环境） | 低 | PASS |
| A3 | 内置能力 | `f-research-v1` 流水线状态校验器 | 可用（`--before P4` PASS） | `skills/f-research-v1/scripts/pipeline_state.py` | 低 | PASS |
| A4 | 内置能力 | `research-github-sync` 清单脚本与自检 | 可用（7 项测试 PASS） | `skills/research-github-sync/scripts/checkpoint_manifest.py` | 低 | PASS |
| A5 | 可检索能力 | 已安装的 ccf-* / nature-* / paper-* 等科研子技能 | 已安装，尚未在本轮调用 | Skill 工具 | 低 | PASS |
| A6 | 外部资源 | 官方 PyG GNNBenchmarkDataset 本机缓存（5/5 哈希匹配） | 可用 | 该数据根由 `GNN2SNN_DATA_ROOT` 指向 | 低 | PASS |
| A7 | 外部资源 | 钉死 DCGS 源码 checkout（commit 精确匹配） | 可用 | `DCGS_REPO` 指向本机 checkout | 低 | PASS |
| A8 | 外部资源 | 9 个教师权重与 verified-evidence.zip、seed41/42/43 教师目录 | 可用 | 主工作树原路径只读 | 低 | PASS |
| A9 | 外部资源 | GitHub `ZtL-sysu/gnn2snn` 远端读写 | 可用（fetch 成功；push 待核对） | `git push` 到本 agent 分支 | 中 | PASS（仅 agent 分支） |
| A10 | 可组合编排 | 定时自动化心跳续跑 | **已启用 ACTIVE**：每 15 分钟，按 GitHub 已推送进度、强制 GPU 满载队列 | automation_update / GPU_QUEUE.md | 中 | PASS |
| — | 已剔除 | novix exp-4090 远程 GPU | 用户最新指令禁止 | — | — | 剔除：仅本机 3060 |

## 3 成功标准证据清单（锁定，来自冻结协议；不得修改）

| 编号 | 标准（可观测事实） | 判定者 | 反例排除 | 当前状态 |
|------|-------------------|--------|---------|---------|
| S1 | E0 冻结复现：seed41 全 5000 验证 ANN70.66/DC4 70.72/DC8 70.66，规范数组逐位一致，历史 T8 一处类别漂移如实披露 | 系统指标+报告 | 放宽容差、只报聚合、宣称逐样本历史复现 | PASS（v3 已验收，旧 v1/v2 失败保留） |
| S2 | E1 七行（+模拟头对照、共享 theta 附加对照）× T1/2/4/8 × seeds41/42/43 全部完成，逐样本证据与哈希齐全 | 文档产物 | 只跑三 affine 模式/单 seed 即宣称 E1 完成 | FAIL（R5：10 个行族 ×4T ×3seed 共 **120 份逐样本 NPZ 与哈希齐全并独立核验 PASS**——ANN 12、既有六行 72、新增残余三行 36；seed41 六站点 DC-adapt T4/T8 与 E0 canonical 逐位一致，shared-theta 非平凡别名。**仍不判 PASS**：R7b 已补 E3 完整成本账与 E5 37/37 匹配墙钟，但成本腿与墙钟腿均未通过、9 项 unresolved 保留；协议 E1 每行还要求 per-site events / complete known cost plus unresolved items / ms/graph，不得把精度或逐样本证据齐全等同于整行完成） |
| S3 | E1 门槛：T4 与 T8 三 seed 验证均值距最强转换对照 ≤0.3pp **且** 完整已知成本或匹配墙钟更低 | 系统指标 | 只用精度、省略成本项、用未实现零跳过当节能 | FAIL（R5 按补充协议「ALL conversion controls，含两个 DC readout 版本，排除 ANN 与候选」重算：T4 centered 0.705267 vs 最强 `centered_shared_theta` 0.705933，差距 **0.0667pp**；T8 centered 0.705667 vs 最强 `dc_adapt_six_site` 0.706133，差距 **0.0467pp**；均 ≤0.3pp，精度条件单独仍满足。**此数字取代 R4 的 0.0333pp**（那是缺行时的不完整对照集）。**R7b2 已用真实实测闭合证据缺口并重算**：精度腿仍单独满足；成本腿 false（T4 AC 69,174,673 vs 68,964,103 pJ/图，T8 102,182,276 vs 44,065,575；保守 MAC 亦不更低）；墙钟腿 false（T4 6.2449 vs 6.2130，T8 12.0486 vs 3.1315 ms/图）；9 项 unresolved 保留 → `verdict: E1 NOT PASSED (method advantage not established)`。状态由 NOT EVALUABLE 变为**有证据的 NOT PASSED**，协议「neither → Story B」，Story 书面判定仍 DEFERRED。不得用精度单独通过、不得用事件数冒充节能、不得因结果不好降门槛） |
| S4 | E2/E4/E6 忠实对照：NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN；CIFAR Gated 3 seeds、MNIST Gated、DGCNN ModelNet40 3 teachers | 文档产物 | 假定算子适配失败、跳过弱结果、用公开数字冒充实测 | FAIL（R8：**NaiveIF 首例已完成并发布**——3 seeds × T{1,2,4,8} × 全 5000 官方验证图，stage-1 阈值仅来自全部 45000 label-free 训练图，插桩保真差 0.0、canonical 逐位门禁 PASS、12 份逐样本 NPZ + 3×16 行指标。真实弱结果如实登记：acc 0.0972–0.1680，严重退化且未随 T 可靠恢复，不假定、不跳过、不冒充实测。**仍不判 PASS**：其余 6 个忠实对照、MNIST Gated、DGCNN ModelNet40 全部未做） |
| S5 | E3/E5 完整成本账（各操作项、AC/保守 MAC 双约定、3 次重复中位墙钟）、真实数据 gate-bound 与误差、falsifier | 文档产物 | 只报事件数、只报宽松界、芯片能耗声明 | FAIL（E3 成本账 108 行/218 指标与 E5 37/37 墙钟已实测并发布。**R8 更正此前「v3 GPU 作业未启动」的过时描述**：falsifier v1/v2 的两次实现崩溃已留证并由 v3 修复；v3 已真实 GPU 跑完 **536.3 s**，`e5_gate_bound_v1.json`（SHA `1fb6b433…`）+ 120 行 `falsifier_METRICS.csv` 已发布——保真 `checked_pairs 160` / `frozen_vs_instrumented_max_abs_diff 0.0`、fresh-vs-stale 与 raw-vs-decoded falsifier 判定齐全、degree/sharpness 子集与 message/layer/logit 误差齐全。**机制腿证据缺口已闭合，但 S5 仍 FAIL**：E3 成本账仍保留 9 项 unresolved（「Cost with unresolved omitted work cannot pass」）、S4 忠实对照未齐、无独立评审；且该界是局部实数算术消息界，宽松界如实报宽松，不是浮点证书 / 精度保证 / 芯片能耗声明） |
| S6 | 仅基于验证集的书面 Story 判定，之后新测试集一次确认并披露历史暴露 | 文档产物 | 用测试集调参/选点 | 待验证 |
| S7 | 全新六节稿件 + 完整证据 E0–E6 + 引用≥36 篇且近 3–5 年占多数 + 去重审计 | 文档产物 | 用旧稿冒充、想法评分代替 | FAIL |
| S8 | 当前完整稿经实质独立 CCFA 评审 ≥7/10 | 独立评审 | 同角色自评、空评阅、历史稿评分 | FAIL |
| S9 | 过程检查点按 research-github-sync 增量同步到 agent 分支且远端 SHA 核对 | 远端确认 | 仅本地 commit 就报「已上传」 | 进行中（R1–R3 + R4b 检查点均已 push 并 `git ls-remote` 核对；R4b 证据提交 `5ece63c582f6639e177f94fd64434d279e427d97`、远端确认写回提交 `6d6a62173583c0931556a2dafdbad0a6b3edf776` 均已核对本地 HEAD == 远端 SHA；R5 证据提交 `1fd6437e4c4c234704b125aeaeaba0fb0e982fc3` 已 push 且 `git ls-remote` 返回同一 SHA，81 个明确文件、72 产物清单 `verify` PASS、上传预检 PASS（80 文件 / 48 NPZ / 0 problems）；该远端事实的写回提交 SHA 将在下次唤醒记录；协议整体未完成，故 S9 不判 PASS） |

## 4 里程碑链

| 里程碑 | 绑定证据 | 依据 | 状态 |
|--------|---------|------|------|
| M1 恢复与环境复核 | S9 | 独立 worktree、协议哈希、数据/源码/环境校验、首个检查点远端确认 | 进行中 |
| M2 E1 控制评估器 | S2 | 新 evaluator + 测试通过 + 冻结阈值映射可复现 | PASS（4 项合成测试、双校准预检、官方 5000 图实跑并跨运行逐位复现） |
| M3 E1 全网格 | S2,S3 | seed41→42/43 固定网格全部行、逐样本 NPZ 与成本账 | 进行中（R5：10 个行族 ×4T ×3seed 共 120 份逐样本 NPZ 与哈希齐全且核验 PASS，全行描述性汇总与判据重算完成；**每行完整已知成本账、ms/graph 与匹配墙钟仍待**，故 M3 未 PASS） |
| M4 E2/E4/E6 对照活动 | S4 | 全部忠实对照与跨任务教师/转换实测 | 未开始 |
| M5 E3/E5 成本与机制 | S5 | 完整操作计数、墙钟、falsifier | 未开始 |
| M6 Story 判定与新测试确认 | S6 | 书面判定 + 冻结后一次测试 | 未开始 |
| M7 稿件与引用 | S7 | 六节新稿、36+ 引用、审计 | 未开始 |
| M8 独立评审 | S8 | 独立 CCFA≥7/10 | 未开始 |

## 5 当前策略卡

| 项 | 内容 |
|----|------|
| 当前策略 | 先补齐 E1 全部行与逐样本证据（R5 已完成），再以 E3/E5 完整操作成本 + 3 次重复中位墙钟闭合 S3 双门槛；成本未闭合前不给 P4 加 pass、不写 Story |
| 核心假设 | 本机 3060 + dyg-cold + 本机官方数据/教师/DCGS 足以复现既有 E0/E1 结果并继续 E1；冻结阈值与 scale1 不需重算。R5 已验证历史六站点 DC-adapt 与 shared-theta 可由冻结 helper 忠实复现，但该假设对 E2/E4/E6 的算子适配尚未验证 |
| 动作清单 | ① **E3/E5 完整操作计数（全部操作组、AC/保守 MAC 双约定、pJ 算术代理）+ 同 GPU/batch128/顺序/warmup 的 3 次重复中位墙钟 + 真实数据 gate-bound 与 message/layer/logit 误差、raw-vs-decoded 与 fresh-vs-stale gate falsifier（S3 前置，最高优先）**；② E2/E4/E6 忠实对照（NaiveIF/官方 DCGS/QCFS/FTBC/SNN Calibration/One-Spike/直接训练 SpikingGCN）与 MNIST Gated / DGCNN ModelNet40；③ 验证集-only Story 判定 + 冻结后一次新测试确认；④ 六节全新终稿 / ≥36 篇经核查引用且近 3–5 年占多数 / 去重审计 / 当前稿独立 CCFA≥7/10 |
| 敏感标记 | 无新增（push agent 分支在授权档清单内）；merge/远程 GPU/发布为清单外 |
| 连败计数 | 0（R5 三个 GPU 作业全部 `rc=0`。上一轮 `7M53yo` 的三次 CLI 默认参数失败属**已修复的旧策略**，失败证据完整保留并去标识公开，不累计为当前策略连败；重启前又拦下一处 `rel()` 外部路径必现缺陷，未形成 GPU 失败） |
| 真实作业句柄 | R4：唤醒核实无存活作业（GPU 0%/955MiB、无 python 训练进程）→ 顺序启动 4 个作业，全部 exit 0：seed42/43 校准（89.49/89.04 s）、E1 seed42（697.4 s）、E1 seed43（11m49s）；18:04Z 复核无存活作业。R4b（18:20Z）：`git fetch` 后 GPU 0%/955MiB、PowerShell CIM 查询 0 个 python 进程 → **无存活作业，本轮未启动任何 GPU 作业**。**R5（20:11Z）：`git fetch` 后 GPU 0%/955MiB、CIM 精确过滤 0 个 python/conda 进程 → 启动单一后台驱动 `LYwybZ`，seed41/42/43 顺序全 5000 验证分别 251.8/252.3/252.8 s，全部 `rc=0`，20:24:14Z 结束；20:24Z 复核 GPU 0%/955MiB、无存活 Python 进程。** 恢复/复现命令见 HANDOFF R5 段与 `20261113T202500Z-e1-fullrow-residual/CHECKPOINT.md` 第 9 节。**R6/R6b/R6c（本轮）**：23:37Z 心跳启动 `ThE3rS`(PID 27348) 与 23:51Z 本会话 `jvgzz3`(PID 15052) 并发占用同一 3060，两者计时因 GPU 争用失真且旧脚本有「阈值计时内重建」缺陷 → 全部终止、证据归档 `logs/e3e5-latency-concurrency-incident-v1/`、数值作废；00:04:49Z 启动**唯一**干净作业 `TsI47h`（`latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume`，逐配置增量 `e5_latency_v1.partial.jsonl` 可续跑），实测单一进程链、GPU 89%/5130 MiB。`e5_falsifier_v1.py` 已就绪但 GPU 运行排队中。恢复命令与「在跑不得重启」规则见 `GPU_QUEUE.md` 顶部与 `20261114T000800Z-e3-cost-e5-clean-latency/CHECKPOINT.md` 第 8 节。**R6d（01:36–01:42Z）**：`TsI47h` 约 00:14Z 被环境终止（7/37 配置、无 traceback、无报告），中断快照归档 `logs/e5-latency-interruption-v1/`，7 条干净增量有效；01:40:39Z 启动唯一串行驱动链 `v10FQp`（driver PID 5776 / latency PID 11560，`gpu_queue_driver_v1.py` 串行 latency resume → aggregate → falsifier run → verify），宽匹配确认单一 latency 进程，实测 GPU 90% / 5130 MiB / 112.24 W；在跑期间绝不重启、绝不另起第二个 GPU 作业。恢复命令见 R6d `HANDOFF.md` 与 `GPU_QUEUE.md` 顶部。 |

## 6 执行轮与换道日志

| 轮次 | 策略 | 动作与结果 | 证据核验 | 败因归类 | 换道决策 |
|------|------|-----------|---------|---------|---------|
| R1 | 恢复优先 | fetch 全部远端，定位协议与交接；建独立 worktree/分支；协议/数据/源码哈希与 GPU/环境实测；合成检查 PASS、单测 2 passed；P4 前置校验 PASS | S1 PASS（继承），S9 进行中，S2–S8 FAIL/待验证 | 无失败；S2 缺口源于此前暂停中断 | 继续 M2（实现控制评估器） |
| R2 | 补齐被中断的 M2 | 实现双校准绑定的对照评估器 + 4 项合成测试通过；预检 18 个绑定输入 OK；启动首个官方全 5000 图控制网格（3 行×4 T，seed41，scale1） | S9 首个检查点远端确认；S2 仍 FAIL（作业未完成）；S3–S8 未动 | 无失败；CLI 导入顺序问题已修复并重测 | 等作业完成后发布结果检查点，继续 M3 |
| R3 | 固定网格实跑 + 证据核验 | 作业 exit 0，5000 图 12 行逐样本证据；教师 0.7066 与前后哈希一致；与既有独立 generic 运行逐位一致（4/4 T）；生成 36 行固定列指标 | S2 部分推进（seed41 新 3 行实测）但仍 FAIL；S3 仍 FAIL（成本未测）；S9 三个检查点远端确认 | 无失败 | 继续 M3：先做 42/43 冻结校准，再不变重跑，并补共享 theta 与完整成本 |
| R4 | 不变流程多 seed 补网格 | 冻结输入门禁全 PASS；进程/GPU 判无存活作业后顺序跑 4 个本机 GPU 作业全部 exit 0：42/43 冻结全训练无标签校准（45000×2 遍，复用 frozen `calibrate_loader`）、42/43 E1 六行 ×4T 全 5000 验证（各 24 逐样本 NPZ）；流程等价性交叉验证 PASS、逐样本核验 42/43 均 PASS；跨 seed 汇总 336 行固定列指标 | S2 大幅推进（72 组逐样本证据齐全）但仍 FAIL（缺 ANN/历史 DC-adapt 汇总/共享 theta）；S3 精度条件满足但双门槛 NOT EVALUABLE（成本/墙钟缺）故仍 FAIL；S4–S8 未动；S9 待本轮 push 远端确认 | 无失败（连败 0） | 继续 M3 收尾：共享 theta + 历史六站点 DC-adapt + ANN 行，再 E3/E5 完整成本与 falsifier（S3 前置），之后 E2/E4/E6 |
| R4b | 证据完整性修复 + 清单发布（非科学推进） | 唤醒门禁全 PASS（12 项冻结输入 / contract v4 / P4 running / 0 个 python 进程 / 0 GPU 作业）；发现未发布的 R4 23 产物草稿清单因 4 个新增脚本去标识而失效，按不可变规则不覆盖，改新建 74 产物权威清单（含全部 48 个逐样本 NPZ）并 `verify` PASS；新增冻结门禁与上传内容检查两个只读工具；重验流程等价性、E1 预检、10/10 公开副本均 PASS | S2/S3 状态不变（仍 FAIL），S4–S8 未动；S9 本地清单 verify PASS，仍待远端 SHA 核对；**E1 判据 NOT EVALUABLE 不变，P4 仍 running** | 无失败（连败 0）；工具层 `tasklist /FI` 在 Git Bash 被路径改写，已用 PowerShell CIM 替代；`conda run -c` 多行脚本报 NotImplementedError，已改单行 | 不把打包当科学进展；下一轮直接进 M3 收尾（共享 theta / 历史六站点 / ANN 行 / 完整成本与墙钟） |
| R5 | 补齐 E1 残余行 + 全行判据重算 | 唤醒门禁 PASS（12 项冻结输入 / 5/5 官方 processed 哈希 / contract v4 / P4 running / 0 个 python-conda 进程 / GPU 0%955MiB）；保留上一轮三次 CLI 失败证据并去标识公开，修复默认参数解析与 `rel()` 外部 DCGS 路径两处缺陷；单一后台作业 `LYwybZ` 顺序跑完 seed41/42/43，各 5000 图、3 行 ×4T，全部 `rc=0`（251.8/252.3/252.8 s）；新增 36 份逐样本 NPZ，`verify` PASS；seed41 六站点 DC-adapt T4/T8 与 E0 canonical 逐位一致；ANN 上界行 12 NPZ 独立复验；`aggregate_e1_fullrows_v1.py` 汇总 10 行族 ×4T ×3seed = 120 份逐样本证据、40 描述行、560 固定列指标；新建 72 产物权威清单 `create`+`verify` PASS | **S2 大幅推进但仍 FAIL**（120 份逐样本证据与哈希齐全并核验 PASS，但每行完整已知成本/未决项/ms/graph 仍缺）；**S3 仍 FAIL 且数字被诚实重算**：按 ALL 8 转换对照（含两个 DC readout 版本）取最强，T4 差距 0.0667pp、T8 0.0467pp，取代 R4 的 0.0333pp，精度条件仍单独满足但双门槛 NOT EVALUABLE；S4–S8 未动；P4 仍 running；S9 本地清单 PASS、远端 SHA 待写回 | 无 GPU 失败（连败 0）；三次 CLI 失败为旧策略且已修复，证据保留；重启前静态复核拦下第二处必现缺陷，未形成 GPU 失败 | 不因精度条件满足而给 E1 加 pass；下一轮最高优先 E3/E5 完整操作成本 + 3 次重复中位墙钟 + gate-bound/误差 falsifier，再 E2/E4/E6，再 Story/稿件/引用/独立评审 |
| R6 | E3 成本账 + 启动 E5 墙钟 | 修复 `e3_e5_v1.py` 的 120→108 行硬编码；`precheck` PASS；`cost` 生成 108 行 / 218 指标成本账；23:37Z 启动全 5000 图 3 重复墙钟作业 `ThE3rS`；建 `GPU_QUEUE.md` 与 15 分钟心跳 | S5 部分推进（成本账完成）；`aggregate` 未跑，E1 仍 NOT EVALUABLE；S9 R6 检查点已发布 | 无 GPU 失败 | 继续监督墙钟，完成后 `aggregate` |
| R6b | 清单证据完整性修正 | 发现 R6 清单误把 4 个活文件纳入不可变清单，按 R4b/R5 惯例新建只绑定 5 个不可变产物的 R6b 清单，`create`/`verify` PASS | 无科学变化；S2/S3 仍 FAIL，P4 running | 非科学失败（清单设计） | 后续清单只绑不可变证据 |
| R6c | 静态复核 + 并发事故处置 + 干净重跑 | 冻结门禁/`--before P4` PASS；GPU 前拦下导入顺序、`阈值计时内重建`、恒真式代表性检查、storage 声明不一致四处缺陷，修正前成本账带 SHA-256 保留且 `cost_METRICS.csv` 字节不变；全 37 配置 dry-run PASS；发现并终止两个并发 GPU 作业，污染证据归档；启动唯一干净作业 `TsI47h`（全 5000 图 / 1 warmup / 3 repeats / 增量可续跑）；新增 `e5_falsifier_v1.py` 且 precheck PASS | **S5 仍 FAIL**：E3 成本账完成、E5 墙钟在跑、gate-bound/falsifier 未实测；E1 仍 `NOT EVALUABLE / E1 NOT PASSED`（`aggregate` 未跑）；S2/S3/S4/S7/S8 FAIL、S6 待验证、S9 进行中；P4 仍 `running` | 并发事故属协调/环境中断，非 GPU 科学失败；连败计数不累计；**窄进程名查活漏报已如实记录并改宽匹配** | 不重启在跑作业；完成后 `aggregate` → E5 falsifier GPU 单作业 → 结果检查点 → E2/E4/E6 |
| R6 | 按 GitHub 进度补 E3/E5，恢复 GPU 满载 | 用户要求「按已上传进度推进 + 3060 拉满 + 间隔调小」；远端 `9865101` 后 3 小时空转。修复未上传草稿的 120→108 硬编码，precheck PASS、E3 成本账完成（108 行 / 218 指标 / 双约定 / unresolved 保留）；启动 E5 全 5000 图 37 配置 ×3 重复 GPU 长作业 `ThE3rS`（PID 27348，GPU 84% / 3824 MiB）；心跳改每 15 分钟并建立 GPU_QUEUE.md | S2/S3 仍 FAIL（aggregate 未跑、E1 NOT EVALUABLE）；S9 本检查点待远端核对；S4–S8 未动 | 3 小时空转属**条件不备**（草稿脚本阻塞且未上传），已补弹药库 A10/A11 并换道为「按 GitHub 进度 + 队列满载」 | 继续：作业在跑则监督、结束后 aggregate → 结果检查点 → E2/E4/E6 |
| R6d | 环境中断恢复 + 单一串行驱动满载 | 以远端 `a2f59ac`（R6c）为唯一进度依据；冻结门禁 12 项 PASS / contract v4 / P4 running；宽匹配查活 0 进程、GPU 0%/955MiB。发现 `TsI47h` 于 00:14Z 被环境终止（7/37 配置、无 traceback、无报告），中断快照 + SHA256 + `INTERRUPTED.json` 归档，7 条干净增量判为有效并与并发作废数值区分；新增 `gpu_queue_driver_v1.py`（py_compile / dry-print / 去标识 PASS），按**完全相同** `--batches 40 --warmup 1 --repeats 3 --resume` 续跑并串行排队 `aggregate` → falsifier `run` → `verify`，失败不整轮停摆；启动唯一链 `v10FQp`（driver 5776 / latency 11560，01:40:39Z），实测 GPU 90%/5130 MiB/112.24 W | **科学状态零变化**：三份结果报告均未生成、`aggregate` 未跑、E1 仍 `NOT EVALUABLE / E1 NOT PASSED`、P4 `running`、S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中；本轮只恢复真实 GPU 作业并留证，不把「启动/归档/写回」当科学验收 | 82 分钟空转属**条件不备**（会话边界终止作业），非 GPU 科学失败；连败计数保持 0 | 监督 `v10FQp` 不重启；完成后收集延迟/判据/falsifier 结果并发布检查点，**同一次唤醒**启动 E2/E4/E6 首个 GPU 作业；若链再次被终止，重复「归档 + 不变参数 resume」 |
| R7 | 根因修复：会话外托管 + 第三次中断恢复 | 以远端 `7bd7558f`（R6d 写回）为唯一进度依据；冻结门禁 12 项 PASS / contract v4 / P4 running / DCGS `5e0e4c73…` 实测匹配 / 四脚本 compile + dry-print PASS；宽匹配 0 进程、GPU `0 % / 955 MiB / 12 W`。归档 `v10FQp` 第三次环境中断（9/37、无 STEP_END、无 status.jsonl、空转约 75 分钟），快照 + SHA-256 + `INTERRUPTED.json` 于 `logs/e5-latency-interruption-v2/`，9 条干净增量判为有效。**换道修复连续三次同因中断的根因**：新增 `gpu_queue_launcher_v1.py`，用无触发器计划任务 `GNN2SNN-E5-Driver-R7`（Triggers=0 / IgnoreNew / 24h）托管同一驱动，实测父进程为 `svchost.exe -k netsvcs -p -s Schedule`，跨会话存活；03:08:04Z 起 **GPU 88–92 % / 5155–5159 MiB / 110–118 W**，唯一 `latency` 进程。真实增量 **9/37 → 12/37**：`generic T1` 1.9336、`T2` 3.7015、`T4` 7.1592 ms/graph（各 3 重复中位、全 5000 图）；派生 24 行固定列快照、7 产物清单 `create`+`verify` PASS | **科学状态零变化**：三份结果报告未生成、`aggregate` 未跑、E1 仍 `NOT EVALUABLE / E1 NOT PASSED`、P4 `running`、S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中；本轮为根因修复 + 真实 GPU 增量，不把启动/归档/写回当验收 | 第三次环境中断属**条件不备**（会话边界杀死后台进程），非 GPU 科学失败，连败计数保持 0；已换道为「计划任务会话外托管」而非重复原策略 | 监督计划任务不重启；`DRIVER_END` 后收集三份报告发布结果检查点，**同一次唤醒**启动 E2/E4/E6 首个 GPU 作业；若链消失，先宽匹配查活再 `Start-ScheduledTask` 幂等续跑（参数不变） |
| R7b | E5 完整墙钟 + E1 有证据重算 + 失败修复换道 | R7 会话外托管驱动**完整跑完 48.6 分钟未中断**（根因修复实证有效）：`01 latency rc=0` 产出 **37/37 配置** `e5_latency_v1.json`（`6887d2ba…`，全 5000 图 / batch128 / 1 warmup / 3 重复中位 / 冻结阈值循环外构造 / 代表批次独立校验）；`02 aggregate rc=0` 产出 `e3_e5_criterion_v1.json`（`a474c1f0…`）。**E1 由 `NOT EVALUABLE` 变为有证据支撑的 `E1 NOT PASSED`**：精度腿 true（T4 0.0667 / T8 0.0467 pp），成本腿 false（T4 AC 69174673.484 vs 68964102.8846；T8 102182275.9832 vs 44065574.72696 pJ/图），墙钟腿 false（T4 6.2449 vs 6.2130；T8 12.0486 vs 3.1315 ms/图），9 项 unresolved、Story DEFERRED、P4 running。`03/03b/04` v1 falsifier `rc=1`（`Acc.add` 张量 tol 未随 mask 切片，写报告前崩溃、**零数值产出**），原始日志 + SHA256SUMS + `FAILURE.json` 归档。按清单纪律不改 R6c 已绑定的 v1 字节（清单复核仍 PASS），新建 `e5_falsifier_v2.py` 最小修复 + 回归测试（复现崩溃、三种调用形状、参数逐项相同全 PASS）+ v2 `precheck` PASS；04:04:08Z 用无触发器计划任务 `GNN2SNN-E5-Falsifier-V2` 启动单一 GPU 作业（GPU 95 % / 4627 MiB，batch 1/40 已越过崩溃点）。R7b 12 产物清单 `create`+`verify` PASS | **E1 `NOT PASSED`（证据缺口闭合、门槛未降、未通过、未加 P4 pass）**；P4 `running`；S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中；无 Story、无新测试推理、无稿件、无引用核查、无独立评审。S5 成本与墙钟腿已有实测，机制/falsifier 腿待 v2 结果 | falsifier 崩溃属**实现缺陷**（precheck 只做哈希/结构门禁、不跑 GPU 前向），非 GPU 科学失败；失败证据保留，连败计数不累计；换道为「新建 v2 + 回归复现 + 同参数重跑」，不改写已发布清单绑定文件 | 监督 `GNN2SNN-E5-Falsifier-V2` 不重启；收集 gate-bound 报告 + `verify` 发布机制证据检查点；**同一次唤醒**启动 E2/E4/E6 首个 GPU 作业；之后 MNIST Gated / ModelNet40 → Story → 新测试 → 六节稿 → ≥36 引用 → 独立 CCFA≥7/10 |
| R7b-sync | 主会话补传（非科学进展） | 用户要求先把本地未推送成果传 GitHub；明确 stage 31 个文件（R7b2 权威检查点 + 12 产物、补充清单 12 产物、四活文件），两份清单 `verify` PASS，push 后 `git ls-remote` 核对远端 == 本地 `a85fb09`；原始 traceback、两个未发布草稿、大本地材料一律不传 | S9 推进（R7b-sync 已同步）；S2/S3/S5 状态栏同步为最新实测，但均仍 FAIL；E1 有证据 NOT PASSED、P4 running | 无失败 | 心跳下一轮以最新已推送检查点为准：先核实 v3 前置条件并启动唯一 v3 GPU 作业，不得重跑已发布的 latency/aggregate |

## 7 守护记录

| 项 | 内容 |
|----|------|
| 白名单校验 | 剔除远程 GPU/新 novix 作业（用户最新指令）；其余 A1–A9 实测可用 |
| 确认档位切换 | 由严格档切至授权档，依据用户 22:52 一次性授权「本机跑 + 增量传本仓库 + 完成协议工作」；清单外操作仍需逐项确认 |
| 敏感操作确认 | 仅 push 到本 agent 任务分支属已授权；不改 owner 的 TASK.md，不合并 main |
| 合规存疑处理 | 运行目录含 139 MiB 本地材料（含 40 MiB 证据包、calibration JSON），只上传明确列出的轻量文件，不整目录提交 |

## 8 停止门状态

| 门 | 状态 | 依据 |
|----|------|------|
| ①达成 | 未触发 | S2–S8 均未完成 |
| ②用户叫停 | 未触发 | 用户本轮是恢复指令 |
| ③穷尽 | 未触发 | 弹药库 A1–A9 可用，策略库未穷尽 |
| ④期限 | 未触发 | 用户未设期限；心跳每 15 分钟 ACTIVE，随时可叫停 |

## 9 迭代与复盘

| 项 | 内容 |
|----|------|
| 账本版本 | v1.0 |
| 目标迭代记录 | 无（成功标准锁定，未改） |
| 复盘沉淀 | 仓库的「暂停」是状态记录而非永久禁止；恢复须用户明确指令。既有 run root 与三个 agent 分支的交接可复用，但 luna-controls 对照实现实际为空，不能假定已存在。 |
| 归档状态 | 进行中 |
