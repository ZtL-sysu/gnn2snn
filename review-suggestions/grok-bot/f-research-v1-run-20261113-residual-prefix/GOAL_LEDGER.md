# 目标死磕状态账本

> 账本路径：本 run root 下 `GOAL_LEDGER.md`（唯一进度载体）。R1 建立于 2026-11-13T14:59:20Z。历史记录只追加不改写。

## 1 目标档案

| 项 | 内容 |
|----|------|
| 目标陈述 | 完成 `PROTOCOL-v3.md` + 分类头补充 + CURRENT_SCOPE 中 E0–E6 全部实验与公平对照、验证集上的书面 Story 判定、六节全新终稿、引用核查与当前稿独立 CCFA≥7/10；并同步到本仓库 agent 分支 |
| 背景与动机 | 用户 2026-11-13 22:52 +0800 指令：读取 GitHub gnn2snn 最新进展与协议，按科研流水线标准推进，代码本机 3060 跑，按 research-github-sync 传仓库，目标为完成协议中的所有工作 |
| 边界 | 不改研究方向、不缩数据子集、不伪造基线/效率、不远程 GPU、不擅自合并 main、不 force push、不发布 Release、不上传凭证/私人对话/大失败镜像；不修改旧证据哈希 |
| 时间尺度 | 心跳期（跨会话续跑；心跳配置需用户确认后创建，当前未创建） |
| 资源约束 | 本机 RTX 3060 12 GiB，Conda dyg-cold；contract v4 计算上限 E1≤80 / E2≤200 / E4≤80 / E3+E5≤40 / E6≤100 GPU 小时；单 GPU 单作业 |
| 判定者 | 系统指标（冻结协议的 .3pp + 完整成本门槛）、文档产物（六节新稿与审计）、独立评审（CCFA≥7/10） |
| 确认档位 | 授权档：用户已一次性授权「本机跑代码 + 增量 push 到 agent 分支 + 完成协议工作」。清单外操作（merge/force push/远程计算/发布）仍需逐项确认 |
| 运行模式 | 会话内死磕 + 建议心跳续跑（待用户确认） |

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
| A10 | 可组合编排 | 定时自动化心跳续跑 | 能力可用但**未启用**（需用户确认配置） | automation_update | 中 | 待确认 |
| — | 已剔除 | novix exp-4090 远程 GPU | 用户最新指令禁止 | — | — | 剔除：仅本机 3060 |

## 3 成功标准证据清单（锁定，来自冻结协议；不得修改）

| 编号 | 标准（可观测事实） | 判定者 | 反例排除 | 当前状态 |
|------|-------------------|--------|---------|---------|
| S1 | E0 冻结复现：seed41 全 5000 验证 ANN70.66/DC4 70.72/DC8 70.66，规范数组逐位一致，历史 T8 一处类别漂移如实披露 | 系统指标+报告 | 放宽容差、只报聚合、宣称逐样本历史复现 | PASS（v3 已验收，旧 v1/v2 失败保留） |
| S2 | E1 七行（+模拟头对照、共享 theta 附加对照）× T1/2/4/8 × seeds41/42/43 全部完成，逐样本证据与哈希齐全 | 文档产物 | 只跑三 affine 模式/单 seed 即宣称 E1 完成 | FAIL（R5：10 个行族 ×4T ×3seed 共 **120 份逐样本 NPZ 与哈希齐全并独立核验 PASS**——ANN 12、既有六行 72、新增残余三行 36；seed41 六站点 DC-adapt T4/T8 与 E0 canonical 逐位一致，shared-theta 非平凡别名。**仍不判 PASS**：协议 E1 每行还要求 per-site events、complete known cost plus unresolved items、ms/graph，完整操作成本与匹配墙钟仍缺；不得把精度/逐样本证据齐全等同于整行完成） |
| S3 | E1 门槛：T4 与 T8 三 seed 验证均值距最强转换对照 ≤0.3pp **且** 完整已知成本或匹配墙钟更低 | 系统指标 | 只用精度、省略成本项、用未实现零跳过当节能 | FAIL（R5 按补充协议「ALL conversion controls，含两个 DC readout 版本，排除 ANN 与候选」重算：T4 centered 0.705267 vs 最强 `centered_shared_theta` 0.705933，差距 **0.0667pp**；T8 centered 0.705667 vs 最强 `dc_adapt_six_site` 0.706133，差距 **0.0467pp**；均 ≤0.3pp，精度条件单独仍满足。**此数字取代 R4 的 0.0333pp**（那是缺行时的不完整对照集）。但完整已知成本与匹配墙钟仍缺，双门槛判为 NOT EVALUABLE / E1 NOT PASSED，不得用精度单独通过、不得用事件数冒充节能） |
| S4 | E2/E4/E6 忠实对照：NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN；CIFAR Gated 3 seeds、MNIST Gated、DGCNN ModelNet40 3 teachers | 文档产物 | 假定算子适配失败、跳过弱结果、用公开数字冒充实测 | FAIL |
| S5 | E3/E5 完整成本账（各操作项、AC/保守 MAC 双约定、3 次重复中位墙钟）、真实数据 gate-bound 与误差、falsifier | 文档产物 | 只报事件数、只报宽松界、芯片能耗声明 | FAIL |
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
| 真实作业句柄 | R4：唤醒核实无存活作业（GPU 0%/955MiB、无 python 训练进程）→ 顺序启动 4 个作业，全部 exit 0：seed42/43 校准（89.49/89.04 s）、E1 seed42（697.4 s）、E1 seed43（11m49s）；18:04Z 复核无存活作业。R4b（18:20Z）：`git fetch` 后 GPU 0%/955MiB、PowerShell CIM 查询 0 个 python 进程 → **无存活作业，本轮未启动任何 GPU 作业**。**R5（20:11Z）：`git fetch` 后 GPU 0%/955MiB、CIM 精确过滤 0 个 python/conda 进程 → 启动单一后台驱动 `LYwybZ`，seed41/42/43 顺序全 5000 验证分别 251.8/252.3/252.8 s，全部 `rc=0`，20:24:14Z 结束；20:24Z 复核 GPU 0%/955MiB、无存活 Python 进程。** 恢复/复现命令见 HANDOFF R5 段与 `20261113T202500Z-e1-fullrow-residual/CHECKPOINT.md` 第 9 节。 |

## 6 执行轮与换道日志

| 轮次 | 策略 | 动作与结果 | 证据核验 | 败因归类 | 换道决策 |
|------|------|-----------|---------|---------|---------|
| R1 | 恢复优先 | fetch 全部远端，定位协议与交接；建独立 worktree/分支；协议/数据/源码哈希与 GPU/环境实测；合成检查 PASS、单测 2 passed；P4 前置校验 PASS | S1 PASS（继承），S9 进行中，S2–S8 FAIL/待验证 | 无失败；S2 缺口源于此前暂停中断 | 继续 M2（实现控制评估器） |
| R2 | 补齐被中断的 M2 | 实现双校准绑定的对照评估器 + 4 项合成测试通过；预检 18 个绑定输入 OK；启动首个官方全 5000 图控制网格（3 行×4 T，seed41，scale1） | S9 首个检查点远端确认；S2 仍 FAIL（作业未完成）；S3–S8 未动 | 无失败；CLI 导入顺序问题已修复并重测 | 等作业完成后发布结果检查点，继续 M3 |
| R3 | 固定网格实跑 + 证据核验 | 作业 exit 0，5000 图 12 行逐样本证据；教师 0.7066 与前后哈希一致；与既有独立 generic 运行逐位一致（4/4 T）；生成 36 行固定列指标 | S2 部分推进（seed41 新 3 行实测）但仍 FAIL；S3 仍 FAIL（成本未测）；S9 三个检查点远端确认 | 无失败 | 继续 M3：先做 42/43 冻结校准，再不变重跑，并补共享 theta 与完整成本 |
| R4 | 不变流程多 seed 补网格 | 冻结输入门禁全 PASS；进程/GPU 判无存活作业后顺序跑 4 个本机 GPU 作业全部 exit 0：42/43 冻结全训练无标签校准（45000×2 遍，复用 frozen `calibrate_loader`）、42/43 E1 六行 ×4T 全 5000 验证（各 24 逐样本 NPZ）；流程等价性交叉验证 PASS、逐样本核验 42/43 均 PASS；跨 seed 汇总 336 行固定列指标 | S2 大幅推进（72 组逐样本证据齐全）但仍 FAIL（缺 ANN/历史 DC-adapt 汇总/共享 theta）；S3 精度条件满足但双门槛 NOT EVALUABLE（成本/墙钟缺）故仍 FAIL；S4–S8 未动；S9 待本轮 push 远端确认 | 无失败（连败 0） | 继续 M3 收尾：共享 theta + 历史六站点 DC-adapt + ANN 行，再 E3/E5 完整成本与 falsifier（S3 前置），之后 E2/E4/E6 |
| R4b | 证据完整性修复 + 清单发布（非科学推进） | 唤醒门禁全 PASS（12 项冻结输入 / contract v4 / P4 running / 0 个 python 进程 / 0 GPU 作业）；发现未发布的 R4 23 产物草稿清单因 4 个新增脚本去标识而失效，按不可变规则不覆盖，改新建 74 产物权威清单（含全部 48 个逐样本 NPZ）并 `verify` PASS；新增冻结门禁与上传内容检查两个只读工具；重验流程等价性、E1 预检、10/10 公开副本均 PASS | S2/S3 状态不变（仍 FAIL），S4–S8 未动；S9 本地清单 verify PASS，仍待远端 SHA 核对；**E1 判据 NOT EVALUABLE 不变，P4 仍 running** | 无失败（连败 0）；工具层 `tasklist /FI` 在 Git Bash 被路径改写，已用 PowerShell CIM 替代；`conda run -c` 多行脚本报 NotImplementedError，已改单行 | 不把打包当科学进展；下一轮直接进 M3 收尾（共享 theta / 历史六站点 / ANN 行 / 完整成本与墙钟） |
| R5 | 补齐 E1 残余行 + 全行判据重算 | 唤醒门禁 PASS（12 项冻结输入 / 5/5 官方 processed 哈希 / contract v4 / P4 running / 0 个 python-conda 进程 / GPU 0%955MiB）；保留上一轮三次 CLI 失败证据并去标识公开，修复默认参数解析与 `rel()` 外部 DCGS 路径两处缺陷；单一后台作业 `LYwybZ` 顺序跑完 seed41/42/43，各 5000 图、3 行 ×4T，全部 `rc=0`（251.8/252.3/252.8 s）；新增 36 份逐样本 NPZ，`verify` PASS；seed41 六站点 DC-adapt T4/T8 与 E0 canonical 逐位一致；ANN 上界行 12 NPZ 独立复验；`aggregate_e1_fullrows_v1.py` 汇总 10 行族 ×4T ×3seed = 120 份逐样本证据、40 描述行、560 固定列指标；新建 72 产物权威清单 `create`+`verify` PASS | **S2 大幅推进但仍 FAIL**（120 份逐样本证据与哈希齐全并核验 PASS，但每行完整已知成本/未决项/ms/graph 仍缺）；**S3 仍 FAIL 且数字被诚实重算**：按 ALL 8 转换对照（含两个 DC readout 版本）取最强，T4 差距 0.0667pp、T8 0.0467pp，取代 R4 的 0.0333pp，精度条件仍单独满足但双门槛 NOT EVALUABLE；S4–S8 未动；P4 仍 running；S9 本地清单 PASS、远端 SHA 待写回 | 无 GPU 失败（连败 0）；三次 CLI 失败为旧策略且已修复，证据保留；重启前静态复核拦下第二处必现缺陷，未形成 GPU 失败 | 不因精度条件满足而给 E1 加 pass；下一轮最高优先 E3/E5 完整操作成本 + 3 次重复中位墙钟 + gate-bound/误差 falsifier，再 E2/E4/E6，再 Story/稿件/引用/独立评审 |

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
| ④期限 | 未触发 | 用户未设期限；心跳配置待确认 |

## 9 迭代与复盘

| 项 | 内容 |
|----|------|
| 账本版本 | v1.0 |
| 目标迭代记录 | 无（成功标准锁定，未改） |
| 复盘沉淀 | 仓库的「暂停」是状态记录而非永久禁止；恢复须用户明确指令。既有 run root 与三个 agent 分支的交接可复用，但 luna-controls 对照实现实际为空，不能假定已存在。 |
| 归档状态 | 进行中 |
