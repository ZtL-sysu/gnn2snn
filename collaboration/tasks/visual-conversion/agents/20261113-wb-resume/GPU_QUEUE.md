# GPU 作业队列 — 20261113-wb-resume

维护规则：以 GitHub 已推送的最新检查点为唯一进度依据；每次心跳唤醒先查 `nvidia-smi` 与精确 Python 进程，GPU 空闲就消费队首并启动真实作业，**不得只做文档写回**。作业在跑则监督/收集结果并预排后续，不重复启动。所有作业只用本机 RTX 3060 + Conda `dyg-cold`。

> **⚠ 2026-11-14T00:05Z 并发事故与唯一在跑作业（最高优先，先读这一段）**
> 23:37Z 的 `ThE3rS`(PID 27348) 与 23:51Z 的 `jvgzz3`(PID 15052) 曾**同时**占用同一块 3060，违反单 GPU 单作业并污染匹配墙钟。两个作业链已全部终止，污染日志与增量记录原样归档于 `logs/e3e5-latency-concurrency-incident-v1/`，**其所有计时数值一律作废**。
> 现在**唯一**在跑的作业：后台句柄 `TsI47h`，命令 `python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume`，日志 `logs/e5_latency_v1.clean.driver.log`，增量证据 `results/e3-e5-v1/e5_latency_v1.partial.jsonl`（每完成一个配置写一行，可断点续跑）。
> 心跳唤醒时：**先用 `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'e3_e5_v1\\.py' }` 做宽匹配查活**（窄的进程名过滤曾漏报存活作业），只要该作业仍在跑就**不得重启、不得再起第二个 GPU 作业**；若进程确已消失且 `e5_latency_v1.json` 不存在，才用**完全相同**的命令（含 `--resume`）续跑。禁止缩短批次、降低 repeats 或改用旧脚本。

| # | 作业 | 命令 / 产物 | 预计时长 | 依赖 | 状态 |
|---|------|-------------|---------|------|------|
| 1 | E5 匹配墙钟（3 次重复中位数） | `python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume` → `results/e3-e5-v1/e5_latency_v1.json` + `latency_METRICS.csv`；日志 `logs/e5_latency_v1.clean.driver.log` | 约 60–90 分钟（37 配置 × 40 批 × 4 遍） | precheck PASS、E3 cost 已完成 | **运行中（干净重跑）**：后台句柄 `TsI47h`，2026-11-14T00:05Z 启动；此前两个并发作业的计时全部作废，见顶部事故说明 |
| 2 | E1 双门槛重算 | `... e3_e5_v1.py aggregate` → `results/e3-e5-v1/e3_e5_criterion_v1.json`；读 `e1-fullrow-summary-v1` 的精度差 | <1 分钟（CPU） | #1 的 `e5_latency_v1.json` | 待 #1；注意 unresolved items 仍会使 `Cost with unresolved omitted work cannot pass` 生效 |
| 3 | R6 结果检查点上传 | CHECKPOINT + METRICS + 清单 + HANDOFF/LOG/账本写回，push 后 `git ls-remote` 核对 | 数分钟 | #2 | 待 #2 |
| 4 | E2/E4/E6 忠实对照准备 | 按 R5 检查点：NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN；先做可复现入口与预检，再排训练 | 长（训练为主） | #3 完成、冻结输入门禁 PASS | 排队中 |
| 5 | MNIST Gated / DGCNN ModelNet40 跨任务 | 冻结 split 与 own-trajectory dynamic kNN，弱结果照实跑 | 长 | #4 | 排队中 |
| 6 | E5 机制与 falsifier 补强 | gate-bound vs message/layer/logit 误差、raw-vs-decoded、fresh-vs-stale gates，全 T 与 degree/sharpness 子集 | 中 | #1/#2 的实测数据 | 排队中 |

## 恢复与失败处理

- **会话边界风险（重要）**：#1 的干净重跑带 `--resume`，每完成一个配置即写一行 `e5_latency_v1.partial.jsonl`。若主会话结束时该进程被环境终止，下一次心跳（≤15 分钟）先做宽匹配查活，确认无 `e3_e5_v1.py` 进程且 `e5_latency_v1.json` 不存在后，**用完全相同的 `--batches 40 --warmup 1 --repeats 3 --resume` 续跑**（已完成的增量行会被跳过，不重测）；把中断日志改名保留为 `logs/e5_latency_v1.interrupted-*.log`，并在检查点如实记录「环境终止导致续跑，非科学失败」。不得缩短批次或降重复次数。
- #1 若因其它原因中断：先查 `nvidia-smi`/精确进程确认无存活作业；`e5_latency_v1.json` 不存在时按同一命令重跑（`require(not lp.exists())` 只拒绝覆盖已发布报告）。保留中断前的日志为证据，不要改脚本或阈值。
- #1 完成后立刻跑 #2 `aggregate`（CPU，<1 分钟）并发布 #3；**#3 上传后必须在同一次唤醒里启动 #4 的第一个 GPU 作业**，不允许「结果整理完就空卡等下一次」。
- 严禁：缩短 `--batches` 来赶进度、降 `repeats`、挑 T、改 pJ 常数、用事件数冒充能耗/加速、把成本账的 unresolved items 静默删掉。
- 心跳现在 `FREQ=HOURLY;INTERVAL=1;BYMINUTE=0,15,30,45`（每 15 分钟）。每次唤醒必须报告 GPU 占用事实与队列消费情况；若队列某项预计 >15 分钟，用单一后台驱动器串行跑完，不要拆成多次唤醒。

## 实测节奏（用于估算）

- R5 单 seed 全 5000 图 3 行 ×4 T 网格：251.8 / 252.3 / 252.8 s ≈ 4.2 分钟。
- R6 #1：config 1（ANN）3.6 s；config 9（first_analog T8）累计 1140.2 s，约 127 s/配置 → 37 配置约 78 分钟。
- 因此 60 分钟心跳下作业结束后平均空转约 30 分钟；15 分钟间隔把空转压到 ≤7.5 分钟。平台拒绝 `BYHOUR` 列表，故用 `HOURLY` + `BYMINUTE=0,15,30,45` 实现。
