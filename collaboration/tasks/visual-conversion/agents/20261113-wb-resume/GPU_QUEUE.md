# GPU 作业队列 — 20261113-wb-resume

维护规则：以 GitHub 已推送的最新检查点为唯一进度依据；每次心跳唤醒先查 `nvidia-smi` 与精确 Python 进程，GPU 空闲就消费队首并启动真实作业，**不得只做文档写回**。作业在跑则监督/收集结果并预排后续，不重复启动。所有作业只用本机 RTX 3060 + Conda `dyg-cold`。

> **⚠ 2026-11-14T01:40Z（R6d）当前唯一在跑作业链（最高优先，先读这一段）**
> 上一轮干净作业 `TsI47h`（00:04:49Z 启动）在 **00:14Z 左右被环境/会话边界终止**：驱动日志只有 START 一行、无 traceback、无完成行，增量只写到 7/37 配置，`e5_latency_v1.json` 不存在。唤醒时（01:36Z）`nvidia-smi` 为 `0 % / 955 MiB`，`Get-CimInstance Win32_Process` 宽匹配返回 0 个计算进程 → 判定为**环境中断，非科学失败**。中断快照（驱动日志 + 7 行增量 + SHA256SUMS + `INTERRUPTED.json`）归档于 `logs/e5-latency-interruption-v1/`。
> **这 7 条增量数值有效**：它们来自并发事故处置后启动的单一干净作业、复核后的 `e3_e5_v1.py`、全 5000 图固定顺序、1 warmup / 3 repeats，**不是** `logs/e3e5-latency-concurrency-incident-v1/` 中作废的污染计时。已作为 `--resume` 起点，续跑会逐行校验 `graphs_per_repeat` 与 `repeats` 后跳过。
> **现在唯一在跑的作业链**：后台句柄 `v10FQp`，单一后台驱动 `research/visual-graphs/continuation_20261113/code/gpu_queue_driver_v1.py`（driver PID 5776，latency 子进程 PID 11560，01:40:39Z 启动），串行执行下表 #1→#2→#3→#4。总日志 `logs/gpu_queue_driver_v1.driver.log`，逐步日志 `logs/gpu_queue_driver_v1/<step>.log`，逐步 rc/耗时 `logs/gpu_queue_driver_v1.status.jsonl`。启动后实测 **GPU 90 % / 5130 MiB / 112.24 W**；宽匹配确认只有**一个** `e3_e5_v1.py latency` 进程。
> 心跳唤醒时：**先用 `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'e3_e5_v1\.py|e5_falsifier_v1\.py|gpu_queue_driver_v1\.py' }` 做宽匹配查活**（窄进程名过滤曾误报 0，导致两作业并发）。只要驱动或任一子作业仍在跑就**不得重启、不得再起第二个 GPU 作业**；改为监督、收集已完成产物并预排后续。若整条驱动链确已消失，按下方恢复规则用**完全相同参数**续跑。

| # | 作业 | 命令 / 产物 | 预计时长 | 依赖 | 状态 |
|---|------|-------------|---------|------|------|
| 1 | E5 匹配墙钟（3 次重复中位数）续跑 | `e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume` → `results/e3-e5-v1/e5_latency_v1.json` + `latency_METRICS.csv`；增量 `e5_latency_v1.partial.jsonl`；逐步日志 `logs/gpu_queue_driver_v1/01_e5_latency_resume.log` | 剩余 30 配置，约 40–65 分钟 | precheck PASS、E3 cost 已完成、7 条有效增量 | **运行中（驱动 `v10FQp` step 01）**：`resumed_completed_configs: 7 / configs_in_this_run: 37`，参数一字未改 |
| 2 | E1 双门槛重算 | `e3_e5_v1.py aggregate` → `results/e3-e5-v1/e3_e5_criterion_v1.json`；读 `e1-fullrow-summary-v1` 的精度差 | <1 分钟（CPU） | #1 的 `e5_latency_v1.json` | **已排入同一驱动 step 02**；注意 9 项 unresolved 仍会使 `Cost with unresolved omitted work cannot pass` 生效，`verdict` 预期仍 `E1 NOT PASSED` |
| 3 | E5 机制证据（GPU 单作业） | `e5_falsifier_v1.py run --device cuda --batches 40` → `results/e5-falsifier-v1/e5_gate_bound_v1.json` + `falsifier_METRICS.csv`；紧接 `e5_falsifier_v1.py verify` | 中（40 批 ×4 T × 多层反事实） | #1/#2 完成（同一驱动内串行，不与 latency 并发） | **已排入同一驱动 step 03/04**；`precheck` 已 PASS，运行时仍须通过「开关全关与冻结前向一致 ≤1e-4」保真校验 |
| 4 | R6d 结果检查点上传 | CHECKPOINT + METRICS + SHA-256 清单 + HANDOFF/LOG/账本写回，push 后 `git ls-remote` 核对 | 数分钟 | #1–#3 的真实产物 | 待驱动跑完；本轮先发布「恢复 + 单一驱动启动」检查点 |
| 5 | E2/E4/E6 忠实对照准备 | 按 R5 检查点：NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN；先做可复现入口与预检，再排训练 | 长（训练为主） | #4 完成、冻结输入门禁 PASS | 排队中；**必须在 #4 上传的同一次唤醒里启动第一个 GPU 作业**，不许空卡等下一次 |
| 6 | MNIST Gated / DGCNN ModelNet40 跨任务 | 冻结 split 与 own-trajectory dynamic kNN，弱结果照实跑 | 长 | #5 | 排队中 |

## 恢复与失败处理

- **单一驱动链（R6d 新增）**：#1→#2→#3→#4 由 `gpu_queue_driver_v1.py` 串行执行，每步独立日志与 rc，写 `logs/gpu_queue_driver_v1.status.jsonl`。任一步失败**不整轮停摆**：驱动保留原始失败日志并继续下一步；若 #1 未产出报告则自动插入一次**完全相同**的 `--resume` 重试（并补跑 `aggregate`），若 #3 未产出报告则插入一次相同 `run` 重试（并补跑 `verify`）。重试仍使用 `--batches 40 --warmup 1 --repeats 3 --resume`，不缩短批次、不降 repeats。
- **会话边界风险**：若整条驱动链被环境终止，下一次心跳（≤15 分钟）先宽匹配查活 → 确认无任何 `e3_e5_v1.py` / `e5_falsifier_v1.py` / `gpu_queue_driver_v1.py` 进程且 `e5_latency_v1.json` 不存在 → 把当前驱动日志与增量快照复制归档为 `logs/e5_latency_v1.interrupted-<UTC>.*` 并留 SHA-256 → 用完全相同命令重启驱动（增量行自动跳过）。在检查点如实记录「环境终止导致续跑，非科学失败」。
- **并发纪律**：同一张 3060 任何时刻只允许一个 `e3_e5_v1.py latency`。启动前必须按**完整 CommandLine**宽匹配（含 `conda-script.py run -n dyg-cold` wrapper 及子进程）；发现任何同脚本进程绝不启动第二个。若确需替换，先显式终止旧链、隔离受污染中间结果并留 SHA256，再用不变参数重启单一干净作业。被并发污染的墙钟数字一律不得作为 E3/E5 证据。
- #1 完成后驱动立即跑 #2（CPU），#2 后跑 #3/#4（GPU+CPU）；**#4 上传后必须在同一次唤醒里启动 #5 的第一个 GPU 作业**，不允许「结果整理完就空卡等下一次」。
- 严禁：缩短 `--batches` 来赶进度、降 `repeats`、挑 T、改 pJ 常数、用事件数冒充能耗/加速、把成本账的 unresolved items 静默删掉、把计划当结果。
- 心跳 `FREQ=HOURLY;INTERVAL=1;BYMINUTE=0,15,30,45`（每 15 分钟）。每次唤醒必须报告 GPU 占用事实与队列消费情况；预计 >15 分钟的作业用单一后台驱动器串行跑完，不拆成多次唤醒。

## 实测节奏（用于估算）

- R5 单 seed 全 5000 图 3 行 ×4 T 网格：251.8 / 252.3 / 252.8 s ≈ 4.2 分钟。
- R6 #1：config 1（ANN）3.6 s；config 9（first_analog T8）累计 1140.2 s，约 127 s/配置 → 37 配置约 78 分钟。
- R6c `TsI47h`：00:04:49Z→约 00:14Z 完成 7 配置（约 85 s/配置），随后环境中断、GPU 空转至 01:40Z。R6d 续跑剩余 30 配置，预计 40–65 分钟；驱动随后继续 falsifier，避免作业结束后空卡。
