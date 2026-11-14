# GPU 作业队列 — 20261113-wb-resume

维护规则：以 GitHub 已推送的最新检查点为唯一进度依据；每次心跳唤醒先查 `nvidia-smi` 与精确 Python 进程，GPU 空闲就消费队首并启动真实作业，**不得只做文档写回**。作业在跑则监督/收集结果并预排后续，不重复启动。所有作业只用本机 RTX 3060 + Conda `dyg-cold`。

> **⚠ 2026-11-14T06:23Z（R8）当前唯一在跑作业链（最高优先，先读这一段）**
> 无触发器计划任务 `GNN2SNN-GPU-Queue-V3` 已从 `Disabled` / `LastResult=267014`（0x41306 被终止，第 4 次环境中断）恢复为 `Running`：`launcher 22092（父 1820 = svchost.exe -k netsvcs -p -s Schedule）→ driver 33172 → 单个子作业`。驱动每步后重读 `logs/gpu_queue_v3.json`，可在作业运行中追加下一步，保证 3060 连续占用；宽匹配确认任何时刻只有一条计算链。
> **本轮已完成（真实 GPU，全部 `rc=0`）**：`01` v3 falsifier run **536.3 s** + `02` verify → `e5_gate_bound_v1.json` + 120 行 `falsifier_METRICS.csv`；`06` seed43 全训练校准 25.8 s + `07` verify（3/3 stage-1 报告）；`08` stage-2 precheck；`09/10/11` NaiveIF 三 seed × 全 5000 验证图 × T{1,2,4,8} 评估各 28.3 s；`12` verify（12 NPZ、canonical 索引/标签逐位一致）。实测 **95 % / 1823 MiB / 126.67 W** 与 **91 % / 1823 MiB / 130.36 W**。权威检查点 `20261114T062300Z-r8-e5-mechanism-e2-naiveif/`（33 产物清单 create+verify PASS）。
> **队首（正在跑）**：`13_e5_falsifier_repro_run` — `e5_falsifier_repro_v1.py run --device cuda --batches 40`（约 540 s，**独立确定性复现检查，不是新科学结论**；只读 wrapper 导入被清单绑定的 v3 脚本、仅重定向 `OUT`，canonical 报告永不覆盖）→ `14 verify` → `15 compare`（除时间戳外数值报告相同且固定列 CSV 逐字节相同；不一致如实记 FAIL，绝不隐藏）。
> **查活**：`Get-ScheduledTask -TaskName 'GNN2SNN-GPU-Queue-V3'` + `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gpu_queue_driver_v3|e5_falsifier|e2_naiveif|e3_e5_v1' }`。在跑绝不重启、绝不起第二个 GPU 作业；整条链消失才 `Start-ScheduledTask -TaskName 'GNN2SNN-GPU-Queue-V3'` 幂等续跑。
> **队列消费完必须同一次唤醒追加真实 GPU 作业，禁止空卡等下一次心跳**：E2/E4/E6 其余忠实对照（官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN，各 3 seeds × 全 5000 官方验证图，弱结果照实跑照实报），再 MNIST Gated / DGCNN ModelNet40；之后才做验证集-only Story 判定、冻结后一次新测试确认、六节新稿、≥36 引用核查、去重审计、独立 CCFA≥7/10。
> 下方 R7b / R7 段与表格均为**历史记录**；状态以本段与 `logs/gpu_queue_v3.json`（15 步）为准。

> **历史段（R7b，2026-11-14T04:04Z）：E5 37/37 墙钟完成与 v2 falsifier 作业（均已结束，状态以 R8 顶部段为准）**
> R7 的会话外托管驱动 `GNN2SNN-E5-Driver-R7` 已**完整跑完**（03:08:04Z→03:56:42Z，48.6 分钟未中断，根因修复实证有效）：`01 latency rc=0`（**37/37 配置**，`results/e3-e5-v1/e5_latency_v1.json` + `latency_METRICS.csv`）、`02 aggregate rc=0`（`e3_e5_criterion_v1.json`）。**E1 由 `NOT EVALUABLE` 变为有证据支撑的 `E1 NOT PASSED`**（精度腿 0.0667/0.0467 pp 满足；成本腿与墙钟腿均不更低；9 项 unresolved 保留；Story DEFERRED；P4 仍 running）——证据缺口闭合但**门槛未降、未通过**。
> `03/03b e5_falsifier_v1.py run` 与 `04 verify` 均 `rc=1`：`Acc.add()` 中张量型 `tol` 未随 `mask` 切片导致广播失败；崩溃发生在写报告前，**无任何 falsifier 数值**。原始日志 + SHA256SUMS + `FAILURE.json` 归档于 `logs/e5-falsifier-failure-v1/`。`e5_falsifier_v1.py` 被 R6c 清单绑定、字节未改；新建 `e5_falsifier_v2.py` 做一处最小修复，`test_e5_falsifier_v2.py` 复现缺陷并验证参数逐项不变，CPU 回归 + `precheck` 全 PASS。
> **现在唯一在跑的作业链**：无触发器计划任务 `GNN2SNN-E5-Falsifier-V2`（`Triggers=0`、`IgnoreNew`、12h），`launcher 25360 (父 1820 = svchost Schedule) → driver 26076 → 唯一 falsifier 9424`，04:04:08Z 启动，串行 `e5_falsifier_v2.py run --device cuda --batches 40` → `verify`。实测 **GPU 95 % / 4627 MiB / 118 W**，`{"batch": 1, "of": 40}` 已越过 v1 崩溃点。日志 `logs/gpu_queue_driver_v2.driver.log`、`logs/gpu_queue_driver_v2/`、`gpu_queue_driver_v2.status.jsonl`。**在跑绝不重启、绝不起第二个 GPU 作业。** 权威结果检查点 `20261114T041000Z-r7b2-e5-latency-criterion-public/`（12 产物清单 create+verify PASS、上传预检 PASS）；本地草稿 `20261114T040000Z-r7b-e5-latency-criterion/` 因清单绑定含身份路径的原始日志被预检拒绝，从未 push、清单作废、事实已并入权威检查点。
> 查活：`Get-ScheduledTask -TaskName 'GNN2SNN-E5-Falsifier-V2'` + 宽匹配 `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'e5_falsifier_v2\.py|gpu_queue_driver_v2\.py|gpu_queue_launcher_v2\.py' }`。链消失才归档后 `Start-ScheduledTask -TaskName 'GNN2SNN-E5-Falsifier-V2'` 幂等续跑。

> **历史段（R7，2026-11-14T03:08Z）：会话外托管根因修复**
> **根因已修复**：R6c `TsI47h` 与 R6d `v10FQp` 都是自动化会话的后台进程，每次在会话边界被终止，分别造成约 82 / 75 分钟 GPU 空转（连续三次同因中断）。R7 改用**无触发器的 Windows 计划任务** `GNN2SNN-E5-Driver-R7` 托管同一个 `gpu_queue_driver_v1.py`，其进程树属于 Task Scheduler 服务，**不随会话结束而终止**。
> R6d 链 `v10FQp`（01:40:39Z 启动）在约 **01:46Z 被环境终止**：驱动日志含 `DRIVER_START`/`STEP_START` 但无 `STEP_END`、无 traceback、`status.jsonl` 不存在，增量恰为 9/37 且最后写入 01:46Z。唤醒时（03:00:55Z）`nvidia-smi` `0 % / 955 MiB / 12 W`、宽匹配 0 个计算进程 → 判定为**环境中断，非科学失败**。中断快照（driver + step01 + 9 行增量 + SHA256SUMS + `INTERRUPTED.json`）归档于 `logs/e5-latency-interruption-v2/`。**9 条增量数值有效**（1–7 来自并发事故处置后的干净作业 `TsI47h`，8–9 来自 R6d 单一驱动；均全 5000 图 / 1 warmup / 3 repeats），与 `logs/e3e5-latency-concurrency-incident-v1/` 中**作废**的污染计时严格区分。
> **现在唯一在跑的作业链**：计划任务 `GNN2SNN-E5-Driver-R7`（`Triggers: 0`、`MultipleInstances: IgnoreNew`、State `Running`），`launcher PID 32844 → driver PID 23620 → latency PID 29800`，launcher 父进程 **PID 1820 = `svchost.exe -k netsvcs -p -s Schedule`**。03:08:04Z 启动，`resumed_completed_configs: 9 / 37`。串行执行下表 #1→#2→#3→#4。总日志 `logs/gpu_queue_driver_v1.driver.log`，launcher 日志 `logs/gpu_queue_driver_v1.launcher.log`，逐步日志 `logs/gpu_queue_driver_v1/<step>.log`，rc `logs/gpu_queue_driver_v1.status.jsonl`。实测 **GPU 88–92 % / 5155–5159 MiB / 110–118 W**；宽匹配确认只有**一个** `e3_e5_v1.py latency`。本轮增量已推进到 **12/37**（新测 `generic T1` 1.9336、`T2` 3.7015、`T4` 7.1592 ms/graph）。
> 心跳唤醒时：**先 `Get-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7'` + `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'e3_e5_v1\.py|e5_falsifier_v1\.py|gpu_queue_driver_v1\.py|gpu_queue_launcher_v1\.py' }` 宽匹配查活**（窄进程名过滤曾误报 0，导致两作业并发）。只要驱动或任一子作业仍在跑就**不得重启、不得再起第二个 GPU 作业**；改为监督、收集已完成产物并预排后续。仅当整条链确已消失，才用 `Start-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7'` 幂等续跑（参数完全不变，`IgnoreNew` 会忽略已有实例上的重复请求）。

| # | 作业 | 命令 / 产物 | 预计时长 | 依赖 | 状态 |
|---|------|-------------|---------|------|------|
| 1 | E5 匹配墙钟（3 次重复中位数）续跑 | `e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume` → `results/e3-e5-v1/e5_latency_v1.json` + `latency_METRICS.csv`；增量 `e5_latency_v1.partial.jsonl` | 已完成 2898.2 s | precheck PASS、E3 cost 已完成、9 条有效增量 | **DONE rc=0（03:56:22Z）**：**37/37 配置**、`resumed_configs: 9`、全 5000 图 ×3 重复中位、1 warmup；报告已发布并进 R7b 清单 |
| 2 | E1 双门槛重算 | `e3_e5_v1.py aggregate` → `results/e3-e5-v1/e3_e5_criterion_v1.json` | <1 分钟（CPU） | #1 | **DONE rc=0（03:56:22Z）**：`verdict E1 NOT PASSED (method advantage not established)`、`evidence_backed_criterion_evaluation_no_method_advantage_claim`；精度腿 true（T4 0.0667 / T8 0.0467 pp），成本腿 false、墙钟腿 false、9 项 unresolved、Story DEFERRED、P4 running。证据缺口闭合，门槛未降、未通过 |
| 3 | E5 机制证据（GPU 单作业） | `e5_falsifier_v2.py run --device cuda --batches 40` → `results/e5-falsifier-v1/e5_gate_bound_v1.json` + `falsifier_METRICS.csv`；紧接 `verify` | 40 批 ×4 T × 多层反事实（v2 batch1 13.7 s） | v1 失败已归档并修复；#1/#2 已完成 | **运行中（R7b 计划任务 `GNN2SNN-E5-Falsifier-V2`，04:04:08Z 启动）**：GPU 95 % / 4627 MiB；v1 两次 run + verify `rc=1` 为已保留的实现缺陷；参数未变，运行时仍须通过「开关全关与冻结前向一致 ≤1e-4」保真校验 |
| 4 | R7b 结果检查点上传 | CHECKPOINT + 12 产物 SHA-256 清单 + HANDOFF/LOG/账本写回，push 后 `git ls-remote` 核对 | 数分钟 | #1–#2 真实产物 + #3 失败证据与 v2 修复 | **本轮已发布权威版** `20261114T041000Z-r7b2-e5-latency-criterion-public/`（清单 create+verify PASS、预检 PASS）；草稿 `...040000Z...` 未 push 已声明作废；#3 结果出来后再发机制证据检查点 |
| 5 | E2/E4/E6 忠实对照准备 | 按 R5 检查点：NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN；先做可复现入口与预检，再排训练 | 长（训练为主） | #4 完成、冻结输入门禁 PASS | 排队中；**必须在 #4 上传的同一次唤醒里启动第一个 GPU 作业**，不许空卡等下一次 |
| 6 | MNIST Gated / DGCNN ModelNet40 跨任务 | 冻结 split 与 own-trajectory dynamic kNN，弱结果照实跑 | 长 | #5 | 排队中 |

## 恢复与失败处理

- **单一驱动链（R6d 新增）**：#1→#2→#3→#4 由 `gpu_queue_driver_v1.py` 串行执行，每步独立日志与 rc，写 `logs/gpu_queue_driver_v1.status.jsonl`。任一步失败**不整轮停摆**：驱动保留原始失败日志并继续下一步；若 #1 未产出报告则自动插入一次**完全相同**的 `--resume` 重试（并补跑 `aggregate`），若 #3 未产出报告则插入一次相同 `run` 重试（并补跑 `verify`）。重试仍使用 `--batches 40 --warmup 1 --repeats 3 --resume`，不缩短批次、不降 repeats。
- **会话边界风险（R7 已换道）**：R6c/R6d 两次证明「会话内后台进程」会被会话边界杀死。R7 起改用无触发器计划任务 `GNN2SNN-E5-Driver-R7`（`launcher_v1.py` 做日志重定向），进程父为 `svchost.exe -k netsvcs -p -s Schedule`，跨会话存活。若整条链仍消失：先宽匹配查活 → 确认无任何 `e3_e5_v1.py` / `e5_falsifier_v1.py` / `gpu_queue_driver_v1.py` / `gpu_queue_launcher_v1.py` 进程且 `e5_latency_v1.json` 不存在 → 把当前驱动日志与增量快照复制归档（`<UTC>` 命名 + SHA-256 + `INTERRUPTED.json`）→ `Start-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7'` 幂等续跑（增量行自动跳过，参数完全不变）。在检查点如实记录「环境终止导致续跑，非科学失败」；**不得另起第二个后台驱动**。
- **R7 计划任务查活命令**：`Get-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7' | Select TaskName,State` 与 `Get-ScheduledTaskInfo -TaskName 'GNN2SNN-E5-Driver-R7' | Select LastRunTime,LastTaskResult`。任务 `Triggers=0`、`MultipleInstances=IgnoreNew`、`ExecutionTimeLimit=24h`；重复 `Start-ScheduledTask` 在已有实例运行时被忽略，不会造成并发。
- **并发纪律**：同一张 3060 任何时刻只允许一个 `e3_e5_v1.py latency`。启动前必须按**完整 CommandLine**宽匹配（含 `conda-script.py run -n dyg-cold` wrapper 及子进程）；发现任何同脚本进程绝不启动第二个。若确需替换，先显式终止旧链、隔离受污染中间结果并留 SHA256，再用不变参数重启单一干净作业。被并发污染的墙钟数字一律不得作为 E3/E5 证据。
- #1 完成后驱动立即跑 #2（CPU），#2 后跑 #3/#4（GPU+CPU）；**#4 上传后必须在同一次唤醒里启动 #5 的第一个 GPU 作业**，不允许「结果整理完就空卡等下一次」。
- 严禁：缩短 `--batches` 来赶进度、降 `repeats`、挑 T、改 pJ 常数、用事件数冒充能耗/加速、把成本账的 unresolved items 静默删掉、把计划当结果。
- 心跳 `FREQ=HOURLY;INTERVAL=1;BYMINUTE=0,15,30,45`（每 15 分钟）。每次唤醒必须报告 GPU 占用事实与队列消费情况；预计 >15 分钟的作业用单一后台驱动器串行跑完，不拆成多次唤醒。

## 实测节奏（用于估算）

- R5 单 seed 全 5000 图 3 行 ×4 T 网格：251.8 / 252.3 / 252.8 s ≈ 4.2 分钟。
- R6 #1：config 1（ANN）3.6 s；config 9（first_analog T8）累计 1140.2 s，约 127 s/配置 → 37 配置约 78 分钟。
- R6c `TsI47h`：00:04:49Z→约 00:14Z 完成 7 配置（约 85 s/配置），随后环境中断、GPU 空转至 01:40Z。R6d `v10FQp`：01:40:39Z→约 01:46Z 完成 2 配置（8–9 行），再次被环境终止、GPU 空转至 03:08Z。
- R7（03:08:04Z 起，计划任务托管）：约 6 分钟内完成 3 配置（10–12 行：`generic T1/T2/T4`，约 85–120 s/配置），12/37；剩余 25 配置预计 35–55 分钟，之后驱动自动串行跑 `aggregate` → falsifier `run` → `verify`，避免作业结束后空卡。
