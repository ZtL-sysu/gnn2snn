# R7：第三次环境中断归档、根因修复（会话外托管）、E5 延迟真实增量 9→12/37

- 时间：2026-11-14T03:00:55Z–03:14Z（+0800 11:00–11:14）
- 分支：`agent/visual-conversion/20261113-wb-resume`；发布时本地 HEAD == 远端 `7bd7558f041d135a70d573b5b884ae63068c87c6`（R6d 远端事实写回提交）
- 流水线：contract v4，P0–P3A `passed`，**P4 仍 `running`**（未加 pass），P5–P21 `pending`
- 本检查点只绑定本轮新增的**不可变**证据产物；不覆盖、不改写任何已发布协议、旧证据、旧清单或旧哈希

## 1 唤醒门禁（真实查询，进度以 GitHub 远端为准）

- `git fetch --all --prune` 成功；`git ls-remote origin refs/heads/agent/visual-conversion/20261113-wb-resume` == `git rev-parse origin/...` == `git rev-parse HEAD` == `7bd7558f041d135a70d573b5b884ae63068c87c6`。读取该提交的 `HANDOFF.md`（R6d Next entry）、`checkpoints/20261114T014200Z-r6d-resume-serial-driver/CHECKPOINT.md` 第 8 节、`GPU_QUEUE.md` 顶部、`GOAL_LEDGER.md` 第 3 节、`LOG.md` 末尾。**本地未提交/未上传草稿、会话记忆、旧 LIVE 文字均不作进度依据。**
- `nvidia-smi`：`0 % / 955 MiB / 12 W`，仅显示类 `C+G` 进程。
- PowerShell `Get-CimInstance Win32_Process` 按**完整 CommandLine** 宽匹配 `e3_e5_v1.py|e5_falsifier_v1.py|gpu_queue_driver_v1.py|conda-script.py run|dyg-cold`：0 个计算进程（只命中查询命令自身）。这是 00:04Z 并发事故后规定的唯一查活口径。
- `reverify_frozen_inputs_v1.py`：`status: PASS`、`frozen_items_verified: 12`、`contract_version: v4`、`P4_status: running`、`errors: []`。PROTOCOL-v3 8573171a…、分类头补充 8bde5832…、affine_cal1 1618e3b4…、gcal_v1 b28485dd…、教师 best.pt d26df76d…、5 个官方 PyG processed 哈希、钉死 DCGS commit `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be` 均未漂移（DCGS checkout 实测 `git rev-parse HEAD` 完全匹配）。
- 官方数据根存在（`.../data/pyg_bench`，含 CIFAR10 / MNIST）；Conda `dyg-cold` 环境内 Python 已解析（本机绝对路径去标识，不写入公开产物）。
- 四份脚本 `py_compile` PASS；`gpu_queue_driver_v1.py --dry-print` PASS（四步命令、数据根、钉死 DCGS checkout 全部正确解析，三份报告当时均不存在）。

## 2 第三次环境中断的发现与证据处置（非科学失败）

- R6d 的单一驱动 `v10FQp`（driver PID 5776 / latency 11560，01:40:39Z 启动）在约 **01:46Z 被环境/会话边界终止**：驱动日志含 `DRIVER_START` 与 `STEP_START` 但无 `STEP_END`、无 traceback、无 `DRIVER_END`；`gpu_queue_driver_v1.status.jsonl` 不存在（step 01 从未完成）；step-01 日志止于 config 9；`e5_latency_v1.partial.jsonl` 恰为 **9/37** 行、最后写入 01:46Z。GPU 此后空转约 **75 分钟**。
- 归档（复制，不删除原件）：`logs/e5-latency-interruption-v2/` 含 `interrupted-20261114T0302Z.driver.log`、`interrupted-20261114T0302Z.step01.log`、`interrupted-20261114T0302Z.partial.jsonl`、`SHA256SUMS.txt`、`INTERRUPTED.json`。SHA-256：
  - driver `ad7bf13ed570c7b39fa01a03a88daa11110d22327b5a16950d7684f1aad255ef`
  - step01 `6f67a8e88315e4ae1cf6d68f93ad34edfb7d45e9cfbbc0e6f0596638acfbc3f0`
  - partial `8730a3f69c18c6f64f81d05008168512336ba55a88bc6566fe8e8dc70c137bd2`
- **9 条增量数值判为有效**：1–7 行来自并发事故处置后启动的单一干净作业 `TsI47h`，8–9 行来自 R6d 的单一串行驱动 `v10FQp`；两者均用复核后的 `e3_e5_v1.py`（冻结阈值在计时循环外构造）、全 5000 图固定顺序、1 warmup / 3 repeats。与 `logs/e3e5-latency-concurrency-incident-v1/` 中**作废**的污染墙钟严格区分。续跑逐行校验 `graphs_per_repeat` 与 `repeat_wall_seconds` 长度后才跳过，不重测、不改写。
- 分类：环境/会话边界中断，不是 GPU 科学失败；失败证据保留，连败计数不累计。

## 3 根因修复：会话外托管（本轮最实质的工程进展）

- 根因：R6c `TsI47h` 与 R6d `v10FQp` 都是**自动化会话的后台进程**，会话边界一到即被终止，分别造成约 82 / 75 分钟 GPU 空转。这是连续第三次同因中断。
- 修复：新增 `research/visual-graphs/continuation_20261113/code/gpu_queue_launcher_v1.py`（stdout/stderr 重定向到日志，避免计划任务无控制台导致 `sys.stdout is None`），并通过**无触发器的 Windows 计划任务** `GNN2SNN-E5-Driver-R7` 启动同一个 `gpu_queue_driver_v1.py`：
  - `Triggers: 0`（不会周期性重复触发）、`MultipleInstances: IgnoreNew`（并发时忽略新实例）、`ExecutionTimeLimit: 24h`、`LogonType: Interactive`、`RunLevel: Limited`
  - 进程树实测：`launcher PID 32844 → driver PID 23620 → latency PID 29800`，launcher 的父进程 **PID 1820 = `C:\WINDOWS\system32\svchost.exe -k netsvcs -p -s Schedule`**（Task Scheduler 服务），**不隶属于本自动化会话**，因此会话结束不会终止作业。
  - 单一作业纪律：启动前后各做一次完整 CommandLine 宽匹配，确认只有**一个** `e3_e5_v1.py latency` 进程；未启动第二个 GPU 作业。
- 命令与参数**一字未改**：`01 latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume` → `02 aggregate` → `03 e5_falsifier_v1.py run --device cuda --batches 40` → `04 verify`。未缩短批次、未降 repeats、未挑 T/seed、未改阈值或 pJ 常数。

## 4 真实 GPU 结果与占用事实（实测，非计划）

| 项 | 实测值 |
|----|--------|
| 后台句柄 | 计划任务 `GNN2SNN-E5-Driver-R7`（无触发器，State `Running`） |
| launcher / driver / latency PID | 32844 / 23620 / 29800 |
| 启动时间 | 2026-11-14T03:08:04Z |
| 续跑起点 | `{"resumed_completed_configs": 9, "configs_in_this_run": 37}` |
| `nvidia-smi` | **88–92 % / 5155–5159 MiB / 110.67–118.39 W**（03:08–03:14Z 多次实测） |
| 增量进度 | **9/37 → 12/37**（本轮新测 3 个配置） |
| 本轮新测中位墙钟 | `generic T1` 1.9336 ms/graph（9.6681 s/5000 图）、`generic T2` 3.7015 ms/graph（18.5077 s）、`generic T4` 7.1592 ms/graph（35.7962 s），各 3 次重复中位数、1 warmup、全 5000 图固定顺序 |
| 总日志 | `logs/gpu_queue_driver_v1.driver.log`（追加）、`logs/gpu_queue_driver_v1.launcher.log`、`logs/gpu_queue_driver_v1/01_e5_latency_resume.log`（追加） |
| 恢复命令 | 仅在整条链被宽匹配确认消失后：`Start-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7'`（幂等；`IgnoreNew` 会在已有实例运行时忽略新请求）。驱动自身以 `--resume` 跳过已完成配置。 |

`METRICS.csv` 是**发布时刻的 12 配置 / 24 行固定列快照**（SHA-256 `f17349a5…`），标注 `resume_valid_clean_baseline`（1–9 行）与 `measured_R7`（10–12 行）；**活文件 `e5_latency_v1.partial.jsonl` 仍在增长，不进清单**，后续轮次不得改写本快照，只能新建检查点。

## 5 清单纪律

- 本检查点 `manifest.json` 只绑定 7 个**本轮新增且不再改动**的不可变产物：`gpu_queue_launcher_v1.py`、三份中断快照、`SHA256SUMS.txt`、`INTERRUPTED.json`、`METRICS.csv`。
- **不绑定**任何活文件：`HANDOFF.md` / `LOG.md` / `GOAL_LEDGER.md` / `GPU_QUEUE.md`、驱动总日志、launcher 日志、逐步日志、`status.jsonl`、仍在追加的 `e5_latency_v1.partial.jsonl`。沿用 R4b/R5/R6b/R6d 惯例。
- 已发布的 R6(`fc288b5`)/R6b(`62eddfa3`)/R6c(`ec297aac`)/R6d(`63ece4bb`, `7bd7558f`) 检查点与清单全部原样保留，未改写、未删除、未重算哈希。
- 本地 `gen_r7_metrics.py` 是 `%TEMP%` 下的一次性派生工具，不上传、不作证据；产物 `METRICS.csv` 由增量 JSONL 直接派生，数值未手抄。
- `logs/gpu_queue_driver_v1.launcher.log` 含本机 Python 绝对路径，**仅本地保留、不上传**；公开的驱动事实取自已 scrub（`<HOME>`）的 `gpu_queue_driver_v1.driver.log` 与 step 日志。其余本地未跟踪的历史草稿/带绝对路径原始日志沿用 R4b/R5 惯例：不作进度依据、不上传，本轮未改写。
- 本检查点发布后 GPU 作业仍在推进，`partial.jsonl` 行数会继续增长；`METRICS.csv` 是**发布时刻快照**，不回溯改写，后续轮次只能在新建检查点里发布新快照。
- **本地未上传脚本的定性（用户要求逐项审查/记录）**：本轮新增并上传的脚本只有 `gpu_queue_launcher_v1.py`（`py_compile` PASS、`--dry-print` PASS、实跑成功、7 产物清单绑定）。另有 9 个历史未跟踪脚本 `accept_admission_v4.py`、`accept_revision_v2.py`、`accept_revision_v3.py`、`accept_startup.py`、`finalize_sync_v1.py`、`prepare_sync_checkpoint.py`、`publish_e1_checkpoint.py`、`run_correctness_checks.py`、`verify_canonical_repeat.py`，mtime 均为 2026-11-13 22:57（早于 R6 23:37），属旧 run-root 草稿、非本轮成果、不在任何已发布清单中、未经本轮预检。按用户「或明确记录为弃用」的口径：**一律标记为弃用**，不上传、不执行、不作进度依据，保留本地字节不动。`%TEMP%/gen_r7_metrics.py` 是一次性派生工具，同样不上传、不作证据。

## 6 科学状态（无任何验收通过）

- `e5_latency_v1.json`、`e3_e5_criterion_v1.json`、`e5_gate_bound_v1.json` **均尚未生成**；`aggregate` 未跑；驱动 step 02/03/04 尚未开始。
- **E1 仍 `NOT EVALUABLE / E1 NOT PASSED`**；T4 0.0667 pp / T8 0.0467 pp 的精度条件单独满足，但完整已知成本与匹配墙钟尚未合并；E3 成本账的 9 项 unresolved items 仍按 PROTOCOL-v3「Cost with unresolved omitted work cannot pass」阻止成本腿给 pass。
- P4 `running`；S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中。无验证集-only Story 判定、无新测试集推理、无六节新稿、无 ≥36 篇引用核查、无独立 CCFA 评审。
- 本轮成果是**根因修复 + 真实 GPU 作业持续满载 + 3 个新配置的匹配墙钟增量**，不是科学验收结论；不把启动、归档、文档写回或哈希复核当作科学进展。

## 7 局限

- 墙钟仍是稠密 PyTorch 原型在本机 3060 上的实测，**不是**芯片能耗/硬件延迟/加速比；不得用事件数冒充节能。
- 计划任务托管规避了「会话边界终止」这一已观测根因，但不能排除断电、驱动崩溃或 24h 执行上限；`--resume` 逐配置增量与 `IgnoreNew` 已把恢复成本降到最低。后续心跳仍须先宽匹配查活再决定是否动作。
- E5 falsifier 结果只有在「开关全关与冻结 `affine_mode_forward` 一致 ≤1e-4」保真校验通过后才可采信。

## 8 下个入口与恢复命令

1. 先 `git fetch --all --prune`，再 `nvidia-smi` + PowerShell `Get-CimInstance Win32_Process` **完整 CommandLine 宽匹配**查活 `GNN2SNN-E5-Driver-R7` 的进程链；`Get-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7'`。**在跑则只监督、收集已完成增量，绝不重启、绝不起第二个 GPU 作业。**
2. 驱动跑完后（检查 `gpu_queue_driver_v1.status.jsonl` 四步 rc 与 `DRIVER_END`），收集 `e5_latency_v1.json` + `latency_METRICS.csv`、`e3_e5_criterion_v1.json`、`e5_gate_bound_v1.json` + `verify`，发布**结果**检查点（清单仍只绑不可变产物）。
3. 在**同一次唤醒**里启动 E2/E4/E6 忠实对照（NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN）的第一个 GPU 作业，再排 MNIST Gated / DGCNN ModelNet40；随后验证集-only Story 判定、冻结后一次新测试确认、六节全新终稿、≥36 篇经核查引用且近 3–5 年占多数、去重审计、当前完整稿实质独立 CCFA≥7/10。
4. 若链再次消失：归档驱动日志与增量快照（SHA-256）、写 `INTERRUPTED.json`，先确认无任何同脚本进程，再 `Start-ScheduledTask -TaskName 'GNN2SNN-E5-Driver-R7'`（参数完全不变），并在检查点如实记录「环境终止导致续跑，非科学失败」。
5. 停止条件不变：冻结输入哈希漂移、canonical 索引/标签漂移、协议/源码/时钟不一致、超出 contract v4 计算上限、用户叫停。失败证据一律保留。

## 9 同步口径

本检查点 push 到自己的 agent 分支并以 `git ls-remote` 核对远端 SHA 后，只能报**同步成功**；**科学验收通过仍不成立**（见第 6 节）。
