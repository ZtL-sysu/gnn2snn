# R6d：干净墙钟作业的环境中断归档、不变参数续跑、单一串行 GPU 驱动启动

- 时间：2026-11-14T01:36Z–01:42Z（+0800 09:36–09:42）
- 分支：`agent/visual-conversion/20261113-wb-resume`；发布时本地 HEAD == 远端 `a2f59acc161cc4bf4bb39d34c84942c9dbd1f1c2`（R6c 远端事实写回提交）
- 流水线：contract v4，P0–P3A `passed`，**P4 仍 `running`**（未加 pass），P5–P21 `pending`
- 本检查点只绑定本轮新增的**不可变**证据产物；不覆盖、不改写任何已发布协议、旧证据、旧清单或旧哈希

## 1 唤醒门禁（真实查询，进度以 GitHub 远端为准）

- `git fetch --all --prune` 成功；`git ls-remote origin refs/heads/agent/visual-conversion/20261113-wb-resume` = `a2f59acc161cc4bf4bb39d34c84942c9dbd1f1c2` = `git rev-parse HEAD`。读取该提交的 `HANDOFF.md`（R6c Next entry）、`checkpoints/20261114T000800Z-e3-cost-e5-clean-latency/CHECKPOINT.md` 第 8 节、`GOAL_LEDGER.md` 第 3 节、`LOG.md` 末尾。
- `nvidia-smi`：`0 % / 955 MiB / 12288 MiB`，只有显示类 `C+G` 进程。
- `Get-CimInstance Win32_Process` 按**完整 CommandLine** 宽匹配 `e3_e5_v1.py|e5_falsifier_v1.py|conda-script.py run`：返回 0 个计算进程（`NO_PYTHON_PROCESS`）。这是 2026-11-14T00:04Z 并发事故后规定的唯一查活口径。
- `reverify_frozen_inputs_v1.py`：`status: PASS`、`frozen_items_verified: 12`、`contract_version: v4`、`P4_status: running`、`errors: []`。PROTOCOL-v3 8573171a…、分类头补充 8bde5832…、affine_cal1 1618e3b4…、gcal_v1 b28485dd…、教师 best.pt d26df76d…、5 个官方 PyG processed 哈希、钉死 DCGS commit 5e0e4c73… 均未漂移。
- Conda `dyg-cold` 实测：Python 3.13.13 / torch 2.11.0+cu128 / CUDA True / NVIDIA GeForce RTX 3060 / PyG 2.8.0.post1。

## 2 环境中断的发现与证据处置（非科学失败）

- R6c 的单一干净作业 `TsI47h`（`latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume`，00:04:49Z 启动）在约 **00:14Z 被环境/会话边界终止**：驱动日志 `logs/e5_latency_v1.clean.driver.log` 只有一行 `CLEAN_JOB_START`，无 traceback、无完成行；`results/e3-e5-v1/e5_latency_v1.partial.jsonl` 仅 7/37 配置且最后修改 00:14Z；`e5_latency_v1.json` 不存在。GPU 此后空转约 82 分钟。
- 归档（复制，不删除原件）：`logs/e5-latency-interruption-v1/` 含 `interrupted-20261114T0014Z.driver.log`（SHA256 `db5363a5…`）、`interrupted-20261114T0014Z.partial.jsonl`（SHA256 `4a28bf41…`）、`SHA256SUMS.txt`、`INTERRUPTED.json`。
- **7 条增量数值判为有效**：由并发事故处置后启动的单一干净作业、复核后的 `e3_e5_v1.py`（冻结阈值在计时循环外构造）、全 5000 图固定顺序、1 warmup / 3 repeats 产生；与 `logs/e3e5-latency-concurrency-incident-v1/` 中**作废**的污染墙钟严格区分。续跑逐行校验 `graphs_per_repeat` 与 `repeat_wall_seconds` 长度后才跳过，不重测、不改写。
- 分类：环境/会话边界中断，不是 GPU 科学失败；失败证据保留，连败计数不累计。

## 3 恢复动作：不变参数续跑 + 单一串行驱动（真实 GPU 作业）

- 新增 `research/visual-graphs/continuation_20261113/code/gpu_queue_driver_v1.py`：一个后台进程串行消费队首，步骤为
  1. `e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume`
  2. `e3_e5_v1.py aggregate`（CPU）
  3. `e5_falsifier_v1.py run --device cuda --batches 40`（GPU）
  4. `e5_falsifier_v1.py verify`（CPU）
- 每步独立原始日志 `logs/gpu_queue_driver_v1/<step>.log` 与 rc/耗时 `logs/gpu_queue_driver_v1.status.jsonl`；**单步失败保留原始证据并继续下一步，不允许整轮停摆**；缺报告时自动插入一次参数完全相同的重试（`01b`/`02b`/`03b`/`04b`）。未缩短批次、未降 repeats、未挑 T/seed、未改阈值或 pJ 常数。
- 预检：`py_compile` PASS；`--dry-print` PASS（正确解析 4 步、数据根与钉死 DCGS checkout）；加入 `scrub()` 去标识，公开日志不含本机用户名/家目录。
- 查活与并发纪律：启动前后各做一次完整 CommandLine 宽匹配，确认 conda wrapper → driver → **唯一一个** `e3_e5_v1.py latency` 子进程；未启动第二个 GPU 作业。

## 4 GPU 占用事实（实测，非计划）

| 项 | 实测值 |
|----|--------|
| 后台句柄 | `v10FQp` |
| 驱动 PID / latency 子进程 PID | 5776 / 11560 |
| 启动时间 | 2026-11-14T01:40:39Z |
| 续跑起点 | `{"resumed_completed_configs": 7, "configs_in_this_run": 37}` |
| `nvidia-smi` | **90 % utilisation / 5130 MiB / 112.24 W**（01:41Z） |
| 总日志 | `logs/gpu_queue_driver_v1.driver.log` |
| 恢复命令 | 设 `GNN2SNN_DATA_ROOT` 为官方 PyG 数据根后执行 `conda run -n dyg-cold python -X utf8 research/visual-graphs/continuation_20261113/code/gpu_queue_driver_v1.py`（`DCGS_REPO` 默认 `<tempdir>/dcgs-5e0e4c73`） |

仅在整条驱动链被宽匹配确认消失后才允许重启；在跑期间绝不重启、绝不起第二个 GPU 作业。

## 5 清单纪律

- 本检查点 `manifest.json` 只绑定 5 个**本轮新增且不再改动**的不可变产物：`gpu_queue_driver_v1.py`、两份中断快照、`SHA256SUMS.txt`、`INTERRUPTED.json`。
- **不绑定**任何活文件：`HANDOFF.md` / `LOG.md` / `GOAL_LEDGER.md` / `GPU_QUEUE.md`、驱动总日志、逐步日志、`status.jsonl`、仍在追加的 `e5_latency_v1.partial.jsonl`。沿用 R4b/R5/R6b 惯例。
- 已发布的 R6(`fc288b5`)/R6b(`62eddfa3`)/R6c(`ec297aac`) 检查点与清单全部原样保留，未改写、未删除、未重算哈希。

## 6 科学状态（无任何验收通过）

- `e5_latency_v1.json`、`e3_e5_criterion_v1.json`、`e5_gate_bound_v1.json` **均尚未生成**；`aggregate` 未跑。
- **E1 仍 `NOT EVALUABLE / E1 NOT PASSED`**；T4 0.0667 pp / T8 0.0467 pp 的精度条件单独满足，但完整已知成本与匹配墙钟尚未合并；E3 成本账的 9 项 unresolved items 仍按 PROTOCOL-v3「Cost with unresolved omitted work cannot pass」阻止成本腿给 pass。
- P4 `running`；S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中。无验证集-only Story 判定、无新测试集推理、无六节新稿、无 ≥36 篇引用核查、无独立 CCFA 评审。
- 本轮成果是**恢复并持续占用真实 GPU 作业 + 完整中断证据**，不是科学结论；不把启动、归档、文档写回或哈希复核当作科学进展。

## 7 局限

- 墙钟仍是稠密 PyTorch 原型在本机 3060 上的实测，**不是**芯片能耗/硬件延迟/加速比；不得用事件数冒充节能。
- 驱动链可能被下一次会话边界再次终止；已用逐配置增量 JSONL 与自动重试把恢复成本降到最低，但不能保证单次唤醒内跑完。
- E5 falsifier 结果只有在「开关全关与冻结 `affine_mode_forward` 一致 ≤1e-4」保真校验通过后才可采信。

## 8 下个入口与恢复命令

1. 先宽匹配查活 `v10FQp`；在跑则只监督、收集已完成增量，不重启。
2. 驱动跑完后收集 `e5_latency_v1.json` + `latency_METRICS.csv`、`e3_e5_criterion_v1.json`、`e5_gate_bound_v1.json` + `verify`，发布**结果**检查点（清单仍只绑不可变产物）。
3. 在同一次唤醒里启动 E2/E4/E6 忠实对照（NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN）的第一个 GPU 作业，再排 MNIST Gated / DGCNN ModelNet40；随后验证集-only Story 判定、冻结后一次新测试确认、六节全新终稿、≥36 篇经核查引用且近 3–5 年占多数、去重审计、当前完整稿实质独立 CCFA≥7/10。
4. 若链再次被环境终止：归档驱动日志与增量快照（SHA-256）、写 `INTERRUPTED.json`，再用**完全相同**参数重启驱动，并在检查点如实记录「环境终止导致续跑，非科学失败」。
5. 停止条件不变：冻结输入哈希漂移、canonical 索引/标签漂移、协议/源码/时钟不一致、超出 contract v4 计算上限、用户叫停。失败证据一律保留。

## 9 同步口径

本检查点 push 到自己的 agent 分支并以 `git ls-remote` 核对远端 SHA 后，只能报**同步成功**；**科学验收通过仍不成立**（见第 6 节）。
