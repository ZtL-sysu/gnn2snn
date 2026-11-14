# R7b2（权威）：E5 完整匹配墙钟 37/37 完成、E1 双门槛有证据重算、falsifier 缺陷修复与 v2 重跑

> **替代关系声明**：本地草稿目录 `checkpoints/20261114T040000Z-r7b-e5-latency-criterion/` 的清单绑定了含本机身份路径的**原始** traceback 日志，上传预检正确报 `forbidden pattern` / `local username present`。按 R4b/R5「清单不可变、不覆盖、新建检查点并声明替代」惯例，该草稿目录**从未 push、不上传、其清单作废、其事实已合并入本检查点**；原始日志仅留本地。本检查点的清单只绑定去标识公开副本 `*.public.log` 与 `PUBLIC_LOG_MAP.json`。已发布的 R6/R6b/R6c/R6d/R7 检查点全部原样保留。

- 时间：2026-11-14T03:56Z–04:06Z（+0800 11:56–12:06）
- 分支：`agent/visual-conversion/20261113-wb-resume`；发布时本地 HEAD == 远端 `6b1fb4a8d521ebef4ba1d956623224ae7d346c8d`（R7 检查点提交）
- 流水线：contract v4，P0–P3A `passed`，**P4 仍 `running`**（未加 pass），P5–P21 `pending`
- 本检查点只绑定不可变证据产物；不覆盖、不改写任何已发布协议、旧证据、旧清单或旧哈希。R7 检查点 `20261114T031100Z-r7-task-scheduler-driver/` 全部原样保留。

## 1 本轮真实 GPU 结果（非计划、非文档动作）

R7 的会话外托管驱动 `GNN2SNN-E5-Driver-R7` 从 03:08:04Z 连续运行到 03:56:42Z（**48.6 分钟未中断**，前两次分别在约 10、6 分钟被会话边界杀死），完整消费队列 #1→#2→#3→#4：

| 步骤 | rc | 耗时 | 产物 |
|------|----|------|------|
| 01 `latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume` | **0** | 2898.2 s | `results/e3-e5-v1/e5_latency_v1.json` + `latency_METRICS.csv` |
| 02 `aggregate`（CPU） | **0** | 0.1 s | `results/e3-e5-v1/e3_e5_criterion_v1.json` |
| 03/03b `e5_falsifier_v1.py run --device cuda --batches 40` | **1 / 1** | 8.7 / 8.7 s | 失败，见第 3 节，无数字产出 |
| 04 `e5_falsifier_v1.py verify` | **1** | 2.3 s | 缺报告，见第 3 节 |

`DRIVER_END {"latency_report": true, "criterion_report": true, "gate_report": false, "failed": ["03_e5_falsifier_run", "03b_e5_falsifier_run_retry", "04_e5_falsifier_verify"], "steps": 5}`。

- **E5 完整匹配墙钟**：`e5_latency_v1.json`（SHA-256 `6887d2ba…`）含 **37/37 配置**、`resumed_configs: 9`、`total_job_seconds: 2888.5`；全 5000 张官方验证图固定顺序、batch 128、1 warmup、3 次重复中位数；`deterministic_algorithms: true`、`cudnn_benchmark: false`、冻结阈值在计时循环外一次性构造；代表性批次独立校验（128 图 / 15019 节点 / 120152 边、教师与 centered 双前向逐位一致、标签未泄漏）。设备 `NVIDIA GeForce RTX 3060`、torch 2.11.0+cu128。ANN 上界 0.1788 ms/graph；末配置 `centered_shared_theta T8` 11.9920 ms/graph。报告自述 `measured_matched_wall_clock_dense_prototype_not_hardware_latency`，明确**不是**芯片能耗/硬件延迟/加速比。
- **E1 双门槛重算**：`e3_e5_criterion_v1.json`（SHA-256 `a474c1f0…`）——`verdict: "E1 NOT PASSED (method advantage not established)"`、`status: evidence_backed_criterion_evaluation_no_method_advantage_claim`。
  - 精度腿单独满足：T4 差距 **0.0667 pp**、T8 **0.0467 pp**，均 ≤0.3 pp，`accuracy_only_ok: true`。
  - 成本腿 `cost_leg_strictly_lower: false`：T4 AC 候选 69,174,673.484 vs 最强对照 68,964,102.8846 pJ/图；T8 AC 候选 102,182,275.9832 vs 44,065,574.72696 pJ/图。保守 MAC 约定同样不更低。9 项 unresolved items 按 PROTOCOL-v3「Cost with unresolved omitted work cannot pass」继续阻止成本腿给 pass。
  - 墙钟腿 `latency_leg_strictly_lower: false`：T4 候选 6.2449 vs 最强 `centered_shared_theta` 6.2130 ms/图；T8 候选 12.0486 vs 最强 `dc_adapt_six_site` 3.1315 ms/图。
  - `story_decision: DEFERRED`，`new_test_inference_performed: false`，`p4_status: running`。
- **科学状态变化（诚实口径）**：E1 从 `NOT EVALUABLE` 变为**有证据支撑的 `E1 NOT PASSED`** —— 此前缺失的「完整已知成本 + 匹配墙钟」证据缺口已用真实实测闭合，但**冻结验收门槛未降低、未宣称通过、未加 P4 pass**。这是证据完备性的进展，不是方法优势的成立。S3 仍 FAIL；S5 的成本与墙钟腿已有实测，机制/falsifier 腿待 v2 结果。

## 2 E5 机制证据的失败与修复（失败证据全部保留）

- v1 `run` 在第一个 degree 子集累加即崩溃：`RuntimeError: The size of tensor a (9278) must match the size of tensor b (15019) at non-singleton dimension 0`，位于 `Acc.add()` 的 `excess = (a - (b + tol))`。根因：`actual`/`bound` 按布尔节点掩码切片，而 `tol = TOL_ABS + TOL_REL * bound` 是**全量未掩码**张量（15019 节点），二者广播失败；`precheck` 是哈希/结构门禁、不跑 GPU 前向，故未拦下。
- 原始失败日志（run / run-retry / verify）与 `SHA256SUMS.txt`、`FAILURE.json` 归档于 `logs/e5-falsifier-failure-v1/`。**崩溃发生在写报告之前，没有任何 gate-bound / 误差 / falsifier 数值产生，也不引用任何数值。** 原始 traceback 含本机 Python 安装路径，按去标识惯例**只留本地不上传**；公开副本 `*.public.log`（`<HOME>` 替换、错误文本/行号/张量尺寸不变）随 `PUBLIC_LOG_MAP.json` 上传并进清单，上传预检 `identity_or_path_violations: 0`。
- **清单纪律**：`e5_falsifier_v1.py` 被已发布 R6c 15 产物清单绑定，字节**绝不改写**；该清单在本检查点复核仍 `verify` PASS。修复以**新文件** `e5_falsifier_v2.py` 发布，只有一处有防护的改动：当 `tol` 是张量且形状等于 `actual` 时用同一掩码切片。`TIMESTEPS`、`TOL_ABS=1e-4`、`TOL_REL=1e-3`、两份协议哈希、DCGS commit、CSV 固定列、`--batches 40 --device cuda`、报告 schema **全部不变**（`test_e5_falsifier_v2.py` 断言参数逐项相同并通过）。
- 回归测试 CPU PASS：精确复现 v1 的 `RuntimeError`、v2 掩码张量容差/无掩码/标量容差三种调用形状、参数一致性。v2 `precheck` PASS（6 冻结文件 / 5 processed / 6 共享 theta 向量 / DCGS `5e0e4c73…`）。
- **新 GPU 作业已启动**（不停摆）：无触发器计划任务 `GNN2SNN-E5-Falsifier-V2`（`Triggers=0`、`IgnoreNew`、12h 上限）经 `gpu_queue_launcher_v2.py` → `gpu_queue_driver_v2.py` 串行执行 `e5_falsifier_v2.py run --device cuda --batches 40` → `verify`。04:04:08Z 启动，进程链 `launcher 25360 (父 1820 = svchost Schedule) → driver 26076 → 唯一 falsifier 9424`；实测 **GPU 95 % / 4627 MiB / 118 W**，step 日志 `{"batch": 1, "of": 40, "elapsed_sec": 13.7}` 已越过 v1 的崩溃点。不重跑 latency/aggregate（两份报告已发布且不可变）。

## 3 清单纪律

- 本检查点 `manifest.json` 只绑定**不可变**产物：`e5_latency_v1.json`、`latency_METRICS.csv`、`e3_e5_criterion_v1.json`、三份 v1 失败日志、`SHA256SUMS.txt`、`FAILURE.json`、`e5_falsifier_v2.py`、`test_e5_falsifier_v2.py`、`gpu_queue_driver_v2.py`、`gpu_queue_launcher_v2.py`。
- **不绑定**：HANDOFF / LOG / GOAL_LEDGER / GPU_QUEUE、v1/v2 驱动日志、逐步日志、`status.jsonl`、v2 正在追加的 step 日志、仍可能增长的报告。已发布的 R6/R6b/R6c/R6d/R7 检查点与清单全部原样保留。

## 4 局限与下个入口

- 墙钟为稠密 PyTorch 原型实测；`report.limitations` 明示含 Python 循环与 kernel launch 开销、排除数据加载，不得用事件数冒充节能/加速。
- 下个入口：监督 `GNN2SNN-E5-Falsifier-V2` → 收集 `e5_gate_bound_v1.json` + `falsifier_METRICS.csv` + `verify`，发布机制证据结果检查点；随后在**同一次唤醒**启动 E2/E4/E6 忠实对照（NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN）首个 GPU 作业，再排 MNIST Gated / DGCNN ModelNet40；之后验证集-only Story 判定、冻结后一次新测试确认、六节全新终稿、≥36 篇经核查引用且近 3–5 年占多数、去重审计、当前完整稿实质独立 CCFA≥7/10。
- 查活：`Get-ScheduledTask -TaskName 'GNN2SNN-E5-Falsifier-V2'` + PowerShell `Get-CimInstance Win32_Process` 完整 CommandLine 宽匹配；在跑绝不重启、绝不起第二个 GPU 作业。链消失才归档后 `Start-ScheduledTask` 幂等续跑。
- 停止条件不变：冻结输入哈希漂移、canonical 索引/标签漂移、协议/源码/时钟不一致、超出 contract v4 计算上限、用户叫停。失败证据一律保留。

## 5 同步口径

push 后以 `git ls-remote` 核对远端 SHA 只能报**同步成功**；**科学验收通过仍不成立**：E1 `NOT PASSED`、P4 `running`、S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中。
