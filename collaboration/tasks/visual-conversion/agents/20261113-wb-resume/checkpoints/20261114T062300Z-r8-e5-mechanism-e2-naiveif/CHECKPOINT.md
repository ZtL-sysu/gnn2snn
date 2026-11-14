# R8 检查点 — E5 机制证据（v3 实测）+ E2 NaiveIF 忠实对照首例（3 seeds 实测）

- 时间：2026-11-14T06:23Z（+0800 14:23）。分支 `agent/visual-conversion/20261113-wb-resume`。
- **进度权威**：本次唤醒先 `git fetch --all --prune`，`git ls-remote origin refs/heads/agent/visual-conversion/20261113-wb-resume` == `origin/...` == 本地 HEAD == `0b472b8cbc1b9170362a23845691222340e5427e`（R7b-sync 补传 + 账本写回）。读取了该提交的 `HANDOFF.md` R7b-sync「Next wake-up」、`GPU_QUEUE.md` 顶部、`GOAL_LEDGER.md` 第 3 节锁定证据清单、`LOG.md` 末尾。**本地未上传草稿、会话记忆、旧 LIVE 文字均未当作进度**；本检查点只登记本次真实 GPU 结果与已预检代码。
- 门禁：`reverify_frozen_inputs_v1.py` → `status PASS` / `frozen_items_verified 12` / `contract_version v4` / `P4_status running` / `errors []`；钉死 DCGS checkout `git rev-parse HEAD` == `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`；官方 PyG 数据根固定；canonical baseline `7c0727a2…` 复核通过。

## 1 唤醒发现：第四次环境中断（不是科学失败）

- `nvidia-smi`：`0 % / 958 MiB / 12 W`，仅桌面合成进程；PowerShell `Get-CimInstance Win32_Process` 按完整 `CommandLine` 宽匹配 `e3_e5_v1|e5_falsifier|e2_naiveif|gpu_queue_driver|gpu_queue_launcher|dyg-cold` 返回 **0 个计算进程**。
- 无触发器计划任务 `GNN2SNN-GPU-Queue-V3` 为 `Disabled`，`LastResult=267014`（0x41306 = 上一次运行被终止），`LastRunTime 13:42:05 +0800`。`gpu_queue_driver_v3.driver.log` 止于 `06:01:50Z STEP_START 06_e2_naiveif_calibrate_seed43`，无 `STEP_END`、无 traceback、无 `DRIVER_END`；`status.jsonl` 无第 06 步记录；`naiveif_calibration_seed43_v1.json` 当时不存在。GPU 自 ~13:42 起空转约 28 分钟。
- 判定为**环境/会话边界终止**（前三次同因，R7 已用计划任务修复；本次任务被显式禁用），非 GPU 或科学失败，连败计数不累计。中断前的 seed41/seed42 全训练校准报告与 v3 falsifier 结果均有效、未受污染。

## 2 本轮真实 GPU 作业（单一链，无并发）

- 恢复动作：`Enable-ScheduledTask` + `Start-ScheduledTask -TaskName 'GNN2SNN-GPU-Queue-V3'`（幂等，`IgnoreNew`）。唯一进程链 `launcher PID 22092（父 1820 = svchost.exe -k netsvcs -p -s Schedule）→ driver PID 33172 → 单个子作业`。驱动按 `logs/gpu_queue_v3.json` 队首串行消费、每步后重读队列，故可在作业运行时追加下一步。
- `01_e5_falsifier_v3_run` `rc=0`（05:42:05Z→05:51:02Z，**536.3 s 真实 GPU**）→ `results/e5-falsifier-v1/e5_gate_bound_v1.json` + `falsifier_METRICS.csv`；`02 verify` `rc=0`。此前 v1/v2 的两次实现缺陷与失败日志已归档、未改写。
- `06_e2_naiveif_calibrate_seed43` `rc=0`（06:13:42Z，25.8 s）→ 全 45000 官方训练图、10 个 ReLU site 的 label-free per-channel 最大预激活。`07 verify` `rc=0`，三份 stage-1 报告齐全。
- `08 precheck` `rc=0`；`09/10/11 e2_naiveif_eval_v1.py run --device cuda --seed 41/42/43` 各 `rc=0`（各 28.3 s，全 5000 官方验证图 × T{1,2,4,8}）；`12 verify` `rc=0`。
- **实测 GPU 占用**：stage-2 执行期间 `nvidia-smi` 为 **95 % / 1823 MiB / 126.67 W** 与 **91 % / 1823 MiB / 130.36 W**。宽匹配确认任何时刻只有一条计算链，绝无第二个 `latency`/falsifier/calibrate 并发。
- 队列继续消费：`13_e5_falsifier_repro_v1.py run --device cuda --batches 40` → `14 verify` → `15 compare`。这是一个**独立确定性复现检查**（约 9 分钟真实 GPU），不是新的科学结论，且不覆盖已发布的 canonical 报告。

## 3 E5 机制证据（真实数据，非计划）

- `e5_gate_bound_v1.json`（SHA-256 `1fb6b433…`）：`status = real_data_gate_bound_and_falsifiers_no_accuracy_or_energy_claim`；覆盖 40 批 / 全 5000 官方验证图 / 固定顺序 / label-free / T{1,2,4,8} / 每层 4 层。
- **保真校验**：`fidelity_check.checked_pairs = 160`，`frozen_vs_instrumented_max_abs_diff = 0.0`（门槛 `1e-4`）——插桩前向与冻结 `affine_mode_forward` 逐位一致，反事实才有意义。
- **falsifier 判定**：`fresh_vs_stale_gates.per_T_gate_refresh_falsified = {1: true, 2: false, 4: false, 8: false}`（仅 T1 因只有一拍而平凡成立）；`raw_vs_decoded_skip.per_T_distinction_falsified` 在 T1/T2/T4/T8 均为 `false`。
- `falsifier_METRICS.csv`（SHA-256 `cdf5f56d…`）：121 行含表头 = **120 条固定列指标**。
- 边界声明原文保留：`|m(z',b')-m(z,b)| <= max_i|b_i'-b_i| + R(b')tanh(eta/2)`；这是**局部实数算术消息界**，不是浮点证书、精度保证、效率或新颖性结果；宽松界如实报宽松。无芯片能耗 / 硬件延迟 / 加速比声明。
- S5 的「机制腿」证据缺口闭合，但 **S5 仍 FAIL**：E3 成本账仍保留 9 项 unresolved（「Cost with unresolved omitted work cannot pass」），S4 忠实对照远未齐备，本轮未做任何独立评审。不因补齐一项就抬高验收状态。

## 4 E2 NaiveIF 忠实对照首例（真实弱结果，如实登记）

- 定义：标准 data-normalised IF，constant input、subtractive reset、跨拍膜电位、per-channel 阈值来自**全部 45000 官方训练图**的 label-free 最大正预激活；sigmoid 门 / BN / residual / pooling **完全不做算子适配**。退化是实测结果，未做任何假设或跳过。
- stage 1：三份 `naiveif_calibration_seed{41,42,43}_v1.json`（SHA `5fd0966c…` / `d21eb8ab…` / `fcf7892a…`），每份 45000 图 / 10 sites，`e2_naiveif_v1.py verify` PASS。
- stage 2：三份 `naiveif_report_v1.json`（SHA `e90ccb4a…` / `aa581a2a…` / `6ef24c95…`）+ 每 seed 4 份逐样本 NPZ + 16 行固定列 `naiveif_METRICS.csv`，`verify` PASS（NPZ 哈希、canonical 索引/标签逐位一致、行数）。
- **硬门禁**：`instrumented_vs_frozen_max_abs_diff = 0.0`（三个 seed 全部）；canonical 索引/标签逐位一致；seed41 首批教师 logits 与 canonical 逐位一致（`seed41_canonical_teacher_bitwise = true`）；处理前后 processed hashes 一致；输出目录已存在即拒绝。
- **实测四指标（accuracy / teacher agreement / logit MSE / teacher-to-output KL）**：

  | seed | T1 acc | T2 acc | T4 acc | T8 acc |
  |------|--------|--------|--------|--------|
  | 41 | 0.0972 | 0.0972 | 0.1110 | 0.1680 |
  | 42 | 0.1038 | 0.1038 | 0.1316 | 0.1104 |
  | 43 | 0.1038 | 0.1038 | 0.1080 | 0.1476 |

  相对 ANN 教师约 0.7066 的验证精度是**严重退化**，且随 T 增长并未可靠恢复；`teacher_agreement` 0.107–0.189、`logit_mse` 16.8–64.2、KL 1.69–10.27。这是无算子适配基线的真实测量结果，**不删弱结果、不用公开数字冒充实测、不假定适配失败**。它不构成方法优势、能耗、加速或新颖性声明。

## 5 代码审查与失败证据（全部保留）

- `e2_naiveif_eval_v1.py` 的**未发布 v1 草稿**逐字节保留于 `logs/e2-naiveif-eval-draft-v1/e2_naiveif_eval_v1.draft.py`（SHA `2d26d947…`），`DEFECTS.json` 记录静态审查发现的 4 个工具缺陷：绕开作者 `GatedGCNLayer.forward` 手写层数学、label-free 批次只取 1 个标签、`--limit` 冒烟写入 canonical 输出目录、`coverage.graphs` 空生成器表达式。
- 修复采用与 stage-1 完全相同的 **10 个 forward hook 位点**（`bn_node_h` / `bn_node_e` / `FC_layers[0]` / `FC_layers[1]`），调用作者 `model(batch)` 原样前向；作者随后的 `F.relu` 对非负脉冲序列是恒等映射。`py_compile` PASS、`precheck` PASS、CPU 2 批冒烟 PASS（fidelity 0.0 / 16 行 / 4 NPZ）。
- `e5_falsifier_v3.py` 被已发布 R7b-sync 补充清单绑定，**字节未改**；确定性复现改用只读 wrapper `e5_falsifier_repro_v1.py`（仅重定向 `OUT`）+ `smoke_e5_falsifier_repro_v1.py`。
- 保留的工具失败：`logs/e2_naiveif_v1.prefailure.log`（`TypeError: precheck() takes 0 positional arguments but 1 was given` 与另两处预修缺陷）、`logs/e5_falsifier_repro_smoke_v1.inline-failure.log`（内联 `python -c` 反斜杠引号 `SyntaxError`，canonical 报告哈希前后均为 `1fb6b433…` 未受影响）、`logs/archive_v2_failure_v1.failure.log`。全部归类为工具缺陷，非科学失败。

## 6 清单纪律与排除项

- 本检查点的 SHA-256 清单**只绑定不可变证据产物**（结果 JSON/CSV/NPZ、脚本、失败证据、草稿归档）与自身 `CHECKPOINT.md`。
- **不绑定**任何活文件：`HANDOFF.md` / `LOG.md` / `GOAL_LEDGER.md` / `GPU_QUEUE.md` / `logs/gpu_queue_v3.json` / 驱动日志 / `status.jsonl` / partial JSONL / 仍在写入的 repro 结果。
- 不上传：原始 traceback 日志（含本机身份路径）、未发布草稿检查点目录、`.workbuddy/`、数据缓存、教师权重包、任何单文件 >50 MiB 的内容、凭证或私人对话。公开文本已去标识。
- 已发布的 R6/R6b/R6c/R6d/R7/R7b2/R7b-sync 检查点与清单**原样保留，未改写、未删除**。

## 7 科学状态（如实，不粉饰）

- **E1 仍 `NOT PASSED (method advantage not established)`**：R7b2 双门槛重算的证据缺口已闭合，精度腿满足但成本腿与匹配墙钟腿均为 `false`，9 项 unresolved 保留；`story_decision: DEFERRED`、`new_test_inference_performed: false`。本轮**没有**降低门槛、**没有**加 P4 pass 标志。
- P0–P3A `passed`，**P4 仍 `running`**，P5–P21 `pending`。S1 PASS；**S2/S3/S4/S5/S7/S8 仍 FAIL**（S4 仅完成 NaiveIF 一例，官方 DCGS / QCFS / FTBC / SNN Calibration / One-Spike / 直接训练 SpikingGCN 与 MNIST Gated / DGCNN ModelNet40 均未做）；S6 待验证；S9 进行中。
- 无 Story 书面判定、无冻结后一次新测试确认、无全新六节终稿、无 ≥36 篇经核查引用、无去重审计、无当前完整稿的实质独立 CCFA≥7/10 评审。
- **同步成功 ≠ 科学验收通过**；本检查点登记的是真实 GPU 证据增量与工程修复，不是验收结论。

## 8 下个入口（严格按队列与冻结协议）

1. 监督 `GNN2SNN-GPU-Queue-V3` 的 `13 run` → `14 verify` → `15 compare`（**在跑绝不重启、绝不起第二个 GPU 作业**）；`compare` 要求除时间戳外数值报告完全相同且固定列 CSV 逐字节相同，不一致就如实记 FAIL。收集后单独发布复现检查点。
2. 队列消费完必须**同一次唤醒**继续追加真实 GPU 作业，禁止空卡：E2/E4/E6 其余忠实对照（官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN，均需 3 seeds 全 5000 验证图），再 MNIST Gated / DGCNN ModelNet40 跨任务；弱结果照实跑、照实报。
3. 之后才做：仅基于验证集的 Story 书面判定 → 冻结后一次新测试确认（含历史暴露披露）→ 全新六节终稿 → ≥36 篇经核查引用且近 3–5 年占多数 → 去重审计 → 当前完整稿实质独立 CCFA≥7/10。
4. 停止条件：冻结输入哈希漂移、canonical 索引/标签漂移、协议源码时钟不一致、超出 contract v4 计算上限、用户叫停。所有失败证据保留。
5. 查活：`Get-ScheduledTask -TaskName 'GNN2SNN-GPU-Queue-V3'` + `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gpu_queue_driver_v3|e5_falsifier|e2_naiveif|e3_e5_v1' }`；整条链消失才 `Start-ScheduledTask` 幂等续跑。
