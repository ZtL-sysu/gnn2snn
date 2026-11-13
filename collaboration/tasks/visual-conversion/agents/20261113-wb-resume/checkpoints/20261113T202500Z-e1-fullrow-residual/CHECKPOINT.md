# R5：E1 残余行补齐、ANN 上界行、全行汇总与判据重算

- 时间：2026-11-13T20:11Z–20:25Z（+0800 2026-11-14 04:11–04:25）
- 分支：`agent/visual-conversion/20261113-wb-resume`；agent：`20261113-wb-resume`；task：`visual-conversion`
- 流水线：contract v4，P0–P3A `passed`，**P4 仍 `running`**（未加 pass），P5–P21 `pending`。状态 `in_progress`。
- 本检查点是 R5 新产物的**权威发布清单**；不覆盖、不修改任何已发布协议、旧证据或旧哈希。

## 1 唤醒动作 1：分支与活跃作业（真实查询，不采信旧日志）

- `git fetch --all --prune` 成功；本地 HEAD 与 `git ls-remote origin refs/heads/agent/visual-conversion/20261113-wb-resume` 均为 `940cd5b0b4adf1b881b077ea5be31a70bff3c6af`。
- `nvidia-smi`：GPU 0 `0 % / 955 MiB / 12288 MiB`；`Get-CimInstance Win32_Process` 精确过滤 `python.exe/pythonw.exe/conda.exe` 返回 **0 个进程** → 无存活作业，可以安全启动新作业。
- 作业结束后复核：GPU `0 % / 955 MiB`，无存活 Python 进程。

## 2 唤醒动作 2：冻结输入硬门禁

`reverify_frozen_inputs_v1.py` 输出 `status: PASS`、`frozen_items_verified: 12`、`contract_version: v4`、`P4_status: running`、`errors: []`。另外直接读取 `pipeline-state.json`：P0/P1/P1A/P1B/P2/P3/P3A 均 `passed`，P4 `running`，未修改。
`e1_residual_v1.py precheck`（环境默认路径版，两次）均 PASS：seeds 41/42/43 的 6 个冻结文件哈希、5 个官方 processed 哈希、14 affine sites、6 个六站点向量、shared-theta 独立逐通道 max 重算、E0/E1 两套源码 verifier 同一 `MTH_diff` 类、DCGS commit `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`。报告见 `logs/e1-residual-precheck-v2.json`。

## 3 保留的失败证据与两处缺陷（未静默丢弃）

- 上一轮后台作业 `7M53yo` 的三个 `conda run` 全部立即失败：`AttributeError: 'NoneType' object has no attribute 'resolve'`（`run()` 未处理 `--data-root/--author-repo` 默认值）。**无任何 GPU 工作、无 partial 输出目录**；三个原始日志已复制为 `*.cli-default-failure.log` 本地保留。
- 重启前静态复核又发现第二处必现缺陷：`run()` 的 ROOT 相对源码清单会对位于仓库根之外的钉死 DCGS 源文件调用 `rel()` 并抛 `ValueError`，同样会在任何 GPU 工作之前中止。已改为以去标识短键 + 起止 SHA-256 绑定这两个外部文件，并由 `author_source_provenance`（相对路径、哈希、head blob oid、pinned commit、E0/E1 类同一性）在作业末尾与 `verify()` 中双重复核。
- 修复后重跑：编译 OK、冻结门禁 PASS、smoke 逻辑未改动、三 seed 正式作业全部 `rc=0`。
- 原始 traceback 含本机身份路径，**不上传**；公开的去标识结构化记录为 `logs/e1-residual-cli-default-failure-v1.json`。

## 4 R5 实跑事实（本机 RTX 3060 / Conda dyg-cold）

- 真实作业句柄：后台任务 `LYwybZ`，驱动日志 `logs/e1-residual-run-v2.driver.log`。
  - `SEED_START 41 20:11:13Z → SEED_END 41 rc=0 20:15:33Z`（251.8 s）
  - `SEED_START 42 20:15:33Z → SEED_END 42 rc=0 20:19:53Z`（252.3 s）
  - `SEED_START 43 20:19:53Z → SEED_END 43 rc=0 20:24:14Z`（252.8 s）
- 每行：全 5000 张官方 CIFAR10 验证图、batch 128、固定顺序、deterministic CUDA、教师两次前向逐位一致；T ∈ {1,2,4,8}，scale 1，阈值验证前固定。
- 新增三行 × 4 T × 3 seeds = **36 份逐样本 NPZ**：`dc_adapt_six_site`（历史六站点 DC-adapt，四个 node residual branches + 两个 hidden MLP readout activations）、`first_analog_shared_theta`、`centered_shared_theta`（同一组冻结 channelwise-max 同向量）。
- 教师精度与冻结值一致：seed41 `0.7066`、seed42 `0.7030`、seed43 `0.7074`。
- **seed41 六站点 DC-adapt 的 T4/T8 logits 与 E0 canonical 数组逐位一致**（`max_abs_difference = 0.0`），证明历史转换器行被忠实复现，而非另写近似实现。
- ANN 上界行（本会话早前生成、本轮独立复验）：3 seeds × 4 T = 12 份 NPZ，`logits == teacher_logits` 逐位一致，accuracy 与报告一致，判定为 upper reference，**不是转换对照**。
- 所有公开 JSON 已扫描：无本机用户名、无本机绝对路径、无凭证。

## 5 独立核验 PASS

`e1_residual_v1.py verify` → `logs/e1-residual-verify-v1.json`，`status: PASS`：

| 项 | 结果 |
|---|---|
| 每 seed 行数 / NPZ 哈希 | 12 / 12（共 36 / 36） |
| canonical 索引标签 | 逐位一致 |
| 四项指标从 NPZ 重算 | 全部可复算 |
| seed41 教师 logits | 与 canonical 逐位一致 |
| seed41 六站点 DC-adapt T4/T8 | 与 E0 canonical 逐位一致 |
| shared-theta 是否等于冻结同模式 T4 | 两个均为 `false`（非平凡别名） |
| 源码 / 数据 / 外部 DCGS 源哈希起止 | 一致 |

补充：`centered_shared_theta` 在 T8 的三 seed 精度恰与 `generic` 相同，但 logits **逐位不同**（max abs diff 0.146 / 0.888 / 0.165），属分类精度巧合而非数值别名；`cse_only`/`generic`/`per_consumer` 的既有别名关系不变，不能算独立方法胜利。

## 6 完整 E1 全行表与判据重算（如实替代 R4b 数字）

`aggregate_e1_fullrows_v1.py` 汇总 10 个行族 × 4 T × 3 seeds = **120 份逐样本证据**，40 个 (mode,T) 三 seed 描述性汇总、560 行固定列 `METRICS.csv`。输出 `results/e1-fullrow-summary-v1/`。

T4 / T8 三 seed 均值（fraction）：

| 行 | T4 | T8 |
|---|---|---|
| ANN（上界，非对照） | 0.705667 | 0.705667 |
| **centered（候选）** | **0.705267** | **0.705667** |
| first_analog | 0.705533 | 0.706000 |
| generic / per_consumer / cse_only | 0.705467 | 0.705467 |
| dc_adapt_analog_head | 0.705600 | 0.705933 |
| dc_adapt_six_site | 0.705867 | **0.706133** |
| first_analog_shared_theta | 0.705667 | 0.706000 |
| centered_shared_theta | **0.705933** | 0.705467 |

按补充协议「ALL conversion controls（含两个 DC readout 版本），不含 ANN、不含候选」取最强对照：

| T | 最强对照 | 对照均值 | 候选均值 | 差距 | ≤0.3pp |
|---|---|---|---|---|---|
| T4 | `centered_shared_theta` | 0.705933 | 0.705267 | **0.0667 pp** | 是 |
| T8 | `dc_adapt_six_site` | 0.706133 | 0.705667 | **0.0467 pp** | 是 |

**R4b 的 0.0333 pp / 最强对照 `dc_adapt_analog_head` 与 `first_analog` 已被本表取代**：那是在缺少六站点 DC-adapt 与 shared-theta 行时算出的不完整参考集。补齐全部对照后差距扩大约一倍，但仍在 0.3pp 内；精度条件**仍单独满足**。

`criterion.verdict = NOT EVALUABLE / E1 NOT PASSED`。协议要求 BOTH T4/T8 的 0.3pp 精度条件 **AND** 更低的完整已知操作成本或匹配墙钟；`complete_known_operation_cost` 与 `matched_wall_clock_latency` 仍缺，`operation_ledger` 仅为部分参考计数器（明确写明稠密算术、访存、硬件延迟与能耗均未测）。不得用精度单独通过，不得用事件数冒充节能/加速。`missing_required_rows = [complete_known_operation_cost, matched_wall_clock_latency]`。

## 7 明确不能下的结论

- **科学验收通过：不成立。** P4 仍 `running`；S2 仍 FAIL（10 行逐样本精度/agreement/MSE/KL 证据已齐 120 份且核验 PASS，但 E1 每行仍缺完整已知成本、未决项与 ms/graph，故不判 PASS）；S3 仍 FAIL（双门槛 NOT EVALUABLE）；S4/S5/S7/S8 仍 FAIL；S6 待验证。无 Story 判定、无新测试集推理、无六节新稿、无 ≥36 篇引用核查、无独立 CCFA 评审。
- **同步成功**仅指本检查点及其产物 push 到自己的 agent 分支且 `git ls-remote` 远端 SHA 与本地 HEAD 一致，**不证明任何科学结论**。
- ANN 是上界参考行，不是转换对照；shared-theta 行是附加对照，不是独立方法胜利；不缩放数据集、不挑 seed、未改阈值或 scale。
- 本轮无失败 GPU 作业（三个 `rc=0`）；上一轮的三次 CLI 失败证据保留，连败计数按「同策略连败」规则不累计为新策略失败。

## 8 下个入口

1. **E3/E5（S3 前置）**：完整已知操作计数（setup/first analog/base/projection/source/decoder/1-t/graded banks/gates/sigmoid/products/normalization/scatter/residual/BN/head/pooling/storage/traffic，AC 与保守 MAC 双约定、ADD/MUL/MAC pJ 算术代理）、同 GPU/batch128/顺序/warmup 的 3 次重复中位墙钟（ANN/DC/centered/最强对照），真实数据 gate-bound 与 message/layer/logit 误差、raw-vs-decoded 与 fresh-vs-stale gate falsifier。
2. 之后 E2/E4/E6 忠实对照（NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN）与 MNIST Gated / DGCNN ModelNet40 跨任务。
3. 然后验证集-only Story 判定、冻结后一次新测试确认、六节全新终稿、≥36 篇经核查引用且近 3–5 年占多数、去重审计、当前完整稿实质独立 CCFA≥7/10。
4. 停止条件不变：冻结输入哈希漂移、canonical 索引/标签漂移、协议/源码/时钟不一致、超出 contract v4 计算上限、用户叫停。失败证据一律保留。

## 9 恢复命令

环境与解冻输入由 `GNN2SNN_DATA_ROOT`、`DCGS_REPO` 两个环境变量指向本机官方 PyG 数据根与钉死 DCGS checkout；解释器为 Conda `dyg-cold` 的 `python.exe`。在仓库根执行：

```
python -X utf8 research/visual-graphs/continuation_20261113/code/e1_residual_v1.py precheck
python -X utf8 research/visual-graphs/continuation_20261113/code/e1_residual_v1.py run --seed <41|42|43> --output-dir <results>/e1-residual-seed<seed>-v1
python -X utf8 research/visual-graphs/continuation_20261113/code/e1_residual_v1.py verify
python -X utf8 research/visual-graphs/continuation_20261113/code/aggregate_e1_fullrows_v1.py
```

`run` 会拒绝已存在的输出目录；重跑须另建新目录并保留旧产物，不得覆盖已发布证据。
