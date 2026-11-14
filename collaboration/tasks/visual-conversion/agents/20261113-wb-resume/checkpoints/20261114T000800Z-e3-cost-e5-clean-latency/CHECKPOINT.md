# R6c：E3 完整成本账复核、E5 双作业并发事故处置、干净墙钟作业启动

- 时间：2026-11-13T23:45Z–2026-11-14T00:08Z（+0800 2026-11-14 07:45–08:08）
- 分支：`agent/visual-conversion/20261113-wb-resume`；发布时本地 HEAD == 远端（见第 8 节 `git ls-remote` 结果）
- 流水线：contract v4，P0–P3A `passed`，**P4 仍 `running`**（未加 pass），P5–P21 `pending`
- 本检查点是 R6c 新产物与不可变证据的**权威清单**；不覆盖、不改写任何已发布协议、旧证据或旧哈希

## 1 唤醒门禁（真实查询）

- `git fetch --all --prune` 成功。本轮开始时 HEAD/远端均为 `986510111b0c98d7ddb6d00ed8e73433ddded94f`；期间同一工作树的心跳会话发布了 R6/R6b 两个提交，随后 `git fetch` 后本地与远端同为 `62eddfa3dcb89bca5f06d2838bba6d06a5630cc3`，`git rev-list --left-right --count` = `0 0`，无分叉。
- `nvidia-smi` 起始 `0 % / 955 MiB / 12288 MiB`；`reverify_frozen_inputs_v1.py` → `status: PASS`、`frozen_items_verified: 12`、`contract_version: v4`、`P4_status: running`、`errors: []`；`pipeline_state.py <state> --before P4` → PASS。
- **查活口径修正（如实记录）**：唤醒时使用的窄过滤 `python.exe|pythonw.exe|conda.exe` 返回 0，但后来用 `CommandLine -match 'e3_e5_v1\.py'` 宽匹配发现 23:37Z 启动的作业其实一直存活。窄查询曾漏报；本检查点之后一律要求宽匹配查活，见 `GPU_QUEUE.md` 顶部。

## 2 静态复核拦下的缺陷（GPU 作业前修复，失败证据保留）

对 `e3_e5_v1.py` 逐项核对冻结报告真实键名、站点命名、计数器整除性、helper 签名后，修复以下必现或方法学问题：

1. `latency` 在 `R.load_modules()` 把 `takeover_20261112` 加入 `sys.path` **之前**导入 `matched_controls` → `ModuleNotFoundError: gated_causal_hybrid`（dry-run 实测报错）。已调整导入顺序。
2. 冻结阈值张量原本在**计时循环内**每个 batch 重建，把校准查表开销按模式不同地计入墙钟。改为在计时前一次性构造 `TH`，并在作业末尾逐元素校验 `TH` 未被就地修改。
3. 代表性批次检查原为恒真式（同一公式写两遍）。改为从冻结 seed41 `best.pt` 的 `layers.{i}.{A,B,C,D,E}.weight` 与 `MLP_layer.FC_layers.{j}.weight` 真实形状独立推导投影 MAC（70×70 ×5、head 35/17/10），并与 batch 0 真实 N/E 核对；`HEAD_MACS=3215` 亦由权重重算。
4. `accounting_model.formulas.storage` 声明式与实现不一致，已改为按 n_sites 声明。
5. `cost()` 拒绝覆盖既有产物；上一轮由**修正前脚本**生成的完整成本账（108 行、数值有效）未丢弃，连同 SHA-256 一并保存在 `logs/e3e5-cost-draft-superseded-v1/`（草稿大文件仅本地保留，公开只上传摘要与哈希记录）。重跑后 `cost_METRICS.csv` 哈希与草稿**完全一致**，证明复核只改校验与声明、未改动任何数值。
6. 新增 `--dry-run` / `--limit` / `--resume` 与逐配置增量 `e5_latency_v1.partial.jsonl`，使 2 小时级作业可跨心跳续跑而不丢证据。

## 3 E3 完整已知操作成本账（CPU，已完成）

`results/e3-e5-v1/e3_cost_ledger_v1.json` + `cost_METRICS.csv`：

- **108 个冻结转换行**（9 行族 × 3 seeds × 4 T）+ ANN 上界参考（由 `build_ann_cost()` 单独构造）= 120 个 E1 单元；218 条固定列指标。
- AC 与 conservative MAC 双约定；`ADD 0.9 / MUL 3.7 / MAC 4.6 pJ` 算术代理；逐组计数 setup/first-analog-or-base/projection/source/decoder/1-t/graded banks/gates/sigmoid/products/normalization/scatter/residual/BN/head/pooling/storage/traffic，末层未用边输出对所有行同规则剪枝。
- ANN 上界 `132,285,413.2264 pJ/图`；候选 `centered` T4 AC `68,832,223.9346 pJ/图`。
- **9 项 unresolved items 显式列出且不得省略**：硬件能耗/延迟、真实 DRAM 流量、事件 bank 构造与预缩放、cache 等价性维护、超越函数硬件成本、精度/累加器宽度、BN fold 选择、fused first-analog 缓存与事件驱动后端**未实现**（稠密原型每 tick 重算四层）、host 侧开销。按 PROTOCOL-v3「Cost with unresolved omitted work cannot pass」，该成本账**单独不能给 pass**。
- 明确声明：稠密 PyTorch 原型的算术计数，**不是芯片能耗、不是硬件延迟、不是加速比**，不得用事件数冒充节能。

## 4 E5 匹配墙钟：并发事故与干净重跑

- 23:37Z 心跳启动 `ThE3rS`(PID 27348，`--batches 40 --repeats 3 --warmup 1`，未复核脚本)；23:51Z 本会话启动 `jvgzz3`(PID 15052，带 `--resume`)。两者**同时占用同一块 3060**，违反「单 GPU 单作业」，并使匹配墙钟因 GPU 争用失真；旧作业还带有「阈值在计时循环内重建」的方法学缺陷。
- 处置：两个作业链全部终止 → 宽匹配确认 0 个相关进程、GPU `0 % / 955 MiB` → 归档两份日志与 3 条受污染增量记录到 `logs/e3e5-latency-concurrency-incident-v1/`（含 `SHA256SUMS.txt` 与 `INCIDENT.json`）→ **其全部计时数值作废，不作为任何证据**。
- 00:04:49Z 启动**唯一**干净作业：后台句柄 `TsI47h`，命令 `python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume`；日志 `logs/e5_latency_v1.clean.driver.log`；逐配置增量 `results/e3-e5-v1/e5_latency_v1.partial.jsonl`。启动后实测 GPU `89 % / 5130 MiB`，宽匹配为单一进程链。参数与协议要求一致（同机 3060、batch128、固定 5000 图顺序、1 warmup、3 重复取中位、37 个配置覆盖 ANN + 9 行族 × 4 T），**未缩短批次、未降 repeats、未挑 T**。
- `GPU_QUEUE.md` 顶部已写入醒目协调说明，要求心跳宽匹配查活、在跑时不得重启或另起第二个 GPU 作业。

## 5 E5 真实数据 gate-bound 与 falsifier（代码就绪，GPU 运行待排队）

新增 `research/visual-graphs/continuation_20261113/code/e5_falsifier_v1.py`，`py_compile` / 模块导入 / `precheck` 均 PASS（6 冻结文件、5/5 processed 哈希、6 个 shared-theta 向量、DCGS `5e0e4c73…`）：

- 不另写近似转换器：复用冻结 `gated_affine_base_v1`，在 centered 前向内联计算 `|m(z',b')-m(z,b)| ≤ max|b'-b| + R(b')tanh(η/2)`，逐 T、逐层、degree（入度中位切分）与 sharpness（教师 top1−top2 中位切分）子集报告 bound 与实际 message / pre-BN layer / logit 误差。
- 两个反事实开关：`decoded_skip`（用解码前缀替代 raw skip）与 `stale_gates`（复用 tick-1 gates）；**开关全关时必须与冻结 `affine_mode_forward` 输出一致（≤1e-4）才接受任何 falsifier 结果**。
- 因单 GPU 单作业，本轮**尚未**跑 GPU；`run` 排在干净墙钟作业之后。当前不构成 E5 机制证据。

## 6 与已发布 R6/R6b 清单的关系（不掩盖哈希变化）

- R6(`fc288b5`) / R6b(`62eddfa3`) 的 `manifest.json` 绑定了**修正前**的 `e3_e5_v1.py` 与 `e3_cost_ledger_v1.json`。本轮复核改动了这两个工作树文件，因此在**当前 HEAD** 上那两份清单会报 `artifact changed`；这是脚本修复与证据质量改进，不是改协议、改阈值或把失败改成通过。
- 旧字节完整保存在 `fc288b5` / `62eddfa3` 的 git 历史中，旧清单在各自提交处仍可 `verify`；本检查点绑定当前**修正后**版本。旧清单文件本身不删除、不重算、不覆盖。
- `cost_METRICS.csv` 与 `e3e5-precheck-v1.json` 字节未变（数值与预检结果一致）。
- 活文件 `HANDOFF.md` / `LOG.md` / `GOAL_LEDGER.md` / `GPU_QUEUE.md` 与两个 live 日志/增量文件**不进入不可变清单**，沿用 R4b/R5/R6b 惯例。

## 7 科学状态（无验收通过）

- **E1 仍 `NOT EVALUABLE / E1 NOT PASSED`**：T4 0.0667pp、T8 0.0467pp 的精度条件单独满足，但完整已知成本与匹配墙钟尚未合并重算（`aggregate` 未跑）；即便合并，unresolved items 仍使成本腿不能给 pass。
- S2 FAIL、S3 FAIL、S4 FAIL、**S5 仍 FAIL**（E3 成本账已完成、E5 墙钟在跑、gate-bound/falsifier 未实测）、S6 待验证、S7 FAIL、S8 FAIL、S9 进行中；P4 `running`。
- 无 Story 判定、无新测试集推理、无六节新稿、无 ≥36 篇引用核查、无独立 CCFA 评审。
- **「同步成功」仅指本检查点 push 到自己的 agent 分支且 `git ls-remote` 远端 SHA == 本地 HEAD，绝不表示科学验收通过。**

## 8 下个入口与恢复命令

1. 先按 `GPU_QUEUE.md` 顶部宽匹配查活：`Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'e3_e5_v1\.py' }`。`TsI47h` 在跑则只监督，不重启。
2. 若进程已消失且 `e5_latency_v1.json` 不存在：用完全相同命令（含 `--resume`）续跑，增量行自动跳过。
3. 作业结束后（CPU）：`python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py aggregate` → `results/e3-e5-v1/e3_e5_criterion_v1.json`，如实重算双门槛。
4. 然后跑 E5 机制证据（GPU，单作业）：`python -X utf8 research/visual-graphs/continuation_20261113/code/e5_falsifier_v1.py run --device cuda --batches 40` → `verify`；再发布结果检查点。
5. 之后 E2/E4/E6 忠实对照 → 验证集-only Story 判定 → 一次新测试确认 → 六节新稿 → ≥36 经核查引用 → 当前稿实质独立 CCFA≥7/10。
6. 停止条件不变：冻结输入哈希漂移、canonical 索引/标签漂移、协议/源码/时钟不一致、超出 contract v4 计算上限、用户叫停。失败证据一律保留。
