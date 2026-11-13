# R4b：三 seed E1 逐样本证据完整清单与同步前门禁

- 时间：2026-11-13T18:20Z 唤醒（+0800 2026-11-14 02:20）
- 分支：`agent/visual-conversion/20261113-wb-resume`；agent：`20261113-wb-resume`；task：`visual-conversion`
- 本检查点是 R4（seed42/43 冻结校准 + 六行 E1 全网格）证据的**权威发布清单**，取代两个仅本地的草稿检查点（见「清单事件」）。
- 流水线：contract v4，P0–P3A `passed`，**P4 仍 `running`**（未加 pass），P5–P21 `pending`。状态 `in_progress`。

## 1 唤醒动作 1：分支与活跃作业（真实查询，不采信旧日志）

- `git fetch --all --prune` 成功；本地 HEAD 与 `git ls-remote origin refs/heads/agent/visual-conversion/20261113-wb-resume` 均为 `1eafeb78325f4fe2a226ce8769e0dc53f36b5dc3`；工作区有 R4 未提交产物。
- `nvidia-smi`：GPU 0 `0 % / 955 MiB / 12288 MiB`；`Get-CimInstance Win32_Process -Filter "Name like 'python%'"` 返回 0 个进程 → **无存活作业**，本轮未启动任何 GPU 作业。
- 首次 `tasklist /FI` 在 Git Bash 下被路径改写为 `C:/Program Files/.../FI` 而报错，已改用 PowerShell 的 CIM 查询核实，不影响结论。

## 2 唤醒动作 2：冻结输入硬门禁

`reverify_frozen_inputs_v1.py`（新增，只读、无 GPU）输出 `status: PASS`，`frozen_items_verified: 12`：

| 项 | 结果 |
|---|---|
| PROTOCOL-v3.md `8573171a…`、分类头补充 `8bde5832…` | PASS |
| affine_cal1.json `1618e3b4…`、gcal_v1.json `b28485dd…` | PASS |
| seed41 教师 best.pt `d26df76d…` | PASS |
| 官方 PyG CIFAR10 processed ×5 | 5/5 PASS |
| CURRENT_SCOPE.md `e4439c00…` | PASS |
| 钉死 DCGS checkout HEAD `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be` | PASS |
| pipeline-state.json：contract v4、P0–P3A passed、P4 running、P4 之后无 passed | PASS |

报告：`logs/frozen-input-gate-r4b-v1.json`（只含相对名与哈希，无本机绝对路径）。冻结协议、旧证据、旧哈希、owner 的 `TASK.md` 均未修改。

## 3 清单事件（如实记录，未篡改已发布证据）

- R4 的 `checkpoints/20261113T180412Z-multiseed-e1/manifest.json` 为**未发布的本地草稿**，含 23 个产物，先前 `verify` 通过。
- 本轮为去除本机身份路径，修改了 4 个新增的 R4 代码文件（见第 4 节）。重验旧草稿清单返回 `artifact changed: .../test_flow_equivalence_v1.py`；按 `checkpoint_manifest.py` 的不可变规则**不覆盖它**，改为另建本检查点。
- 本检查点 `manifest.json` 在同一次唤醒、push 之前，随新增脚本最终定稿而重建过（每次都是删除尚未发布的草稿后重新 `create`，并以 `verify` 收尾）。**没有任何已发布产物、协议或历史证据被改动**；最终清单 `create` + `verify` 均为 `verified / 74 artifacts`。最终上传内容检查 `upload-content-check-r4b-v1.json` 为 `PASS / 80 files / 48 NPZ / 0 problems`。
- 两个草稿检查点目录（`20261113T180412Z-multiseed-e1/` 与本次 18:20Z 的早期草稿）均只留本地、不 stage、不上传；其事实内容已并入本文。

## 4 去标识与等价性复核

- `test_flow_equivalence_v1.py`、`precheck_e1_multiseed_v1.py`：把硬编码的本机用户目录 DCGS 路径改为 `Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"`（运行时实测仍解析到同一钉死 checkout，git HEAD 一致）。
- `deidentify_v2.py`、`preflight_upload_v1.py`：本机用户名改为运行时从环境 / home 目录名推导，源码内不再含任何本机身份字符串。
- 修改后重跑：`test_flow_equivalence_v1.py` 六行阈值映射对 seed41 参考仍 **逐位相等 / PASS**；`precheck_e1_multiseed_v1.py` 42/43 阈值形状与校准哈希 **PASS**；10 份 `*.public.json` 复检 `identity_or_path_violations: [] / PASS`（`remaining_windows_drive_paths=0`、无本机用户名）。
- 原始 `report.json`、4 份原始校准 JSON、2 份 `calibration_summary.json`、原始 `e1_multiseed_summary_v1.json` 仍只留本地；公开副本只替换绝对路径键/字符串，数值与哈希逐字不变。

## 5 R4 事实（本清单覆盖的证据）

- seed42/43 冻结全训练无标签校准，复用 frozen `calibrate_loader`（45000 图 ×2 遍、512 bins、q=0.999、batch128、scale1）：affine `07bc8629…` / `339dd792…`，六站点 `9da8ad79…` / `a3405ed1…`。
- E1 六行 × T1/2/4/8 × 全 5000 官方验证图：seed42（697.4 s，教师 0.703）、seed43（11m49s，教师 0.7074），各 24 个逐样本 NPZ + 报告；四份作业日志均 `EXIT=0`。
- `verify_e1_multiseed_v1.py 42 43` 均 PASS：NPZ 字段/形状、canonical 索引标签逐位、教师双前向确定性、报告精度可从 NPZ 重算、NPZ 哈希一致、`generic/cse_only/per_consumer` 别名关系。
- 跨 seed 描述性汇总（336 行固定列 `METRICS.csv`）：centered T4 `0.7053`、T8 `0.7057`；最强匹配对照 T4 `dc_adapt_analog_head 0.7056`、T8 `first_analog 0.7060`；差距均 `0.0333pp`。
- **E1 未通过**：`criterion.verdict = NOT EVALUABLE / E1 NOT PASSED`，精度条件单独满足，但完整已知操作成本与匹配墙钟证据缺失；`missing_required_rows = [ANN, historical_six_site_DC_adapt, shared_theta_controls, complete_known_operation_cost, matched_wall_clock_latency]`。

## 6 本检查点清单范围（74 个产物，SHA-256 已 verify）

11 个代码文件、4 份 GPU 作业日志、1 份冻结门禁报告、6 份校准公开副本、2 份 `report.public.json`、1 份汇总公开副本、1 份 `METRICS.csv`、**48 份逐样本 NPZ**（每个约 380 KiB，合计约 17.7 MiB，单文件远低于 50 MiB）。`metrics.csv` 为固定列指标表副本。上传前内容检查报告：`logs/upload-content-check-r4b-v1.json`。

## 7 明确不能下的结论

- **同步成功**：仅当 push 完成后用 `git ls-remote` 核对远端 SHA 等于本地 HEAD 才成立；本文件写就时尚未 push。
- **科学验收通过**：不成立。P4 仍 `running`；S2 仍 FAIL（缺 ANN 行、历史六站点 DC-adapt 汇总、shared-theta 附加对照）；S3 仍 FAIL（双门槛 NOT EVALUABLE）；S4/S5/S7/S8 仍 FAIL；S6 待验证。无 Story 判定、无新测试集推理、无论文、无引用核查、无独立 CCFA 评审。
- 本轮无 GPU 作业、无失败作业；连败计数仍为 0。

## 8 下个入口

1. shared-theta 附加对照（first_analog/centered 的 channelwise max 同向量，冻结后验证）+ 历史六站点 DC-adapt 行汇总 + ANN 行。
2. E3/E5 完整操作计数（AC/保守 MAC 双约定）、3 次重复中位墙钟（同 GPU/batch128/顺序/warmup）、真实数据 gate-bound 与 message/layer/logit 误差 falsifier；之后才能判 S3。
3. 然后 E2/E4/E6 忠实对照与 MNIST Gated / DGCNN ModelNet40 跨任务。
4. 停止条件不变：冻结输入哈希漂移、canonical 索引/标签漂移、协议/源码/时钟不一致、超出 contract v4 计算上限、用户叫停。失败证据一律保留。
