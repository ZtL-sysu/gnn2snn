# R7b-sync 补充清单（主会话补传，非科学进展）

- 时间：2026-11-14T05:54Z（+0800 13:54）；触发：用户要求「先把这些东西传github上」。
- 背景：R7b/R7b2 的 E5 完整墙钟、E1 有证据重算、v1 失败归档与 v2 修复脚本在 11:56–12:12 +0800 于本地完成，但**未推送**；远端权威进度仍停在 `6b1fb4a`（R7）。本次由主会话按 research-github-sync 明确 stage 并 push，不改变任何科学结论。

## 与既有清单的关系

- **主清单**：`checkpoints/20261114T041000Z-r7b2-e5-latency-criterion-public/manifest.json` 绑定 12 个不可变产物（`e5_latency_v1.json`、`latency_METRICS.csv`、`e3_e5_criterion_v1.json`、三份 v1 失败 `*.public.log`、`PUBLIC_LOG_MAP.json`、`FAILURE.json`、`e5_falsifier_v2.py`、`test_e5_falsifier_v2.py`、`gpu_queue_driver_v2.py`、`gpu_queue_launcher_v2.py`），`create`+`verify` PASS，`input_git_head` 恰为本次提交前的 HEAD `6b1fb4a`。该清单**原样发布、未改写**。
- **本补充清单**：只绑定随本次同步新增、且不在主清单里的不可变文件——v2 失败归档的去标识公开副本与元数据、五个 v3 修复/驱动/测试脚本、`gpu_queue_v3.json`。不绑定任何活文件（HANDOFF / LOG / GOAL_LEDGER / GPU_QUEUE）、不绑定仍可能被追加的驱动日志、`status.jsonl` 或 `partial.jsonl`。
- 已发布的 R6/R6b/R6c/R6d/R7 检查点与清单全部原样保留；本地未发布草稿 `20261114T040000Z-r7b-e5-latency-criterion/`（清单绑定含本机路径的原始 traceback）与失效草稿 `20261113T180412Z-multiseed-e1/` **不上传**。

## 明确排除（内容与隐私检查）

- 原始 traceback `*.log`（本机 Python 安装路径/用户名）、`.workbuddy/`、`clean-restore-v1/` 等 139 MiB 本地材料、教师权重包、数据缓存、重复大失败产物。
- 上传前对全部 staged 文本做凭证/本机身份路径扫描；单文件均远低于 50 MiB；NPZ 类逐样本证据已在 R7b2 之前的检查点发布。

## 真实状态（如实登记，不粉饰）

- E5 机制/falsifier：v1 失败（张量掩码广播）、**v2 也失败**（`dump_acc()` 第 471 行 `AttributeError: 'Acc' object has no attribute 'items'`，`run`/`verify` 均 `rc=1`），**没有任何 gate-bound / message-layer-logit 误差 / falsifier 数值**。`e5_gate_bound_v1.json` 不存在。
- v3 修复脚本（`e5_falsifier_v3.py`、`test_e5_falsifier_v3.py`、`smoke_e5_falsifier_v3.py`、`gpu_queue_driver_v3.py`、`gpu_queue_launcher_v3.py`）compile PASS、precheck PASS；`gpu_queue_v3.json` 两步队列（v3 run 预计 540 s → verify）已备，**GPU 作业在本检查点时未启动**，无 python 计算进程、GPU 空闲。两个旧计划任务为 Ready。
- 科学结论不变：E1 `NOT PASSED (method advantage not established)`（精度腿满足、成本腿与墙钟腿均不满足、9 项 unresolved 保留），`story_decision: DEFERRED`，无新测试推理；P4 `running`，P5–P21 `pending`；S2/S3/S4/S5/S7/S8 FAIL、S6 待验证、S9 进行中。无六节新稿、无 ≥36 引用核查、无独立 CCFA 评审。

## 同步口径

push 后以 `git ls-remote` 核对远端 SHA 一致，只能报 **同步成功**；**科学验收通过仍不成立**。下一轮心跳以本分支最新已推送检查点为唯一进度依据，先核实 v3 前置条件，再启动**唯一一个** v3 GPU 作业，避免并发污染。
