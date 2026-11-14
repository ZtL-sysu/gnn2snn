# R6b：清单拆分修正（非科学进展）

## 问题

R6 启动检查点 `20261113T233724Z-e3e5-start/manifest.json` 把 4 个**活文件**（`GPU_QUEUE.md`、`HANDOFF.md`、`LOG.md`、`GOAL_LEDGER.md`）也纳入了不可变 SHA-256 清单。随后 `GPU_QUEUE.md` 因补充「会话边界风险 / 相同参数重跑 / aggregate 后立即接下一个 GPU 作业 / 实测节奏」而演进，`verify` 正确报出 `artifact changed: .../GPU_QUEUE.md`。

仓库既有做法一致：R4b（74 产物）与 R5（72 产物）的权威清单**均不绑定活文件**（live-bound = 0），活文件按轮次更新、由最新检查点描述。R6 把活文件塞进清单属于本轮失误。

## 修正（不改写任何已发布检查点）

- 旧 R6 检查点目录及其 `manifest.json` **原样保留**，不删除、不覆盖、不重算；其失配只针对活文件，5 个不可变产物（`e3_e5_v1.py`、`e3e5-precheck-v1.json`、`e3_cost_ledger_v1.json`、`cost_METRICS.csv`、R6 `CHECKPOINT.md`）字节未变。
- 新建本 R6b 清单，只绑定这 5 个不可变产物 + 本 `CHECKPOINT.md`，`create` + `verify` 均 PASS。活文件随同一次提交上传，但**不进清单**，与 R4b/R5 惯例一致。
- 后续轮次继续只把不可变证据纳入清单；HANDOFF/LOG/账本/队列作为活文件演进，由每轮最新 CHECKPOINT.md 描述。

## 科学状态（无变化）

- E5 延迟作业 `ThE3rS` 仍在运行（config 9/37、累计 1140.2 s、GPU 持续高占用），`e5_latency_v1.json` 尚未生成。
- E1 仍 **NOT PASSED / NOT EVALUABLE**；`aggregate` 未跑；P4 `running`；S2/S3 FAIL，S4–S8 未动。
- 本检查点不含任何实验结果，仅为证据完整性修正；不构成科学验收，也不改变任何阈值、协议或旧哈希。

## 队列要点（已写入活文件 GPU_QUEUE.md）

- 若主会话结束导致后台作业被终止且 `e5_latency_v1.json` 不存在，下一次心跳（≤15 分钟）须用**完全相同**的 `--batches 40 --repeats 3 --warmup 1` 重跑，保留中断日志，不得缩短批次或降重复次数。
- #1 完成 → 立即 `aggregate`（CPU）→ 发布结果检查点 → **同一次唤醒内启动 #4 的 GPU 作业**，不允许整理完就空卡等下一次。
