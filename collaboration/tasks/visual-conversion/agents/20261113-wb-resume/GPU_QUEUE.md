# GPU 作业队列 — 20261113-wb-resume

维护规则：以 GitHub 已推送的最新检查点为唯一进度依据；每次心跳唤醒先查 `nvidia-smi` 与精确 Python 进程，GPU 空闲就消费队首并启动真实作业，**不得只做文档写回**。作业在跑则监督/收集结果并预排后续，不重复启动。所有作业只用本机 RTX 3060 + Conda `dyg-cold`。

| # | 作业 | 命令 / 产物 | 预计时长 | 依赖 | 状态 |
|---|------|-------------|---------|------|------|
| 1 | E5 匹配墙钟（3 次重复中位数） | `python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py latency --device cuda --batches 40 --repeats 3 --warmup 1` → `results/e3-e5-v1/e5_latency_v1.json` + `latency_METRICS.csv`；日志 `logs/e3e5_latency_v1.log` | 约 60–90 分钟（37 配置 × 40 批 × 4 遍，首配置后可按实测修正） | precheck PASS、E3 cost 已完成 | **运行中**：后台句柄 `ThE3rS`，PID 27348，2026-11-13T23:37:13Z 启动，23:37:24Z 核实 GPU 84% / 3824 MiB |
| 2 | E1 双门槛重算 | `... e3_e5_v1.py aggregate` → `results/e3-e5-v1/e3_e5_criterion_v1.json`；读 `e1-fullrow-summary-v1` 的精度差 | <1 分钟（CPU） | #1 的 `e5_latency_v1.json` | 待 #1；注意 unresolved items 仍会使 `Cost with unresolved omitted work cannot pass` 生效 |
| 3 | R6 结果检查点上传 | CHECKPOINT + METRICS + 清单 + HANDOFF/LOG/账本写回，push 后 `git ls-remote` 核对 | 数分钟 | #2 | 待 #2 |
| 4 | E2/E4/E6 忠实对照准备 | 按 R5 检查点：NaiveIF、官方 DCGS、QCFS、FTBC、SNN Calibration、One-Spike、直接训练 SpikingGCN；先做可复现入口与预检，再排训练 | 长（训练为主） | #3 完成、冻结输入门禁 PASS | 排队中 |
| 5 | MNIST Gated / DGCNN ModelNet40 跨任务 | 冻结 split 与 own-trajectory dynamic kNN，弱结果照实跑 | 长 | #4 | 排队中 |
| 6 | E5 机制与 falsifier 补强 | gate-bound vs message/layer/logit 误差、raw-vs-decoded、fresh-vs-stale gates，全 T 与 degree/sharpness 子集 | 中 | #1/#2 的实测数据 | 排队中 |

## 恢复与失败处理

- #1 若被中断：先查 `nvidia-smi`/进程确认无存活作业；`e5_latency_v1.json` 不存在时可用同一命令重跑（`require(not lp.exists())` 只拒绝覆盖已发布报告），保留中断前的日志为失败证据，不要改脚本或阈值。
- 严禁：缩短 `--batches` 来赶进度、降 `repeats`、挑 T、改 pJ 常数、用事件数冒充能耗/加速、把成本账的 unresolved items 静默删掉。
- 心跳现在 `FREQ=HOURLY;INTERVAL=1;BYMINUTE=0,15,30,45`（每 15 分钟）。每次唤醒必须报告 GPU 占用事实与队列消费情况。
