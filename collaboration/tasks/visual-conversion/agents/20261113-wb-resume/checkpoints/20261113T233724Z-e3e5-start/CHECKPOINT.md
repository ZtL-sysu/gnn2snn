# R6 启动：按 GitHub 进度补 E3/E5，3060 恢复满载

## 进度依据（严格按已上传内容）

- 用户 2026-11-14 07:32 +0800 要求：要求不变；**按传到 GitHub 的进度指导下一步**；把 3060 拉满；把心跳间隔调小。
- `git fetch --all --prune` 后远端分支 `agent/visual-conversion/20261113-wb-resume` HEAD = `986510111b0c98d7ddb6d00ed8e73433ddded94f`（04:31 +0800 写回提交）。权威进度 = 检查点 `20261113T202500Z-e1-fullrow-residual/CHECKPOINT.md` 与 HANDOFF 的 R5 段：E1 全 10 行族 × 4 T × 3 seed = 120 份逐样本证据已完成，`criterion.verdict = NOT EVALUABLE`，**下个入口是 E3/E5 完整成本 + 匹配墙钟**。
- 04:31 之后 3 小时无新提交；`nvidia-smi` 0% / 955 MiB、无 Python 进程 → 卡空转。本地有一个**未上传**的 799 行草稿 `e3_e5_v1.py`，其 `precheck` 因硬编码「期望 120 行」失败（转换行实为 9 模式 × 3 seed × 4 T = 108，ANN 成本由 `build_ann_cost()` 单独构造）。草稿未上传，按用户要求不作为进度。

## 本轮真实动作

1. 修 `e3_e5_v1.py` 三处行数硬编码/说明（120 → `len(expected)` = 108）；`py_compile` PASS。
2. `precheck` **PASS**：三个 seed 冻结绑定、作者源码 provenance、官方验证集形状普查（5000 图 / 587924 节点 / 4703392 边 / E:N=8.0 / 40 批）、108 行 × 站点规格映射、`graph_layer_evaluations = L*T*P`。
3. `cost`（CPU）完成：`results/e3-e5-v1/e3_cost_ledger_v1.json`（108 行、218 条固定列指标、AC 与保守 MAC 双约定、pJ 算术代理、ANN 上界 132285413.23 pJ/图、candidate T4 AC 68832223.93 pJ/图、代表性 batch 手工核算、unresolved items 显式列出）。成本账明确标注为声明式稠密原型操作计数，**不是硬件能耗**，且 unresolved 项未删（协议：含未决项的成本不能单独判胜）。
4. **E5 GPU 长作业已启动**：`latency --device cuda --batches 40 --repeats 3 --warmup 1`，覆盖全部 5000 官方验证图、37 个配置（ANN + 9 行族 × 4 T）、同 GPU/batch128/固定顺序/1 次 warmup/3 次重复中位数。后台句柄 `ThE3rS`，本机 Python PID 27348，启动 2026-11-13T23:37:13Z；23:37:24Z 实测 GPU **84% / 3824 MiB**，日志 `logs/e3e5_latency_v1.log`。
5. 心跳已改为 **每 15 分钟**（`FREQ=HOURLY;INTERVAL=1;BYMINUTE=0,15,30,45`；平台拒绝 `BYHOUR` 列表，故用 HOURLY + BYMINUTE 多值）。新 prompt 把「以 GitHub 最新已推送检查点为唯一进度依据」「GPU 空闲即必须启动真实作业」「维护 GPU_QUEUE.md 串行长队列」写成硬要求，并写明本地草稿的阻塞点。

## 明确不能下的结论

- E1 仍 **NOT PASSED / NOT EVALUABLE**：`aggregate` 尚未运行，双门槛的成本/延迟腿未重算；`e5_latency_v1.json` 本检查点时还不存在。P4 仍 `running`，S2/S3 仍 FAIL，S4–S8 未动。
- 本检查点不含延迟结果，只有成本账与作业登记；清单只证明文件身份，不证明科学结论。
- 未改任何旧协议、旧证据或哈希；owner 的 TASK.md 未动；无 merge、无 force push、无远程 GPU。

## 队列与下一步

见同目录 `GPU_QUEUE.md`：#1 延迟作业（运行中）→ #2 `aggregate` 重算双门槛 → #3 结果检查点上传 → #4 E2/E4/E6 忠实对照 → #5 跨任务 → #6 E5 机制 falsifier。心跳每次唤醒消费队首并回填状态。

## 恢复命令（Conda `dyg-cold`，仓库根）

```
python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py precheck
python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py latency --device cuda --batches 40 --repeats 3 --warmup 1
python -X utf8 research/visual-graphs/continuation_20261113/code/e3_e5_v1.py aggregate
```

`latency` 拒绝覆盖已存在的 `e5_latency_v1.json`；重跑前须确认无存活作业，且旧日志保留为证据。停止条件不变：冻结输入/哈希漂移、canonical 漂移、协议源码时钟不一致、超出 contract v4 计算上限、用户叫停。
