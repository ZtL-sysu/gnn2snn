# R2：补齐 E1 对照评估器，启动首个官方控制验证

## 本轮事实

- 接手 `20261113-luna-controls` 被暂停中断的实现任务：该分支只留下任务分配，没有提交任何评估器代码。本 agent 在独立工作区与分支内，只写分配中登记的两个路径 `code/eval_controls_validation_v2.py`、`code/test_eval_controls_validation_v2.py` 以及自己的 agent 目录；owner 的既有代码/结果/协议与 TASK.md 均未改动。
- 新评估器绑定双冻结校准：`affine_cal1.json`（SHA256 1618e3b4…，18 个绑定输入逐一重算通过）与 `gcal_v1.json`（六站点，SHA256 b28485dd…，与 E0 参考一致）；教师 `best.pt` 哈希 d26df76d… 与 E0/E1 报告一致。预检 PRECHECK OK。
- 阈值映射冻结、不搜索：`dc_adapt_analog_head` 用六站点校准的 `gated_0..gated_3` 四个图分支向量（两个 MLP 站点有意不编码，即分类头匹配补充要求的模拟头对照）；`per_consumer`/`cse_only` 用 `affine_cal1.json` 的 `raw_node_i/raw_edge_i` 八向量（`profile_thresholds(generic, 1)`）。`cse_only` 明确登记为 generic 数值别名，不当独立方法行。
- 合成测试 `test_eval_controls_validation_v2.py`：4 passed（阈值绑定/形状、错误宽度拒绝、协议常量、alias 与 generic 逐位一致）。
- 2026-11-13T15:08:08Z（本机 23:08:08 +0800）在 RTX 3060 + Conda dyg-cold 启动第一个官方全 5000 图控制验证：3 行 × T1/2/4/8 = 12 组合，seed41，scale1，无 mask、无测试集。

## 长作业登记（真实句柄，不等同于完成）

- Provider：本机 RTX 3060；工具后台任务句柄 `tzmLS9`；本机 Python 进程 PID 33240（进程列表与启动时间已核实）。
- 最近核实：2026-11-13T15:08:34Z。首批 128 图 `elapsed_sec=9.071`，峰值显存 2,978,871,808 字节；该数字只是资源可行性探测，不是延迟基准。
- 输出目录（本 agent run root 内，运行前确认不存在）：`results/e1-seed41-controls-v1/`；日志 `logs/e1_seed41_controls_v1.log`。
- 恢复命令：重跑 `conda run -n dyg-cold python -X utf8 code/eval_controls_validation_v2.py …`（参数见 HANDOFF），先确认输出目录不存在、无同名存活进程，不要盲目重启。

## 检查与局限

- 评估器保留全部严格门：教师/校准/数据/源码/作者源前后哈希、规范 5000 图索引与教师逐位参考、0.7066 冻结教师精度、确定性设置、输出目录写保护与 E1 计算上限投影。
- 本检查点时**尚无新科学结果**；无 METRICS.csv（无已验收实验指标，不造空表）。清单只证明文件身份。
- 新评估器只产出稠密参考测量与部分观测计数器，明确不含完整已知操作成本、匹配墙钟或能耗；E1 的 .3pp + 成本双门槛仍不可判定。
- Seed 42/43 需各自的冻结全训练无标签校准后再按不变流程验证；共享 theta 附加对照、历史六站点 DC-adapt 行仍待 E1 汇总。

## 下一步

等待作业结束并核验逐样本 NPZ/报告哈希；把成功结果（或真实失败摘要）与代码一并发布为下一检查点，然后推进 seeds 42/43 与完整成本账。
