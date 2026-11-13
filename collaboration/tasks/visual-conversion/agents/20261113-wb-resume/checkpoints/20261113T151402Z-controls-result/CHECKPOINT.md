# R3：seed41 新增 E1 对照行完整验证结果

## 真实执行

- 作业 `tzmLS9` / 本机 Python PID 33240 于 2026-11-13T15:08:08Z 启动，15:14:02Z 结束，exit 0，历时 5m54s（评估器报告 `elapsed_sec=346.21`）。RTX 3060，Conda dyg-cold，确定性设置。
- 覆盖：CIFAR10 官方验证集**全部 5000 图**，固定顺序、batch128、无 mask、无测试集推理；teacher seed41，scale1，T∈{1,2,4,8}；三行 `dc_adapt_analog_head` / `per_consumer` / `cse_only` = 12 组，每组保存逐样本 indices/labels/teacher_logits/logits 的 NPZ（共 4.4 MiB）。
- 教师精度 0.7066，与冻结参考一致；规范索引/标签与教师 logits 逐位校验通过；源/校准/数据/作者源码前后哈希一致；未触发计算上限。

## 结果（单种子描述性测量，非 E1 验收）

| 行 | T1 | T2 | T4 | T8 |
|---|---:|---:|---:|---:|
| dc_adapt_analog_head | 68.26% | 70.78% | 70.66% | 70.68% |
| per_consumer | 44.98% | 67.96% | 70.74% | 70.66% |
| cse_only（generic 别名） | 44.98% | 67.96% | 70.74% | 70.66% |

固定列指标（accuracy / teacher_agreement / logit_mse，共 36 行）见本检查点 `METRICS.csv`；完整指标、逐站点事件、操作计数与环境见 `report.json`。

## 跨运行独立一致性核验

把本轮 `per_consumer` 与 `cse_only` 的逐样本 logits 与既有独立运行 `results/e1-seed41-affine-v1/generic_T{1,2,4,8}.npz` 逐位比较：四个 T 全部 `np.array_equal=True`，labels 与 teacher_logits 亦一致。这与补充条款的预期相符——数值轨迹相同，**执行/编码状态与成本语义不同**，因此不能把 per_consumer 或 CSE-only 记成独立的方法胜出或重复统计证据。

新增的 `dc_adapt_analog_head` 是分类头匹配补充要求的控制行：只编码四个图分支 ReLU 站点并复用原始模拟 MLP 头一次；历史六站点 DC-adapt 仍是另一条必需行，二者不得混称。

## 明确不能下的结论

- **E1 未通过**：seeds 42/43 未跑（需各自冻结的全训练无标签校准）、共享 theta 附加对照未跑、完整已知操作成本与匹配墙钟未测、历史六站点 DC-adapt 行与其余必需行的汇总未完成。
- 0.3pp + 完整成本/延迟的双门槛**不可判定**；本轮只有部分观测计数器，没有 dense MAC、内存流量、硬件延迟或能耗，不能声称节能/加速。
- 无 Story 判定、无新测试集推理、无论文或独立 CCFA≥7 评审。P4 仍为 running。

## 产物、去标识与哈希

- 结果目录：`review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/results/e1-seed41-controls-v1/`。本地原始 `report.json` 含绑定输入的绝对路径键与钉死源码的本机路径，按仓库既有惯例**不公开**，原始 SHA256 `34883e4ce199a63484cc0fab3fb9f2e137aeaa8fae10ee33039b8a4b01d16a20`。
- 公开副本 `report.public.json` SHA256 `6d633bfca0ff6a49d794223bd5df0bad194c1a1dbde35dba5a14642c5da62b73`，由 `deidentify_report_v1.py` 生成：只替换盘符路径（仓库根→`repo/`、数据根→`data/`、钉死源码→`<pinned-dcgs>/`）与 `author_repo`，**所有数值、哈希、12 行指标与逐样本证据逐字不变**；副本内嵌原始 SHA 供本地复核。原始科学文件从未被改写。
- 上传的是公开副本 + 12 份逐样本 NPZ（合计约 4.4 MiB，单文件远低于 50 MiB）、运行日志、指标表与检查点清单；不上传原始 report.json、数据缓存、权重包或失败大产物。

## 下一步

- 为 teachers 42/43 生成冻结的全训练无标签校准，再以不变流程重跑全部行；补齐共享 theta 控制与历史六站点 DC-adapt 行；进入 E3 完整成本与 E2/E4/E6 对照活动。任何失败原样保留。
