# 恢复认领与环境复核（R1）

## 本轮事实

- 用户 2026-11-13 22:52 +0800 明确恢复研究：按冻结协议与流水线推进，实验只用本机 RTX 3060，过程按 research-github-sync 增量同步到本仓库 agent 分支。原归档暂停状态对本 agent 不再生效；TASK.md 属 owner 写入范围，未改动。
- 建立独立 worktree 与唯一分支 `agent/visual-conversion/20261113-wb-resume`，输入 HEAD `2af2a975291b4cbc05b9be93fa418257b6ed3afe`（远端 codex-vision 最新）。
- 冻结协议三份文件与协作协议 SHA256 已本地复算，PROTOCOL-v3 与 TASK.md 记录一致，未改任何旧协议/证据/哈希。
- 既有 f-research-v1 运行目录（contract v4，P0–P3A passed，P4 running）完整续接，`--before P4` 校验通过。

## 检查

- 官方 CIFAR10 五个 processed 文件 SHA256 与 seed41 教师元数据逐项一致。
- 钉死 DCGS 源码 HEAD = `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`。
- Conda `dyg-cold`：Python 3.13.13 / torch 2.11.0+cu128 / CUDA 可用 / PyG 2.8.0.post1；GPU RTX 3060 12 GiB。
- `check_matched_controls.py` 合成检查 PASS；`test_eval_affine_validation.py` 2 passed。
- 三个教师 seed 权重目录均在本机存在。

## 局限

- 本轮没有运行官方数据的推理或训练，没有产生新的科学结果；未生成 METRICS.csv（无实验指标，不造空表）。
- 清单只证明文件身份，不证明科学结论。P4 仍为 running，E1 远未完成。
- 暂停期间被中断的 `20261113-luna-controls` 对照评估器任务没有任何已提交实现，需重新实现。
- 公开仓库不含本机绝对路径、凭证、数据缓存、大权重包；运行目录内的本地大产物不上传。

## 下一步

实现 `code/eval_controls_validation_v2.py` 及其测试，把 `matched_controls.py` 的三个控制接入冻结阈值的官方 5000 图验证；随后按 seed41→42/43 顺序执行固定网格，再计完整成本与 E2–E6。遇输入哈希/规范索引/协议/时钟不一致立即停并留证。
