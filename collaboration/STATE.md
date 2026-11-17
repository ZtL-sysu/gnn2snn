# Project State

## Overview

- 项目目标：统一保存 GNN→SNN 研究的数据、代码、模型权重、论文与可接续记录。
- 当前阶段：2026-11-17 完成同级 `gnn2snn*` 资料归档、去重、根目录 legacy 归档与 GitHub 研究同步。
- 当前路线：以 `phd-snn` 为唯一本地入口；保留数据集、研究代码和有用权重；大文件与原始 checkpoint 留在本地，不上传 GitHub。
- 下一步重点：整理记录已同步到 GitHub 分支 `docs/resource-reorg-20261117`；重复项 trash 已按用户确认永久删除，剩余为 PR 与少量探针代码是否上传的选择。
- 最后更新：2026-11-17。

## Key Decisions

### 2026-11-17：以 `phd-snn` 为统一入口

- 决定：把同级的 `gnn2snn`、`gnn2snn-cvpr-sprint`、`gnn2snn-wb-resume`、`gnn2snn-skill-publish`、`gnn2snn-dcgs-reference` 移动到 `archive-gnn2snn/`，修复 Git worktree 指针。
- 原因：这些目录分别是同一 GitHub 仓库的不同 worktree/快照或独立参考实现，移动后可统一备份且保留原有 Git 状态。
- 放弃方案：整包复制或保留在 `C:\code` 同级，会造成重复和路径混乱。
- 依据：`git worktree list --porcelain`、各目录 Git status/remote。

### 2026-11-17：`gnn2snn-prep` 只保留唯一材料

- 决定：保留构建/校验脚本、清单和历史 JSONL；删除或隔离重复证据归档。
- 原因：`research-evidence.zip` 与 5 个分卷是同一份去重 blob 归档；`release-files.json` 的 878 个路径都指向 `phd-snn` 中已有文件，抽查 SHA256 与字节数一致。
- 放弃方案：保留两份约 2.4 GiB/2.5 GiB 的大压缩包。
- 依据：`archive-gnn2snn/gnn2snn-prep-archive/manifests/release-files.json`、`release-assets.json`。

### 2026-11-17：大图数据与 Teacher 权重分开放置

- 决定：保留 `teacher_probe/snapshots` 的 teacher 权重与 `full_teacher_runs` 的 best/last 权重；保留 `data/pyg_bench` 中 MNIST/CIFAR 图数据集。
- 原因：经文件类型抽查，`T1_MNIST_seed*.npz` 是 teacher snapshot；大 `.pt` 是 PyG 图数据（list of Data，含 `x/pos/y/edge_index`），不是模型权重。
- 放弃方案：按文件大小误删大 `.pt` 数据集。
- 依据：`output/duplicates-inventory.json`、Conda 下对候选 `.pt` 的归档 pickle 结构抽查。

## Verified（已验证，有证据）

- 顶层 Git 已初始化并创建分支 `docs/resource-reorg-20261117`；提交 `157db24` 已 push 到 `ZtL-sysu/gnn2snn`，`git ls-remote` 验证远端 SHA 与本地 HEAD 一致。
- 归档后 `git -C archive-gnn2snn/gnn2snn-main worktree list --porcelain` 显示的 3 个同级 worktree 都指向新路径；其余 5 个 worktree 在用户目录下，未移动。
- `verified-evidence.zip` 在 6 个位置 SHA256 完全相同；只保留 `archive-gnn2snn/gnn2snn-main/artifacts/verified-evidence.zip`，其余 5 份移入 trash。
- 对 7 个根目录做文件大小分组 + SHA256 校验，识别 12155 个重复组；只处理前 25 个最大浪费组，共 66 文件约 1.40 GiB 移入 `archive-gnn2snn/trash/large-duplicates/`。完整重复清单曾写入 `output/duplicates-inventory.json`，因含本机全路径已从工作区删除；保留 trash 中的实际隔离副本作为恢复依据。
- 2026-11-17 用户确认“删除吧，把那些没用重复的”：已递归删除 `archive-gnn2snn/trash/`，共 124 文件约 6.38 GiB，删除后路径不存在。
- 2026-11-17 根目录二次整理：18 个 `_r*.py` 探针脚本、平台结果/round/WSL 日志/路径指针、10 个 `_nv*` 运行目录及 12 个 `_nv*` 根文件移入 `legacy/`；旧 `_nv*` 路径用 Junction（目录）和 SymbolicLink（文件）保持兼容。验证无跟踪文件删除。

## Hypothesis（当前假设）

- 其余小文件重复组仍可能有整理价值，但单体收益低；除非需要发布镜像，暂不做全量清理。

## Pending（待验证）

- `third_party/SpikeGraphormer` 仍有原有未提交修改和未跟踪数据，整理过程未触碰。
- `legacy/` 中可能含平台 token 读取逻辑和本机绝对路径，不上传 GitHub。
- 顶层工作区仍有未跟踪的小型 `_r*_probe.py` 和 `_current_*.txt`；它们可能含平台 token 读取逻辑或路径配置，未加入本次公开提交。

## Superseded（已推翻／已替代，勿引用）

- 旧路径 `C:\code\gnn2snn*` 作为同级独立目录的方案 → 替代者：`C:\code\phd-snn\archive-gnn2snn\`；作废原因：统一入口与修复 Git worktree 指针。

## Blockers

- GitHub 同步目标仓库：历史授权文档 `archive-gnn2snn/gnn2snn-main/review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/USER_UPDATE-local-github-sync-v1.md` 明确写出用户已为 `ZtL-sysu/gnn2snn` 调用 research-github-sync，并允许向 agent 任务分支增量 push curated code/results/logs/handoffs；本次按该授权继续，但只同步小型文档与整理记录。

## Next

1. 用户决定是否从 `docs/resource-reorg-20261117` 创建 PR。
2. 如需上传平台探针代码，先逐个去除 token 读取逻辑和路径依赖。
