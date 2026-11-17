# Agent: codex-reorg-20261117

## 当前任务

按用户要求整理 `phd-snn` 与同级 `gnn2snn*` 目录，统一入口到 `C:\code\phd-snn`，保留数据/研究代码/有用权重，删除或隔离无用重复项，并按 `research-github-sync` 模板写本地日记。

## 已完成

- 只读盘点 7 个相关根目录，确认 `gnn2snn` 主仓库、`gnn2snn-cvpr-sprint` 与 `gnn2snn-wb-resume` 是同一仓库 worktree；`gnn2snn-dcgs-reference` 是独立第三方参考仓库；`gnn2snn-skill-publish` 是同一仓库的发布快照；`gnn2snn-prep` 是 staging/打包缓存。
- 把 5 个 gnn2snn 仓库/快照移动到 `archive-gnn2snn/`，并修复 `gnn2snn-cvpr-sprint` 与 `gnn2snn-wb-resume` 的 `.git` 指针和主仓库 `worktrees/*/gitdir`。`git worktree list --porcelain` 已验证新路径。
- 把 `gnn2snn-prep` 的唯一脚本、历史 JSONL、清单与 `publication-redactions.json` 保留到 `archive-gnn2snn/gnn2snn-prep-archive/`。
- 将 2 份重复大压缩包（整包 2.4 GiB + 5 分卷 2.5 GiB）移入 `archive-gnn2snn/trash/`；保留 `release-files.json` 和 `release-assets.json` 作为获取/校验依据。
- 对最大重复组按 SHA256 去重，66 个文件约 1.40 GiB 移入 `archive-gnn2snn/trash/large-duplicates/`；40 个 0 字节或明显 junk 移入 `archive-gnn2snn/trash/junk/`。
- 建立 `AGENTS.md`、`collaboration/STATE.md`、`collaboration/TASKS.md` 与本交接记录。

## 关键发现

- 大 `.pt` 文件中，`data/pyg_bench/**` 是 MNIST/CIFAR 图数据，不是模型权重；teacher 权重是 `teacher_probe/snapshots/*.npz` 与 `full_teacher_runs/**/best.pt|last.pt`。
- `research-evidence.zip` 和 5 分卷内容相同，且所有映射文件都能在 `phd-snn` 找到，无唯一科研内容。
- 顶层 `phd-snn` 原本不是 Git 仓库；直接 commit/push 前需确认目标 remote。

## 未完成与阻塞

- GitHub 同步目标已由历史授权文档确认：`USER_UPDATE-local-github-sync-v1.md` 写明用户为 `ZtL-sysu/gnn2snn` 调用 research-github-sync，并允许向 agent 任务分支增量 push curated 内容。
- `archive-gnn2snn/trash/` 保留约 6.2 GiB，未永久删除。
- 嵌套仓库中所有原有未提交修改都未触碰，包括 `third_party/SpikeGraphormer` 的 `medium/dataset.py` 修改和 `datasets/` 未跟踪数据。

## 下一步（Resume Entry）

- 目标：完成顶层 Git 初始化与 GitHub 同步。
- 入口：先读 `AGENTS.md` → `collaboration/STATE.md` → `collaboration/TASKS.md`；再运行 `git -C C:\code\phd-snn status --short --branch`。
- 已有结果：`archive-gnn2snn/` 已收拢；`output/duplicates-inventory.json` 保留完整重复清单；`manifests/release-files.json` 可定位大文件。
- 建议动作：1. 向用户确认目标仓库；2. 创建 `.gitignore` 排除大文件、trash、嵌套 `.git`、缓存；3. 显式 add `AGENTS.md`、`collaboration/`、`README.md` 与必要小文档；4. commit/push 后用 `git ls-remote` 验证远端 SHA。
- 注意：不要使用 `git add .` / `git add -A`；不要上传原始数据、checkpoint、trash 或凭证；不要清理嵌套仓库的未提交修改。

## 相关位置

- `C:\code\phd-snn\archive-gnn2snn\`：统一归档入口。
- `C:\code\phd-snn\archive-gnn2snn\trash\`：重复项隔离区，约 6.2 GiB。
- `C:\code\phd-snn\archive-gnn2snn\gnn2snn-prep-archive\manifests\release-files.json`：878 个原始数据/权重路径和哈希映射。
- `C:\code\phd-snn\output\duplicates-inventory.json`：SHA256 重复清单。
