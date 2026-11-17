# AGENTS.md — phd-snn 项目入口

> 任何 AI 工具进入本项目先读本文件。研究状态看 `collaboration/STATE.md`，任务看 `collaboration/TASKS.md`。

## 项目与运行

- 项目：GNN→SNN 转换、动态图/视觉图实验、相关论文与历史研究资料总入口。
- 环境与技术栈：本机运行 Python 代码使用 Conda 环境 `C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe`。
- 主要目录：
  - `auto-paper/`：按日期编排的完整研究流程、数据、结果、快照与论文材料。
  - `archive-gnn2snn/`：从同级目录收拢的 gnn2snn 工作树、参考实现、打包脚本和重复项隔离区。
  - `fly-connectome-cvpr2027/`、`third_party/`：专题研究与第三方代码。
  - `_nv*/`、`pull003/`、`round-*`：平台拉取结果、探针脚本和审计日志。
  - `collaboration/`：跨 Agent 接手文档。

## 协作文档位置

- `collaboration/STATE.md`：目标、当前阶段、已验证/假设/被推翻结论、关键决策。
- `collaboration/TASKS.md`：任务状态、负责人、范围。
- `collaboration/agents/<agent_id>.md`：各 Agent 的交接记录。

## 接续规则

1. 开始：按顺序读 `AGENTS.md` → `collaboration/STATE.md` → `collaboration/TASKS.md` → 相关 `agents/*.md`，再按任务需要读源码、配置、结果或论文。
2. 核实：旧计划、旧日志只作背景；重要结论必须核对实际证据文件；日志里写着“运行中”不等于进程还活着。
3. 认领：在 TASKS 标 `doing` 并写范围；任务状态是记录，不是进程锁。
4. 结束：更新 TASKS 里自己的任务状态，并写或更新自己的 `agents/<id>.md`。
5. Git：顶层资料同步使用显式 `git add -- <path>`；不 force push、不 `reset --hard` / `git clean`、不覆盖他人未提交修改、不上传原始数据/模型权重/凭证。

## 当前注意事项

- `archive-gnn2snn/trash/` 是重复项和 junk 隔离区，未确认前不要恢复或上传。
- `archive-gnn2snn/gnn2snn-main`、`gnn2snn-cvpr-sprint`、`gnn2snn-wb-resume` 是同一 GitHub 仓库的 worktree，Git 指针已改到新路径。
- `archive-gnn2snn/gnn2snn-dcgs-reference` 是独立第三方参考仓库，不要当成普通副本删除。
