# phd-snn

GNN→SNN 转换、动态图/视觉图实验与论文资料本地总入口。

## 入口文档

- `AGENTS.md`：项目范围、运行环境、目录职责与接续规则。
- `collaboration/STATE.md`：当前研究状态、关键决策、已验证结论和阻塞。
- `collaboration/TASKS.md`：任务状态。
- `collaboration/agents/codex-reorg-20261117.md`：资源整理交接记录。

## 资料布局

- `auto-paper/`：完整研究流程、数据、结果与论文材料。
- `archive-gnn2snn/`：同级 `gnn2snn*` 目录收拢后的统一归档入口。
- `fly-connectome-cvpr2027/`、`third_party/`：专题研究与第三方代码。
- `_nv*/`、`pull003/`、`round-*`：平台拉取结果、探针脚本与审计日志。
- `collaboration/`：跨 Agent 研究记忆。

## 数据与权重

大型数据集、teacher snapshots 和模型权重保留在本地；GitHub 只同步文档、代码与小型结果。大文件清单与 SHA256 映射见：

- `archive-gnn2snn/gnn2snn-prep-archive/manifests/release-files.json`
- `archive-gnn2snn/gnn2snn-prep-archive/manifests/release-assets.json`

重复项和 junk 隔离区已按用户确认永久删除；原始数据与权重清单仍保留在本地 manifest 中。

## 环境

本机运行 Python 代码使用 Conda 环境 `C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe`。
