# Claude 审阅

本目录是 Claude 对公开仓库 `ZtL-sysu/gnn2snn` 的外部审阅，不改动原有研究证据，也不构成投稿许可。

- [innovation-and-rigor-review.md](innovation-and-rigor-review.md)：创新方向与"做扎实"建议（2026-11-14；对照的 `main` 为 `f19a2f2`）。结论：把主创新从"静态图单次推理内的编码效率"移到图特有的结构决策稳健性和流式视觉图的时间冗余，并补同比特量化 ANN 对照。
- [energy_ratio_check.py](energy_ratio_check.py)：复核审阅稿 4.2 节算术能耗比的只读脚本，只用标准库，读取 `research/visual-graphs/takeover_20261112/affine-six-profile-audit-v1.md`。

## 交接记录

### 2026-11-14 · claude · claude/review-suggestions

- **完成**：通读仓库并写出审阅稿；用仓库已归档数字推算三种 affine 模式的算术能耗比，脚本可复核；核对了审阅稿"参考"一节所列 8 个条目的出处页面。
- **未完成**：未运行仓库内任何训练、推理或检查脚本；审阅稿第 6 节的探针实验（B0、E0、C0）都还没做。
- **注意 / 坑**：本仓库根目录没有 `AGENTS.md` 和 `docs/PROGRESS.md`，因此交接记录写在这里；审阅稿第 9 节列出了未核验事项。
