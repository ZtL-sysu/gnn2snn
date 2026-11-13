# 2026-11-13 研究续接检查点

本目录是精简工作快照；历史大证据包和旧目录不搬移、不重复上传。先读 CURRENT_SCOPE.md、PROTOCOL-v3.md、PROTOCOL-E1-head-addendum-v1.md；协作入口位于仓库 collaboration/。

已测：CIFAR10 GatedGCN seed41，官方完整5000验证图，ANN70.66%、DC4=70.72%、DC8=70.66%。当前固定环境完整重复的全部预测数组逐位一致；历史T8差1张预测且前后均错误，绝不称逐样本历史复现。results/e0_seed41_val.json保留指标和source/model/cal/data哈希；inputs/canonical-baseline-val.npz保存逐样本数组。数值差异、未通过的旧验收及scope限制均保留。

复现依赖：先展开仓库 artifacts/verified-evidence.zip（tools/restore_evidence.py）；设置 GNN2SNN_ROOT 为仓库根、GNN2SNN_DATA_ROOT 为官方PyG数据根、DCGS_REPO 为钉死的作者仓库。使用本机Conda dyg-cold。作者commit5e0e4c73f1d0e8e17d81befa24ff947deec1f4be。

命令入口（从本目录运行，输出使用全新的子目录）：

```powershell
conda run -n dyg-cold python code/check_matched_controls.py --author-repo $env:DCGS_REPO
conda run -n dyg-cold python -m pytest code/test_eval_affine_validation.py -q
conda run -n dyg-cold python code/eval_teacher.py --help
conda run -n dyg-cold python code/eval_affine_validation.py --help
```

E0的历史参考已按原字节补充到 inputs/historical-gd-val1.npz，SHA256 `473d88de22c46dce7fede61df1d6a8abdc7bdd682abd4a64704f978d28f5f177`；默认确定性参考是 inputs/canonical-baseline-val.npz。历史文件附带原记录的朴素IF控制，不能重新命名为忠实QCFS/FTBC。官方数据与作者仓库不重复打包。当前复现使用原Conda环境+新进程/独立恢复目录，没有声称已从零重建依赖。

完整基线复现入口（先设置上述环境变量并展开原ZIP，输出目录须不存在）：

```powershell
$teacher = Join-Path $env:GNN2SNN_ROOT 'research/visual-graphs/takeover_20261112/full_teacher_runs/gated_cifar10_seed41_full_v1'
conda run -n dyg-cold python code/eval_teacher.py --teacher-dir $teacher --calibration "$teacher/gcal_v1.json" --reference inputs/historical-gd-val1.npz --author-repo $env:DCGS_REPO --converter both --split val --device cuda --output-dir results/reproduced-e0-v3
```

代码仍是稠密参考执行；部分事件计数不是完整成本/墙钟/节能证明。CSE-only是generic的明确别名；per-consumer实际使用各消费者独立源状态。旧DC含两层分类头编码，公平比较另需四层图分支编码+相同模拟分类头对照。未完成的对照/多seed/E2–E6不得记作完成。

PUBLICATION_INDEX.json 区分本地原始SHA与公开副本SHA；路径去标识只用于公开副本，绝不修改旧科学源或哈希来冒充原始证据。
