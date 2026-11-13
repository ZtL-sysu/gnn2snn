# Checkpoint: 20261113T115304Z-e0

- Task / agent: visual-conversion / 20261113-codex-vision
- Status: in_progress
- Protocol: research/visual-graphs/continuation_20261113/PROTOCOL-v3.md SHA256 8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543
- Manifest: manifest.json

Changes: portablebaseline/affineval entries, actualindependentconsumer and matchedheadcontrols, currentfrozenprotocol and localGPU/syncscope. Historicalsource/results untouched; publicpathderivatives declared inPUBLICATION_INDEX.

Actualchecks: full5000 CIFAR10 seed41 baseline onRTX3060; rootNPZ/JSON/hash/coverage audit; fullcanonical repeat(all7arrays bitwise);5syntheticchecks; matched-controls actualheadhooks and per-layer/channelθ fixture; affineevaluator2pytest checks. E0v3freshreportcomplete; oldv1/v2failuresretained. No candidateofficialdata or newtest inference.

Measuredmetrics: metrics.csv and research/visual-graphs/continuation_20261113/results/e0_seed41_val.json. Necessarysamplepredictions: inputs/canonical-baseline-val.npz. Source/model/cal/data identities bound. HistoricalDC8onechangedclass(0.02%), both predictionswrong; logitdiffs reported without looserDCtolerance. Same-environment repeatability only, no crossdevice guarantee.

Supported: reproducibleaggregatebaselinefeasibility, currentdeterministic numericalrepeatability and testeddataflowcorrectness. Unproven: candidatemethod advantage, fulloperationcost/latency/energy/novelty/crossvisualgenerality and finalpaperreview. NoSOTA/5×claim.

发布前再次从独立工作区展开37个原证据文件，使用待发布代码、恢复的权重/校准及公开补充的历史参考，完成全部5000验证图。结果见 results/portable-e0-validation.json（位于本检查点对应研究目录）；全部预测数组再次与已公开canonical逐位一致。未重复上传同一组预测或已在原ZIP内的权重。

实际命令包括 `python tools/restore_evidence.py`、`python code/check_matched_controls.py --author-repo $env:DCGS_REPO`、`python -m pytest code/test_eval_affine_validation.py -q`，以及研究目录README给出的完整 `code/eval_teacher.py` 命令；均在本机Conda中执行。两项pytest与控制合成检查通过。当前无官方实验进程运行；两个局部review/probe任务因用户暂停而中断，未将其预期产物当完成证据。

Next: 接续中断的审核/实际DCGS路径探测并完成入场验收，再跑冻结的完整5000图E1；保留全部对照、分类头匹配及随后三种子/成本检查。Failurelogs summarized; repeatedfailuredata and privateabsolute dependencies remainlocal withoriginalhashes.
