# 关键位置与消息保持：第二阶段研究

分支 `research/20261114-critical-site`，起点 `a428537`。长期目标仍为具有可辨识新贡献的方法论文；当前尚未达到。上一轮成果保留在 `../20261114_story/`。

后续更新：新增8个源分位数阈值端点和24个固定机会预算补偿端点，本阶段共142个科学初筛端点。补偿的消息MSE有改善，但分类/结构控制/T2联合门槛失败，未扩展多教师。[最新结果](analysis/RESCUE_SUMMARY.md)、[14问阶段总结](RESEARCH_SUMMARY.md)、[真实消息误差与计时](analysis/allocation_mechanism_cost.csv)。本轮按用户允许的路线D完成可靠实证交付：12页论文、4图4表、源码包与独立解包复编译；尚不具备方法论文投稿证据。入口：[论文](paper/main.pdf)、[源码包](delivery/paper_source.zip)、[最终14问](RESEARCH_SUMMARY.md)、[复编译核验](delivery/package_verification.json)。

三个入口：[研究路线与决策](DECISIONS.md)、[论文证据映射](EVIDENCE_MAP.md)、[实验结果与复现索引](analysis/SUMMARY.md)。[相关工作初审](RELATED_WORK.md)。

## 当前结果

- 两任务、教师41、T1/T2，两个固定学习率的消息膜电位校准未通过门槛。
- 33点硬坐标搜索排除了一个简单的surrogate解释，但粗网格不能证明容量上限。
- 固定八幅值的首位置局部Lloyd在MNIST T1取得82.23%，CIFAR10为48.44%；普通结构加权未优于该匹配局部码本。其存储预算高于膜电位基线，不声称同参数。不能仅凭MNIST正结果写成功方法论文。
- 最终20个工程单元检查通过（依赖库存在弃用警告）；原全位置局部臂四组预测逐位复现历史；142项历史冻结输入不变。
- 110个科学筛选端点（另有小工程smoke），仅训练/验证，不读取测试集。不是三教师稳健验证或未暴露确认。
- 完整消息交叉项码本修正仍未优于匹配局部：T1差值MNIST −1.171875pp，CIFAR10 −1.3671875pp。停止该结构损失扩张。本轮未形成成功的新方法；完整强源基线和独立确认仍是发表缺口。

## 复现

使用本机Conda `dyg-cold` Python。历史资产恢复和数据根要求参见上一轮README；默认资产路径在脚本中，可用 `--assets` 修改，权重与数据严格校验SHA256。示例：

```powershell
& 'C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe' -X utf8 research/20261114_critical/run_critical.py --dataset MNIST --lr 0.001 --output <new-output-directory>
& 'C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe' -X utf8 research/20261114_critical/run_hard_probe.py --dataset MNIST --output <new-output-directory>
& 'C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe' -X utf8 research/20261114_critical/run_codebook.py --dataset MNIST --output <new-output-directory>
& 'C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe' -X utf8 research/20261114_critical/summarize.py
```

所有执行器拒绝覆盖已有目录。训练前源码与原单元T1/2/4/8硬前向逐位校验；每轮保存source快照、数据/教师/校准哈希、索引、原始预测和参数。`screen` 为64步Adam .001，`optimizer_check` 为.01，`hard_probe` 为T1一次146坐标×33候选，`book_screen` 为16次Lloyd固定八幅值。不同优化预算不能合并排名。各run/DONE.json保存总耗时，训练JSON保存拟合耗时，汇总CSV为原始数组重新计算。

`full_book_screen` 由 `run_full_book.py` 生成，统一局部Lloyd起点后一次1168坐标的完整硬目标扫描，详见DECISIONS。汇总器校验初末完整目标不升和原始预测；不是部署性能优化。用户指定f-skills已本地更新，记录 `skill_update.json`。更新后规范从真实P0证据迁移，`pipeline-state.json` 逐阶段记录真实执行与验收；论文生产通过不等于方法有效或投稿准备完成。

本轮新码本仅首位置采用最近邻分界，其余单元不变；它是独立实验性量化机制，不可称完整作者DCGS流程。相同事件数量上限不等于相同能耗：非几何幅值需要查表/一般乘法，尚未测量该推理成本。

## 论文和复现范围

使用Conda环境执行：`python research/20261114_critical/paper/render_tex.py`；在TeX Live可用的环境中于paper目录运行`latexmk -pdf main.tex`。论文源码包包含相对路径所需资产；全量实验脚本、原始数组、配置/索引/哈希留在本目录及历史研究目录。数据集和模型权重仍是有哈希记录的外部依赖，不将大文件重复提交。

独立科学审阅发现的定义、选择暴露和成本边界已修正。新候选仍仅单教师探索性验证，源分位数适配并非完整DCGS复现。自动引用工具6/11验证通过、5条API检索误标由出版社原始页面逐项纠正；详细记录见 audits/P14_falseflags.md。最终源码包不包含未接受的ARC默认模板输出。
