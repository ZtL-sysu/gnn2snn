# 复现与恢复

1. 使用Conda环境。原执行环境dyg-cold：Python3.13、Torch2.11+cu128、PyG2.8、RTX3060 12GB；环境导出另存。版本以原run metadata为准。
2. 运行`python tools/restore_evidence.py`展开约几十MB的关键证据包，核验ZIP和每个文件SHA，不覆盖不同内容。
3. 完整视觉代码在research/visual-graphs/takeover_20261112。脚本及哈希约定保留原件，部分绝对路径仍对应原C:/code/phd-snn布局；跨机器使用需明确重定位并记录差异，不能回改既有实验结果。
4. 官方SuperPixel MNIST55k/5k/10k、CIFAR10 45k/5k/10k可用PyG GNNBenchmarkDataset重新下载。保留完整官方划分、完整图，不切小测试集、不在主轨mask网络。
5. DCGS依赖：https://github.com/h-z-h-cell/ANN-to-SNN-DCGS ，固定commit5e0e4c73f1d0e8e17d81befa24ff947deec1f4be。配置--author-repo指向该checkout。源单元适配与原作者完整converter/threshold迭代并非同一实验。
6. 先读protocol、check与结果报告，再运行任何训练。原研究已暂停，本仓库没有自动训练／定时任务；恢复需用户新指令。

证据包保留9个强教师和完整测试预测。失败路线只在日志中总结，未打包其大型数组。新affine六组诊断与共享theta控制作为小型探索记录保留；训练agreement不是准确率，未据此确认测试提升、效率或新颖性。
