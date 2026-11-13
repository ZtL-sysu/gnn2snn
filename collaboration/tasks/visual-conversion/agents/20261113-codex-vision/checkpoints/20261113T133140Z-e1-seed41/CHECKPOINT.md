# Checkpoint: 20261113T133140Z-e1-seed41

- Task/agent: visual-conversion / 20261113-codex-vision
- Status: in_progress
- Protocol: research/visual-graphs/continuation_20261113/PROTOCOL-v3.md SHA256 8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543; head-matching addendum unchanged.
- Manifest: manifest.json

实际运行：本机Conda/RTX3060，code/eval_affine_validation.py，scale1，seed41，完整5000官方验证图，三模式×四T。会话26002已退出0，PID23080已不存在；没有重复启动。约352秒运行，首批峰值分配约2.55GB是资源诊断，不是硬件延迟基准。

实测：centered T4/T8=70.66/70.70%；first_analog=70.68/70.72%；generic=70.74/70.66%。各mode/T逐样本NPZ和哈希完整，主智能体独立于JSON字段重新计算accuracy/agreement/MSE，覆盖与哈希均通过。这里的数值复算是same-agent数值审计，不是独立论文评审。

还实际调用了钉死的官方DCGS接口：原生PyG输入/模块元信息/FX路径的真实错误已记录；Tensor与元信息适配可执行15个源单元，输出有限4×2×10前缀。但只用了2个合成图，未完成官方全train阈值迭代或转换保真验证，不能算E2成功。

入场审核的另一个agent提交fa1403b实际REVIEW.md只有1字节，未被采信，未合并；本轮准入由主智能体同角色审查完成，旧有效数值复现评议仍保留。最终稿仍须重新取得实质完整的独立CCFA≥7。

支持：预定配置在全验证集上的真实精度/一致率/误差/活动计数。未支持：完整E1通过、方法优于所有对照、完整成本/墙钟/节能、新颖性或论文验收。下一步补齐剩余对照、shared-theta与三种子，之后才能按冻结标准判定。无新测试推断；不改变scale或阈值寻找有利结果。
