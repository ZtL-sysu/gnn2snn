"""复核 claude 审阅稿中的算术能耗比（只读、只用标准库）。

数据来源（均为仓库内已归档文件，本脚本不修改它们）：
- 一致率与投影 ADD：research/visual-graphs/takeover_20261112/affine-six-profile-audit-v1.md 的 Prefix results 表
- 记账约定：同目录 affine-mode-cost-boundary-v1.md 与 gated-dual-arithmetic-screen-v1.md
  ADD 0.9 pJ、MUL 3.7 pJ（MAC 4.6 pJ）；单层门控已知项 4,151 pJ/N；
  门控层求值次数 generic 为 4T，first_analog/centered 为 1+3T；
  一次性基项按 centered 的优化实现 36,064 + 1,610*(B/N) pJ/N（对三种模式统一使用，属于对 generic/first_analog 略有利的近似）；
  ANN 已知项 843,514 + 14,789*(B/N) pJ/N（FP32，未做零值跳过）。
- N、B/N：gated-dual-arithmetic-screen-v1.md（N=5,294,119；B/N=0.008499998）

这是“最乐观下界”口径：只计 基项 + 门控已知项 + 事件投影 ADD。
未计：源单元状态与比较、1/t 解码、sigmoid、除法、scatter、BN、访存、分类头重复调用。
因此表中的比值是上界，真实比值只会更低。一致率是训练集上与教师预测的一致率，不是精度。

用法：python review-suggestions/claude/energy_ratio_check.py
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research/visual-graphs/takeover_20261112/affine-six-profile-audit-v1.md"

N = 5_294_119
B_OVER_N = 0.008499998
ADD_PJ = 0.9
GATE_PJ_PER_LAYER = 4_151
BASE_PJ = 36_064 + 1_610 * B_OVER_N
ANN_PJ = 843_514 + 14_789 * B_OVER_N


def rows():
    pattern = re.compile(r"^\| (generic|first_analog|centered) \| (\d+) \| (\d+) \| ([\d.]+) \| [\d.e-]+ \| [\d.e-]+ \| (\d+) \|$")
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            yield m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), int(m.group(5))


def main():
    found = list(rows())
    if len(found) != 18:
        raise SystemExit(f"期望 18 行，实际解析到 {len(found)} 行；审计表格式可能已变")
    print("| 模式 | scale | T | 教师一致率（训练集） | 投影 pJ/N | 已知项合计 pJ/N | ANN/候选 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for mode, scale, T, agree, adds in found:
        gate_evals = 4 * T if mode == "generic" else 1 + 3 * T
        proj = ADD_PJ * adds / N
        total = BASE_PJ + gate_evals * GATE_PJ_PER_LAYER + proj
        print(f"| {mode} | {scale} | {T} | {agree * 100:.2f}% | {proj:,.0f} | {total:,.0f} | {ANN_PJ / total:.2f}× |")


if __name__ == "__main__":
    main()
