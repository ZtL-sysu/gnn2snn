"""Publication figures. All plotted numbers are copied from recorded CSVs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE = "#1f4e79"
TEAL = "#0f6e68"
ORANGE = "#c65d12"
INK = "#1c2430"
GRAY = "#5c6770"
PALE_B = "#eef4fa"
PALE_T = "#e7f4f2"
PALE_O = "#fbf4ee"
PALE = "#f6f8f9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def card(ax, x, y, w, h, fc, ec, lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.55",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.6,
    ))


def overview():
    fig, ax = plt.subplots(figsize=(7.16, 3.62))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)

    # Top pipeline
    card(ax, 1.0, 40.6, 30.2, 8.6, PALE_B, BLUE)
    card(ax, 34.9, 40.6, 30.2, 8.6, PALE_B, BLUE)
    card(ax, 68.8, 40.6, 30.2, 8.6, PALE_T, TEAL)
    ax.annotate("", xy=(34.7, 44.9), xytext=(31.4, 44.9),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9))
    ax.annotate("", xy=(68.6, 44.9), xytext=(65.3, 44.9),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9))

    # superpixel icon
    nodes = [(4.2, 44.2), (7.0, 46.2), (9.6, 43.6), (6.6, 42.4), (10.6, 45.6)]
    for a, b in [(0, 1), (1, 2), (0, 3), (2, 4), (3, 2), (1, 4)]:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=BLUE, lw=0.6, zorder=2)
    for x, y in nodes:
        ax.add_patch(Circle((x, y), 0.55, facecolor="white", edgecolor=BLUE, lw=0.7, zorder=3))
    ax.text(20.5, 46.6, "Visual superpixel graph", ha="center", va="center", fontsize=8.0, color=BLUE, fontweight="bold")
    ax.text(20.5, 44.4, "Schematic nodes and edges", ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(20.5, 42.5, "Not a dataset image", ha="center", va="center", fontsize=6.2, color=GRAY)

    # frozen gnn blocks
    for i, yy in enumerate([46.4, 44.8, 43.2, 41.6]):
        ax.add_patch(FancyBboxPatch((36.4, yy), 6.4, 1.15, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=BLUE, lw=0.6))
        ax.text(39.6, yy + 0.55, f"G{i+1}", ha="center", va="center", fontsize=5.4, color=BLUE)
    ax.annotate("", xy=(44.2, 42.1), xytext=(44.2, 47.2),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.text(44.6, 44.6, "skip", fontsize=5.3, color=GRAY, rotation=90, va="center")
    ax.text(55.2, 46.5, "Frozen residual GNN", ha="center", va="center", fontsize=8.0, color=BLUE, fontweight="bold")
    ax.text(55.2, 44.2, "Fixed graph and weights", ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(55.2, 42.3, "Readout stays in place", ha="center", va="center", fontsize=6.5, color=INK)

    # events
    xs = [71.2, 73.0, 74.8, 76.6]
    hs = [1.5, -0.75, 0.75, 0.75]
    ax.plot([70.4, 77.4], [44.2, 44.2], color=TEAL, lw=0.5)
    for x, h in zip(xs, hs):
        ax.plot([x, x], [44.2, 44.2 + h], color=TEAL if h > 0 else ORANGE, lw=1.5, solid_capstyle="butt")
    ax.text(84.4, 46.6, "Finite-event conversion", ha="center", va="center", fontsize=8.0, color=TEAL, fontweight="bold")
    ax.text(84.4, 44.4, "Signed graded events", ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(84.4, 42.4, "Audit the first node site", ha="center", va="center", fontsize=6.5, color=INK)

    # Middle panels
    card(ax, 1.0, 10.55, 48.3, 28.7, "white", ORANGE, lw=1.0)
    card(ax, 50.7, 10.55, 48.3, 28.7, "white", TEAL, lw=1.0)
    ax.text(2.3, 37.5, "(a)  A better message can worsen a decision", ha="left", va="center",
            fontsize=7.5, color=ORANGE, fontweight="bold")
    ax.text(52.0, 37.5, "(b)  A matched endpoint can change behavior", ha="left", va="center",
            fontsize=7.5, color=TEAL, fontweight="bold")

    # decision geometry
    ax.plot([6.2, 22.4], [16.6, 32.6], ls=(0, (2.2, 1.6)), color=BLUE, lw=0.9, zorder=2)
    ax.annotate("", xy=(18.6, 33.4), xytext=(7.4, 24.6),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.35, mutation_scale=8))
    ax.annotate("", xy=(16.4, 23.2), xytext=(10.2, 26.8),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.45, mutation_scale=8))
    ax.add_patch(Circle((10.2, 26.8), 0.42, facecolor=INK, edgecolor=INK, zorder=4))
    ax.text(8.0, 34.6, "larger error", fontsize=6.1, color=GRAY, ha="left")
    ax.text(8.0, 33.1, "stays correct", fontsize=6.1, color=GRAY, ha="left")
    ax.text(25.2, 20.2, "smaller error", fontsize=6.1, color=ORANGE, ha="right")
    ax.text(25.2, 18.7, "crosses boundary", fontsize=6.1, color=ORANGE, ha="right")
    ax.text(7.4, 15.6, "schematic", fontsize=5.6, color=GRAY, ha="center")

    ax.text(36.2, 33.6, "Adverse margins among", ha="center", fontsize=6.3, color=INK)
    ax.text(36.2, 32.0, "message-improving interventions", ha="center", fontsize=6.3, color=INK)
    ax.text(36.2, 29.2, "40.9%   MNIST", ha="center", fontsize=8.2, color=ORANGE, fontweight="bold")
    ax.text(36.2, 26.8, "31.9%   CIFAR-10", ha="center", fontsize=8.2, color=ORANGE, fontweight="bold")
    ax.plot([27.6, 44.8], [24.6, 24.6], color="#eadfd4", lw=0.6)
    ax.text(36.2, 22.6, "MNIST graph 437", ha="center", fontsize=6.3, color=INK, fontweight="bold")
    ax.text(36.2, 20.8, "MSE  0.006047  →  0.006045", ha="center", fontsize=6.2, color=INK)
    ax.text(36.2, 19.0, "Margin  +1.97  →  −3.75", ha="center", fontsize=6.2, color=INK)
    ax.text(36.2, 17.2, "Prediction  5  →  4", ha="center", fontsize=6.2, color=INK)
    ax.text(36.2, 12.6, "Recorded one-event intervention", ha="center", fontsize=5.7, color=GRAY)

    # codewords in data coordinates inside panel b
    # map amplitude into y
    def ay(amp):
        return 20.2 + (amp + 0.5) * 7.2
    ax.plot([55.2, 66.2], [ay(0), ay(0)], color="#e6eaed", lw=0.5, zorder=1)
    ax.plot([55.2, 66.2], [ay(1.0), ay(-0.5)], "-o", color=BLUE, lw=1.25, ms=3.6, zorder=3)
    ax.plot([55.2, 66.2], [ay(0.5), ay(0.5)], "-o", color=ORANGE, lw=1.25, ms=3.6, zorder=3)
    ax.text(55.2, 18.6, "Tick 1", ha="center", fontsize=5.7, color=GRAY)
    ax.text(66.2, 18.6, "Tick 2", ha="center", fontsize=5.7, color=GRAY)
    ax.text(53.6, ay(1.0) + 0.15, "q", ha="right", va="center", fontsize=6.2, color=BLUE)
    ax.text(67.6, ay(-0.5), "−q/2", ha="left", va="center", fontsize=6.2, color=BLUE)
    ax.text(53.6, ay(0.5), "q/2", ha="right", va="center", fontsize=6.2, color=ORANGE)
    ax.text(67.6, ay(0.5) + 1.15, "q/2", ha="left", va="center", fontsize=6.2, color=ORANGE)
    ax.text(60.7, 15.6, "Same endpoint 3q/4, two events", ha="center", fontsize=6.1, color=INK)
    ax.text(60.7, 14.0, "Different prefixes and amplitudes", ha="center", fontsize=6.1, color=INK)
    ax.text(60.7, 12.4, "|A| sums to 3q/2;  |B| sums to q", ha="center", fontsize=5.7, color=GRAY)

    ax.text(86.6, 33.8, "Changed predictions", ha="center", fontsize=6.3, color=INK)
    ax.text(86.6, 31.6, "6/128    16/128", ha="center", fontsize=8.2, color=TEAL, fontweight="bold")
    ax.text(86.6, 29.6, "MNIST      CIFAR-10", ha="center", fontsize=5.8, color=GRAY)
    ax.text(86.6, 26.6, "Activity after halving", ha="center", fontsize=6.3, color=INK)
    ax.text(86.6, 24.4, "+3.41%   +4.99%", ha="center", fontsize=8.2, color=TEAL, fontweight="bold")
    ax.text(86.6, 22.4, "MNIST      CIFAR-10", ha="center", fontsize=5.8, color=GRAY)
    ax.text(86.6, 18.6, "Training graphs", ha="center", fontsize=6.0, color=INK)
    ax.text(86.6, 16.9, "128 per task", ha="center", fontsize=6.0, color=INK)
    ax.text(86.6, 14.2, "Not a timing-only test", ha="center", fontsize=5.7, color=GRAY)

    # outputs
    card(ax, 1.0, 1.15, 31.4, 7.7, PALE, INK, lw=0.7)
    card(ax, 34.3, 1.15, 31.4, 7.7, PALE, INK, lw=0.7)
    card(ax, 67.6, 1.15, 31.4, 7.7, PALE, INK, lw=0.7)
    ax.text(16.7, 6.7, "Decision fidelity", ha="center", fontsize=7.4, color=INK, fontweight="bold")
    ax.text(16.7, 4.5, "Teacher agreement", ha="center", fontsize=6.2, color=INK)
    ax.text(16.7, 2.9, "and class margins", ha="center", fontsize=6.2, color=INK)
    ax.text(50.0, 6.7, "Prefix fidelity", ha="center", fontsize=7.4, color=INK, fontweight="bold")
    ax.text(50.0, 4.5, "Intermediate and", ha="center", fontsize=6.2, color=INK)
    ax.text(50.0, 2.9, "terminal states", ha="center", fontsize=6.2, color=INK)
    ax.text(83.3, 6.7, "Whole-network activity", ha="center", fontsize=7.4, color=INK, fontweight="bold")
    ax.text(83.3, 4.5, "Global event counts", ha="center", fontsize=6.2, color=INK)
    ax.text(83.3, 2.9, "and measured runtime", ha="center", fontsize=6.2, color=INK)
    save(fig, "concept")


def temporal_mechanism():
    fig, axs = plt.subplots(1, 2, figsize=(6.95, 2.55))
    ticks = np.array([1, 2])
    a = np.array([1.0, -0.5])
    b = np.array([0.5, 0.5])
    axs[0].plot(ticks, a, "-o", color=BLUE, lw=1.5, ms=5.5, label="Codeword A: (q, −q/2)", zorder=3)
    axs[0].plot(ticks, b, "-o", color=ORANGE, lw=1.5, ms=5.5, label="Codeword B: (q/2, q/2)", zorder=3)
    axs[0].axhline(0, color="#d5dbdf", lw=0.6)
    axs[0].set_xticks([1, 2])
    axs[0].set_xlim(0.75, 2.38)
    axs[0].set_ylim(-0.85, 1.28)
    axs[0].set_ylabel("Signed amplitude / q")
    axs[0].set_xlabel("Active tick")
    axs[0].text(1, 1.08, "q", color=BLUE, ha="center", fontsize=7)
    axs[0].text(2.08, -0.5, "−q/2", color=BLUE, ha="left", va="center", fontsize=7)
    axs[0].text(1, 0.36, "q/2", color=ORANGE, ha="center", fontsize=7)
    axs[0].text(2.08, 0.5, "q/2", color=ORANGE, ha="left", va="center", fontsize=7)

    da = np.array([1.0, 0.75])
    db = np.array([0.5, 0.75])
    axs[1].plot(ticks, da, "-o", color=BLUE, lw=1.5, ms=5.5, zorder=3)
    axs[1].plot(ticks, db, "-o", color=ORANGE, lw=1.5, ms=5.5, zorder=3)
    axs[1].scatter([2], [0.75], s=48, facecolors="none", edgecolors=TEAL, lw=1.3, zorder=4)
    axs[1].annotate("same endpoint  3/4", xy=(2, 0.75), xytext=(1.05, 0.18),
                    fontsize=7.2, color=TEAL, ha="left",
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.7))
    axs[1].text(1, 1.08, "q", color=BLUE, ha="center", fontsize=7)
    axs[1].text(1, 0.36, "q/2", color=ORANGE, ha="center", fontsize=7)
    axs[1].set_xticks([1, 2])
    axs[1].set_xlim(0.75, 2.35)
    axs[1].set_ylim(-0.15, 1.28)
    axs[1].set_ylabel("Decoded prefix / q")
    axs[1].set_xlabel("Active tick")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for ax in axs:
        ax.grid(axis="y", color="#eef1f3", lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "temporal_mechanism")


def stratified():
    frame = pd.read_csv(DATA / "hard_intervention_stratified.csv")
    frame = frame[frame.readout == 1].copy()
    order = [
        ("deterministic_coordinate", -1, "Coord −"),
        ("deterministic_coordinate", 1, "Coord +"),
        ("residual", -1, "Residual −"),
        ("residual", 1, "Residual +"),
    ]
    colors = [BLUE, "#4f83b8", ORANGE, "#e08a45"]
    fig, axs = plt.subplots(1, 2, figsize=(6.95, 2.45), sharey=True)
    for ax, dataset, title in zip(axs, ["MNIST", "CIFAR10"], ["MNIST", "CIFAR-10"]):
        vals = []
        for sel, direction, _ in order:
            row = frame[(frame.dataset == dataset) & (frame.selector == sel) & (frame.direction == direction)].iloc[0]
            vals.append(float(row.adverse_rate_percent))
        ax.bar(range(4), vals, color=colors, width=0.72)
        ax.set_xticks(range(4), [x[2] for x in order])
        ax.set_title(title)
        ax.set_ylim(0, 68)
        ax.grid(axis="y", color="#eef1f3", lw=0.5)
        ax.set_axisbelow(True)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.3, f"{v:.1f}", ha="center", fontsize=7.3, color=INK)
    axs[0].set_ylabel("Adverse margin rate (%)")
    fig.tight_layout()
    save(fig, "hard_intervention_stratified")


def temporal():
    frame = pd.read_csv(DATA / "temporal_decision_effects.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.95, 2.5))
    labels = ["MNIST\nhalve", "MNIST\ndouble", "CIFAR-10\nhalve", "CIFAR-10\ndouble"]
    keys = [("MNIST", "halve"), ("MNIST", "double"), ("CIFAR10", "halve"), ("CIFAR10", "double")]
    changes, activity = [], []
    for ds, direction in keys:
        row = frame[(frame.dataset == ds) & (frame.direction == direction)].iloc[0]
        changes.append(int(row.prediction_changes))
        base = float(row.whole_network_events_base)
        changed = float(row.whole_network_events_changed)
        activity.append(100.0 * (changed - base) / base)
    cols = [BLUE, TEAL, BLUE, TEAL]
    ax = axs[0]
    ax.bar(range(4), changes, color=cols, width=0.7)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Graphs with a changed prediction")
    ax.set_ylim(0, 22)
    for i, v in enumerate(changes):
        ax.text(i, v + 0.35, str(v), ha="center", fontsize=7.5)
    ax.grid(axis="y", color="#eef1f3", lw=0.5)
    ax.set_axisbelow(True)
    ax = axs[1]
    bar_colors = [BLUE if v >= 0 else ORANGE for v in activity]
    ax.bar(range(4), activity, color=bar_colors, width=0.7)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Whole-network activity change (%)")
    ax.set_ylim(-2.2, 6.4)
    for i, v in enumerate(activity):
        if v >= 0:
            ax.text(i, v + 0.12, f"{v:+.2f}", ha="center", va="bottom", fontsize=7.2)
        else:
            ax.text(i, v - 0.12, f"{v:+.2f}", ha="center", va="top", fontsize=7.2)
    ax.grid(axis="y", color="#eef1f3", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "temporal_decision_effects")


def centered():
    frame = pd.read_csv(DATA / "centered_shared_theta.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.95, 2.45), sharex=True)
    for scale, color, marker in [(1, BLUE, "o"), (8, ORANGE, "s")]:
        g = frame[frame.scale == scale].sort_values("T")
        axs[0].plot(g["T"], g.centered_over_first_analog_mse, marker=marker, color=color, lw=1.3, label=f"scale {scale}")
        axs[1].plot(g["T"], g.centered_over_first_analog_adds, marker=marker, color=color, lw=1.3, label=f"scale {scale}")
    titles = ["Logit MSE", "Coded-projection additions"]
    for ax, title in zip(axs, titles):
        ax.axhline(1.0, color=GRAY, lw=0.7, ls="--")
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("Active ticks")
        ax.set_ylabel("Centered / raw-input coding")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.12)
        ax.grid(axis="y", color="#eef1f3", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7)
    axs[1].text(2.55, 1.035, "raw-input reference", color=GRAY, fontsize=6.2, ha="left")
    fig.tight_layout()
    save(fig, "centered_tradeoff")


def position():
    frame = pd.read_csv(DATA / "position_gap_closure_t1.csv")
    methods = ["site0", "site1", "site2", "site3", "site4", "site5", "nodes0_3", "head4_5"]
    labels = ["N1", "N2", "N3", "N4", "R1", "R2", "Nodes", "Head"]
    fig, ax = plt.subplots(figsize=(6.95, 2.55))
    width = 0.36
    x = np.arange(len(methods))
    ax.axvspan(-0.5, 5.5, color="#f7f9fb", zorder=0)
    ax.axvspan(5.5, 7.5, color="#fbf6f1", zorder=0)
    for ds, offset, color in [("MNIST", -width / 2, BLUE), ("CIFAR10", width / 2, ORANGE)]:
        q = frame[frame.dataset == ds].set_index("method").loc[methods]
        means = 100 * q.agreement_gap_closed_mean.to_numpy()
        sds = 100 * q.agreement_gap_closed_sample_sd.to_numpy()
        ax.bar(x + offset, means, width=width, color=color, label="MNIST" if ds == "MNIST" else "CIFAR-10", zorder=2)
        ax.errorbar(x + offset, means, yerr=sds, fmt="none", ecolor=INK, elinewidth=0.6, capsize=2, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Disagreement gap closed (%)")
    ax.set_ylim(-12, 100)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.30, 0.98))
    ax.grid(axis="y", color="#e6ebef", lw=0.5)
    ax.set_axisbelow(False)
    ax.axvline(5.5, color="#d9dee3", lw=0.6, zorder=1)
    fig.tight_layout()
    save(fig, "position_recovery")


if __name__ == "__main__":
    overview()
    temporal_mechanism()
    stratified()
    temporal()
    centered()
    position()
    print("figures written")
