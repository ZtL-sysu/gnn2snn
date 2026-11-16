"""Render review-round-1 analysis figures from frozen data tables."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE = "#32688e"
ORANGE = "#d9822b"
GRAY = "#68737d"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def finish(fig, name: str) -> None:
    fig.tight_layout(pad=0.55, rect=(0, 0.10, 1, 1))
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def hard_stratification() -> None:
    frame = pd.read_csv(DATA / "hard_intervention_stratified.csv")
    frame = frame[frame.readout == 1]
    labels = ["Coord −", "Coord +", "Residual −", "Residual +"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
    for ax, dataset in zip(axs, ["MNIST", "CIFAR10"]):
        group = frame[frame.dataset == dataset]
        orders = {
            "deterministic_coordinate": {1: 0, -1: 1},
            "residual": {1: 2, -1: 3},
        }
        values = [None] * 4
        for _, row in group.iterrows():
            values[orders[row.selector][row.direction]] = row.adverse_rate_percent
        values = pd.Series(values, index=labels)
        ax.bar(range(4), values.to_numpy(), width=0.68, color=BLUE)
        ax.axhline(50, color=GRAY, lw=0.7, ls="--")
        ax.set_xticks(range(4), labels, rotation=0, ha="center")
        ax.set_title(dataset, fontsize=10)
        ax.grid(axis="y", color="#edf0f2", lw=0.5)
        ax.tick_params(axis="x", labelsize=8, pad=4)
        for index, value in enumerate(values):
            ax.text(index, value + 1.1, f"{value:.1f}%", ha="center", fontsize=8)
    axs[0].set_ylabel("Adverse margin rate among MSE improvements (%)")
    fig.suptitle("The adverse rate varies with the intervention rule", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "One teacher, 512 exposed validation graphs per task; repeated interventions are not independent samples.",
        ha="center",
        fontsize=7.6,
    )
    finish(fig, "hard_intervention_stratified")


def temporal_decisions() -> None:
    frame = pd.read_csv(DATA / "temporal_decision_effects.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
    for ax, dataset in zip(axs, ["MNIST", "CIFAR10"]):
        group = frame[frame.dataset == dataset].reset_index(drop=True)
        x = range(len(group))
        changes = group.prediction_changes.to_numpy()
        lost = group.agreement_lost.to_numpy()
        gained = group.agreement_gained.to_numpy()
        ax.bar(x, changes, width=0.58, color=BLUE, label="prediction changes")
        ax.bar(x, gained, width=0.24, color="#3a8f65", label="agreement gained")
        ax.bar(x, -lost, width=0.24, color=ORANGE, label="agreement lost")
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_xticks(list(x), ["Halve", "Double"])
        ax.set_title(dataset, fontsize=10)
        ax.grid(axis="y", color="#edf0f2", lw=0.5)
        ax.tick_params(axis="x", labelsize=8, pad=4)
        for index, value in enumerate(changes):
            ax.text(index, value + 0.6, f"+{value}", ha="center", fontsize=8.5)
        for index, value in enumerate(lost):
            ax.text(index, -value - 1.1, f"{value}", ha="center", color="#a24b10", fontsize=7.5)
    axs[0].set_ylabel("Graphs out of 128")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=8)
    fig.suptitle("Exact local T2 equality changes downstream decisions", y=1.06)
    fig.text(
        0.5,
        -0.02,
        "All eligible first-site codewords exchanged; local decoded tensors and local event counts are bitwise matched.",
        ha="center",
        fontsize=7.6,
    )
    finish(fig, "temporal_decision_effects")


def main() -> None:
    hard_stratification()
    temporal_decisions()


if __name__ == "__main__":
    main()
