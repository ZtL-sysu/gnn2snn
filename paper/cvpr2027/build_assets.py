"""Build the bounded CVPR 2027 evidence assets from frozen source records.

Run with: conda run -n dyg-cold python paper/cvpr2027/build_assets.py
No models are loaded and no experiment is run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SRC = {
    "ideal": ROOT / "research/20261114_story/analysis/ideal/aggregate.csv",
    "intervention_mnist": ROOT / "research/20261115_decision/runs/intervention_v2/MNIST41/graphs.csv",
    "intervention_cifar": ROOT / "research/20261115_decision/runs/intervention_v2/CIFAR1041/graphs.csv",
    "intervention_ci": ROOT / "research/20261115_decision/analysis/phase3/intervention_cluster_ci.csv",
    "centered_s1": ROOT / "research/visual-graphs/takeover_20261112/affine_centered_shared_s1_tr1.json",
    "first_analog_s1": ROOT / "research/visual-graphs/takeover_20261112/affine_first_analog_shared_s1_tr1.json",
    "centered_s8": ROOT / "research/visual-graphs/takeover_20261112/affine_centered_shared_s8_tr1.json",
    "first_analog_s8": ROOT / "research/visual-graphs/takeover_20261112/affine_first_analog_shared_s8_tr1.json",
    "temporal": ROOT / "research/20261115_decision/analysis/prefix_diagnostic/endpoints.csv",
    "scalar_note": ROOT / "research/20261115_decision/RQ1_diagnostic.md",
    "teacher_fidelity": ROOT / "research/20261114_story/analysis/confirmation/aggregate.csv",
    "historical_calibration": ROOT / "research/20261114_critical/analysis/endpoints.csv",
    "phase3_controls": ROOT / "research/20261115_decision/analysis/phase3/main_table.csv",
}

COLORS = {"blue": "#32688e", "orange": "#d9822b", "green": "#3a8f65", "red": "#bd4b4b", "gray": "#68737d", "purple": "#7863a6"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.2, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(key: str) -> pd.DataFrame:
    p = SRC[key]
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def write_csv(name: str, rows: list[dict], columns: list[str] | None = None) -> Path:
    path = DATA / name
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return path


def savefig(fig, stem: str):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def position_recovery():
    df = read_csv("ideal")
    d = df[(df.scope == "full_validation") & (df["T"] == 1) & (df.dataset.isin(["MNIST", "CIFAR10"]))]
    methods = ["baseline", "site0", "site1", "site2", "site3", "site4", "site5", "nodes0_3", "head4_5", "all6"]
    assert len(d) == 20 and set(d.method) == set(methods) and set(d.n_seeds) == {3} and set(d.n_examples) == {5000}
    rows = []
    for _, r in d.iterrows():
        rows.append({"dataset": r.dataset, "scope": r.scope, "method": r.method,
                     "T": int(r["T"]), "n_seeds": int(r.n_seeds), "n_examples_per_seed": int(r.n_examples),
                     "accuracy_mean_percent": r.accuracy_percent_mean,
                     "accuracy_sample_sd_pp": r.accuracy_percent_sample_sd,
                     "teacher_agreement_mean_percent": r.teacher_agreement_percent_mean,
                     "teacher_agreement_sample_sd_pp": r.teacher_agreement_percent_sample_sd,
                     "qualification": r.qualification})
    write_csv("position_recovery_t1.csv", rows)
    labels = ["Baseline", "Site 0", "Site 1", "Site 2", "Site 3", "Site 4", "Site 5", "Node sites\n0–3", "Head sites\n4–5", "All six"]
    lookup = {(r.dataset, r.method): r for _, r in d.iterrows()}
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=True)
    for ax, ds in zip(axs, ["MNIST", "CIFAR10"]):
        means = [lookup[(ds, m)].accuracy_percent_mean for m in methods]
        sds = [lookup[(ds, m)].accuracy_percent_sample_sd for m in methods]
        ax.errorbar(range(len(methods)), means, yerr=sds, fmt="o", ms=3.5, capsize=2,
                    lw=1.1, color=COLORS["blue"], ecolor=COLORS["gray"])
        ax.axhline(lookup[(ds, "baseline")].accuracy_percent_mean, color=COLORS["gray"], lw=.7, ls="--")
        ax.set_xticks(range(len(methods)), labels, rotation=38, ha="right")
        ax.set_title(ds)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", color="#e7e9eb", lw=.55)
    fig.suptitle("T1 ideal site recovery: full validation, three teacher seeds", y=1.02)
    fig.text(.5, -.03, "Mean ± sample SD across seeds; continuous causal ReLU oracle, not a deployable conversion.", ha="center", fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "position_recovery_t1")


def interventions():
    ci = read_csv("intervention_ci")
    rows, plot_data = [], {}
    for ds, key in [("MNIST", "intervention_mnist"), ("CIFAR10", "intervention_cifar")]:
        g = read_csv(key)
        assert set(g["T"]) == {1, 2} and g.actual_first_event_changed.astype(bool).all()
        g = g.copy()
        g["delta_message_mse"] = g.changed_message_mse - g.baseline_message_mse
        g["delta_teacher_margin"] = g.changed_margin - g.baseline_margin
        plot_data[ds] = g
        for t in (1, 2):
            q = g[g["T"] == t]
            c = ci[(ci.dataset == ds) & (ci["T"] == t)].iloc[0]
            improved = q.delta_message_mse < 0
            worse = q.delta_teacher_margin < 0
            n_improved = int(improved.sum())
            n_worse = int((improved & worse).sum())
            assert n_improved == int(c.message_improving_interventions) and n_worse == int(c.margin_worse)
            rows.append({"dataset": ds, "T": t, "graphs": int(c.graphs), "interventions": len(q),
                         "message_mse_improved": n_improved, "improved_and_margin_worse": n_worse,
                         "fraction": c.margin_worse_fraction, "cluster_bootstrap95_low": c.graph_cluster_bootstrap95_low,
                         "cluster_bootstrap95_high": c.graph_cluster_bootstrap95_high,
                         "margin_improved_count_within_mse_improved": int((improved & (q.delta_teacher_margin > 0)).sum()),
                         "qualification": "repeated interventions clustered by graph; teacher41; exposed validation512"})
    write_csv("hard_intervention_summary.csv", rows)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.85), sharex=True, sharey=True)
    for ax, ds in zip(axs, ["MNIST", "CIFAR10"]):
        q = plot_data[ds]
        for t, marker, color in [(1, "o", COLORS["blue"]), (2, "^", COLORS["orange"])]:
            z = q[q["T"] == t]
            ax.scatter(z.delta_message_mse, z.delta_teacher_margin, s=8, alpha=.32,
                       marker=marker, color=color, linewidths=0, rasterized=True, label=f"T={t}")
        ax.axhline(0, color=COLORS["gray"], lw=.65)
        ax.axvline(0, color=COLORS["gray"], lw=.65)
        ax.set_title(ds)
        ax.set_xlabel("Δ immediate message MSE (changed − baseline)")
        ax.grid(color="#edf0f2", lw=.45)
        row = next(x for x in rows if x["dataset"] == ds and x["T"] == 1)
        text = f"T1: {row['improved_and_margin_worse']}/{row['message_mse_improved']} worsened\n{100*row['fraction']:.2f}% [{100*row['cluster_bootstrap95_low']:.2f}, {100*row['cluster_bootstrap95_high']:.2f}]%"
        ax.text(.03, .97, text, transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
                bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "edgecolor": "#cbd1d6", "alpha": .92})
        ax.legend(frameon=False, loc="lower right", markerscale=1.1)
    axs[0].set_ylabel("Δ teacher-class margin (changed − baseline)")
    fig.suptitle("Hard-event interventions: message error and decision margin can move oppositely", y=1.02)
    fig.tight_layout()
    savefig(fig, "hard_intervention_mse_margin")


def centered_shared_theta():
    records = {}
    for s in (1, 8):
        for mode in ("centered", "first_analog"):
            k = f"{mode}_s{s}"
            records[(mode, s)] = json.loads(SRC[k].read_text(encoding="utf-8"))
        # Provenance metadata declares same shared thresholds; verify exact serialized vectors.
        assert records[("centered", s)]["threshold_vectors"] == records[("first_analog", s)]["threshold_vectors"]
    rows = []
    for s in (1, 8):
        for t in (1, 2, 4):
            a = records[("first_analog", s)]["profiles"][str(t)]
            c = records[("centered", s)]["profiles"][str(t)]
            rows.append({"scale": s, "T": t,
                         "first_analog_logit_mse": a["logit_mse"], "centered_logit_mse": c["logit_mse"],
                         "centered_over_first_analog_mse": c["logit_mse"] / a["logit_mse"],
                         "first_analog_projection_adds": a["prospective_projection_adds"],
                         "centered_projection_adds": c["prospective_projection_adds"],
                         "centered_over_first_analog_adds": c["prospective_projection_adds"] / a["prospective_projection_adds"],
                         "teacher_agreement_first_analog": a["teacher_class_agreement"],
                         "teacher_agreement_centered": c["teacher_class_agreement"],
                         "qualification": "shared fixed theta; one frozen seed41 CIFAR10 GatedGCN; label-free train split"})
    write_csv("centered_shared_theta.csv", rows)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.55), sharex=True)
    for ax, ratio_key, title in [(axs[0], "centered_over_first_analog_mse", "Logit MSE ratio"),
                                 (axs[1], "centered_over_first_analog_adds", "Projection ADD ratio")]:
        for s, color, mark in [(1, COLORS["blue"], "o"), (8, COLORS["orange"], "s")]:
            q = [r for r in rows if r["scale"] == s]
            ax.plot([r["T"] for r in q], [r[ratio_key] for r in q], marker=mark, ms=4, lw=1.2,
                    color=color, label=f"scale {s}")
        ax.axhline(1, color=COLORS["gray"], lw=.75, ls="--")
        ax.set_title(title)
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("Prefix T")
        ax.grid(axis="y", color="#e7e9eb", lw=.55)
    axs[0].set_ylabel("Centered / first_analog (lower is better)")
    axs[1].legend(frameon=False)
    fig.suptitle("Centered profile under exactly shared thresholds", y=1.02)
    fig.text(.5, -.02, "Label-free training diagnostics; one CIFAR10 teacher. Projection ADDs are a partial arithmetic proxy, not total compute.", ha="center", fontsize=7.1)
    fig.tight_layout()
    savefig(fig, "centered_shared_theta")


def temporal_prefix():
    df = read_csv("temporal")
    q = df[(df["T"] == 2) & (df.policy.isin(["all_eligible", "deterministic_half"]))]
    rows = []
    directions = ["halve", "double"]
    policies = ["all_eligible", "deterministic_half"]
    for ds in ["MNIST", "CIFAR10"]:
        for direction in directions:
            baseline = df[(df.dataset == ds) & (df.direction == direction) & (df.policy == "baseline") & (df["T"] == 2)].iloc[0]
            for policy in policies:
                r = q[(q.dataset == ds) & (q.direction == direction) & (q.policy == policy)].iloc[0]
                rows.append({"dataset": ds, "direction": direction, "policy": policy, "T": 2,
                             "baseline_whole_network_events": int(baseline.events), "candidate_whole_network_events": int(r.events),
                             "delta_events": int(r.events - baseline.events),
                             "delta_percent": 100 * (r.events - baseline.events) / baseline.events,
                             "local_decode_max_error": r.local_decode_max_error,
                             "local_T2_decode_checked": r.local_T2_decode_checked,
                             "train_graphs": 128})
    assert len(rows) == 8 and all(r["local_decode_max_error"] == 0 and r["local_T2_decode_checked"] for r in rows)
    write_csv("temporal_prefix_events.csv", rows)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=True)
    labels = ["Halve · all", "Halve · half", "Double · all", "Double · half"]
    keys = [(d, p) for d in directions for p in policies]
    for ax, ds in zip(axs, ["MNIST", "CIFAR10"]):
        vals = [next(r["delta_percent"] for r in rows if r["dataset"] == ds and (r["direction"], r["policy"]) == k) for k in keys]
        colors = [COLORS["blue"] if v < 0 else COLORS["orange"] for v in vals]
        bars = ax.bar(range(4), vals, color=colors, width=.68)
        ax.axhline(0, color=COLORS["gray"], lw=.7)
        ax.set_xticks(range(4), labels, rotation=18, ha="right")
        ax.set_title(ds)
        if ds == "MNIST":
            ax.set_ylabel("T2 whole-network event change vs baseline (%)")
        else:
            ax.set_ylabel("")
        ax.set_ylim(-2.5, 6.6)
        ax.grid(axis="y", color="#e7e9eb", lw=.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v + (0.45 if v >= 0 else -0.35), f"{v:+.2f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
        ax.text(.98, .96, "local T2 decode error = 0", transform=ax.transAxes, ha="right", va="top", fontsize=7.1,
                bbox={"boxstyle":"round,pad=.2", "facecolor":"white", "edgecolor":"#cbd1d6"})
    fig.suptitle("Equal local T2 decode does not fix whole-network event cost", y=1.02)
    fig.text(.5, -.03, "128 training graphs; whole-network events from saved endpoint table.", ha="center", fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "temporal_local_equal_global_events")


def compact_tables():
    teacher = read_csv("teacher_fidelity")
    base = teacher[(teacher.scope == "full_validation") & (teacher.method.isin(["baseline", "all6"])) & (teacher["T"].isin([1, 2]))]
    historical = read_csv("historical_calibration")
    hist = historical[(historical.phase == "screen") & historical.method.isin(["baseline", "all_local"]) & historical["T"].isin([1, 2])]
    p3 = read_csv("phase3_controls")
    p3q = p3[p3.method.isin(["local_t1", "logits", "kl", "ranking"]) & p3["T"].isin([1, 2])]
    rows = []
    for _, r in base.iterrows():
        rows.append({"evidence_family":"three_seed_teacher_fidelity", "dataset":r.dataset,"scope":r.scope,"method":r.method,"T":int(r["T"]),
                     "n_seeds":int(r.n_seeds),"accuracy_percent":r.accuracy_percent_mean,"accuracy_sd_pp":r.accuracy_percent_sample_sd,
                     "teacher_agreement_percent":r.teacher_agreement_percent_mean,"teacher_accuracy_percent":r.teacher_accuracy_percent_mean,
                     "events":"","parameters":"","qualification":"full validation; teacher seed-level mean±sample SD"})
    for _, r in hist.iterrows():
        rows.append({"evidence_family":"historical_calibration", "dataset":r.dataset,"scope":"teacher41_val512","method":r.method,"T":int(r["T"]),
                     "n_seeds":1,"accuracy_percent":r.accuracy_percent,"accuracy_sd_pp":"","teacher_agreement_percent":r.teacher_agreement_percent,
                     "teacher_accuracy_percent":"","events":"","parameters":r.parameters,"qualification":"selection-exposed validation; distinct teacher/data scope"})
    for _, r in p3q.iterrows():
        rows.append({"evidence_family":"phase3_ordinary_objective", "dataset":r.dataset,"scope":"teacher41_val512","method":r.method,"T":int(r["T"]),
                     "n_seeds":1,"accuracy_percent":r.accuracy_percent,"accuracy_sd_pp":"","teacher_agreement_percent":r.teacher_agreement_percent,
                     "teacher_accuracy_percent":r.teacher_accuracy_percent,"events":r.events,"parameters":r.parameters,
                     "qualification":"same 693-offset base; Adam64 lr .01; T share one parameter vector; exposed validation"})
    write_csv("results_context_compact.csv", rows)
    txt = SRC["scalar_note"].read_text(encoding="utf-8")
    assert "0.5078125" in txt and "0.75" in txt and "0.8" in txt and "0.6" in txt
    write_csv("temporal_scalar_counterexample.csv", [{
        "current_history_1":"(0.01, 1.58)", "current_history_2":"(1, -0.4)",
        "weighted_integral_both":0.8,"decoded_output_1":0.5078125,"decoded_output_2":0.75,
        "classification_boundary":0.6,"two_nonzero_events_each":2,
        "qualification":"controlled frozen scalar recurrence counterexample; not observed accuracy"}])


def main():
    missing = [str(p) for p in SRC.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing evidence inputs: " + ", ".join(missing))
    position_recovery()
    interventions()
    centered_shared_theta()
    temporal_prefix()
    compact_tables()
    outputs = sorted([*DATA.glob("*.csv"), *FIG.glob("*.pdf"), *FIG.glob("*.png")])
    manifest = {
        "status": "complete",
        "scope": "derived publication assets only; no model inference or experiment",
        "generator": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "generator_sha256": sha(Path(__file__)),
        "source_files": [{"key": k, "path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "bytes": p.stat().st_size} for k,p in SRC.items()],
        "outputs": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "bytes": p.stat().st_size} for p in outputs],
        "qualifications": ["three-seed means/SD use shared exposed full-validation examples",
                           "hard interventions are repeated within graph; intervals are graph-cluster bootstrap",
                           "centered profiles are label-free training diagnostics for one teacher; ADD is partial arithmetic only",
                           "temporal local decode equality is checked on 128 training graphs and does not imply equal global cost"]
    }
    (HERE / "source_sha_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":"complete", "csv_count":len(list(DATA.glob('*.csv'))),
                      "pdf_count":len(list(FIG.glob('*.pdf'))), "png_count":len(list(FIG.glob('*.png'))),
                      "manifest":str(HERE / 'source_sha_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
