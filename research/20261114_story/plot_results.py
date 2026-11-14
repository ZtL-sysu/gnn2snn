"""Create publication-style figures from the frozen calibration analysis CSVs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("MNIST", "CIFAR10")
TIMESTEPS = (1, 2, 4, 8)
METHODS = ("baseline", "node", "graph", "mix", "output")
METHOD_COLORS = {
    "baseline": "#6B7280",  # neutral gray
    "node": "#0072B2",      # blue
    "graph": "#D55E00",     # vermillion
    "mix": "#009E73",       # bluish green
    "output": "#CC79A7",    # reddish purple
}
SCOPE_COLORS = {"full_validation": "#0072B2", "screen_excluded": "#E69F00"}
SCOPE_MARKERS = {"full_validation": "o", "screen_excluded": "s"}
T95 = 4.30265273


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.3,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "lines.linewidth": 1.65,
    "lines.markersize": 4.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def fval(row, name):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {name}: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite field {name}: {row}")
    return value


def select_row(rows, **fields):
    matches = [row for row in rows if all(str(row.get(k, "")) == str(v) for k, v in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {fields}, found {len(matches)}")
    return matches[0]


def mean_sd_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("plots require exactly three finite seed-level values")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = T95 * sd / np.sqrt(3.0)
    return mean, sd, mean - half, mean + half


def validate_analysis(aggregate, per_seed):
    expected_seeds = {41, 42, 43}
    if any({int(r["seed"]) for r in per_seed if r.get("dataset") == d
            and r.get("scope") == s} != expected_seeds
           for d in DATASETS for s in ("full_validation", "screen_excluded")):
        raise ValueError("per_seed.csv must contain teacher seeds 41/42/43 for both scopes")
    seen = set()
    for row in per_seed:
        key = (row.get("dataset"), row.get("seed"), row.get("scope"),
               row.get("method"), row.get("T"))
        if key in seen:
            raise ValueError(f"duplicate per-seed analysis row: {key}")
        seen.add(key)
        expected_n = 5000 if row.get("scope") == "full_validation" else 4488
        if int(row.get("n_examples", -1)) != expected_n:
            raise ValueError(f"unexpected analysis subset size in {key}")
    for row in aggregate:
        expected_n = 5000 if row.get("scope") == "full_validation" else 4488
        if int(row.get("n_examples", -1)) != expected_n or int(row.get("n_seeds", -1)) != 3:
            raise ValueError(f"unexpected aggregate sample count: {row}")
    for dataset in DATASETS:
        for timestep in TIMESTEPS:
            for method in METHODS:
                select_row(aggregate, dataset=dataset, scope="full_validation", method=method, T=timestep)
                select_row(aggregate, dataset=dataset, scope="screen_excluded", method=method, T=timestep)
            select_row(aggregate, dataset=dataset, scope="full_validation",
                       method="paired_graph-minus-node", T=timestep)
            select_row(aggregate, dataset=dataset, scope="screen_excluded",
                       method="paired_graph-minus-node", T=timestep)
    for dataset in DATASETS:
        for scope in ("full_validation", "screen_excluded"):
            for method in METHODS:
                selected = [r for r in per_seed if r.get("dataset") == dataset
                            and r.get("scope") == scope and r.get("method") == method]
                if len(selected) != 12:
                    raise ValueError(f"expected 3 seeds × 4 T rows for {dataset}/{scope}/{method}")
            paired = [r for r in per_seed if r.get("dataset") == dataset
                      and r.get("scope") == scope and r.get("method") == "paired_graph-minus-node"]
            if len(paired) != 12:
                raise ValueError(f"missing per-seed graph-node pairs for {dataset}/{scope}")


def make_accuracy(aggregate, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    x = np.arange(len(TIMESTEPS))
    for ax, dataset in zip(axes, DATASETS):
        teacher = [select_row(aggregate, dataset=dataset, scope="full_validation",
                              method="baseline", T=t) for t in TIMESTEPS]
        teacher_mean = np.asarray([fval(r, "teacher_accuracy_percent_mean") for r in teacher])
        teacher_sd = np.asarray([fval(r, "teacher_accuracy_percent_sample_sd") for r in teacher])
        tmean = float(teacher_mean.mean())
        tsd = float(teacher_sd.mean())
        ax.axhspan(tmean - tsd, tmean + tsd, color="#9CA3AF", alpha=0.14, zorder=0)
        ax.axhline(tmean, color="#4B5563", linestyle=(0, (4, 2)), linewidth=1.25,
                   label="Teacher mean ± seed SD")
        for method in METHODS:
            rows = [select_row(aggregate, dataset=dataset, scope="full_validation",
                               method=method, T=t) for t in TIMESTEPS]
            means = np.asarray([fval(r, "accuracy_percent_mean") for r in rows])
            sds = np.asarray([fval(r, "accuracy_percent_sample_sd") for r in rows])
            ax.errorbar(x, means, yerr=sds, label=method.replace("_", " "),
                        color=METHOD_COLORS[method], marker={"baseline": "o", "node": "s", "graph": "^", "mix": "D", "output": "v"}[method],
                        capsize=2.3, elinewidth=1.05, linewidth=1.65)
        ax.set_title(dataset)
        ax.set_xticks(x, [f"T={t}" for t in TIMESTEPS])
        ax.set_xlabel("Time steps")
        ax.grid(axis="y")
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.25, wspace=0.12)
    save_figure(fig, out, "accuracy_curves")


def make_paired(aggregate, per_seed, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    x = np.arange(len(TIMESTEPS), dtype=float)
    for ax, dataset in zip(axes, DATASETS):
        ax.axhline(0, color="#4B5563", linewidth=1.1, linestyle=(0, (4, 2)))
        for scope, offset in (("full_validation", -0.075), ("screen_excluded", 0.075)):
            color = SCOPE_COLORS[scope]
            marker = SCOPE_MARKERS[scope]
            for ix, timestep in enumerate(TIMESTEPS):
                summary = select_row(aggregate, dataset=dataset, scope=scope,
                                    method="paired_graph-minus-node", T=timestep)
                mean = fval(summary, "accuracy_percent_mean")
                low = fval(summary, "accuracy_percent_t95_low")
                high = fval(summary, "accuracy_percent_t95_high")
                point_rows = [r for r in per_seed if r.get("dataset") == dataset
                              and r.get("scope") == scope
                              and r.get("method") == "paired_graph-minus-node"
                              and int(r.get("T", -1)) == timestep]
                if len(point_rows) != 3:
                    raise ValueError("paired plot needs three seed-level points per cell")
                point_rows.sort(key=lambda r: int(r["seed"]))
                jitter = np.linspace(-0.022, 0.022, len(point_rows))
                ax.scatter(np.full(3, x[ix] + offset) + jitter,
                           [fval(r, "accuracy_percent") for r in point_rows],
                           s=15, marker=marker, color=color, alpha=0.48, linewidths=0,
                           zorder=2)
                ax.errorbar(x[ix] + offset, mean,
                            yerr=[[max(0.0, mean - low)], [max(0.0, high - mean)]],
                            fmt=marker, color=color, markersize=5.2, capsize=3,
                            elinewidth=1.25, markeredgecolor="white", markeredgewidth=0.45,
                            zorder=3, label=scope.replace("_", " ") if ix == 0 else None)
        ax.set_title(dataset)
        ax.set_xticks(x, [f"T={t}" for t in TIMESTEPS])
        ax.set_xlabel("Time steps")
        ax.grid(axis="y")
    axes[0].set_ylabel("Graph − node accuracy (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.20, wspace=0.12)
    save_figure(fig, out, "paired_objective")


def load_costs(cost_dir):
    files = sorted(Path(cost_dir).rglob("cost_*.json"))
    if not files:
        raise FileNotFoundError(f"no cost_*.json files under {cost_dir}")
    runs = {}
    raw = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = data.get("dataset")
        seed = int(data.get("seed", -1))
        if dataset not in DATASETS or seed not in (41, 42, 43):
            raise ValueError(f"unexpected dataset/seed in cost file {path}")
        if data.get("scope") is None or data.get("hardware_energy") != "NOT MEASURED":
            raise ValueError(f"cost file missing diagnostic-only qualification: {path}")
        if data.get("repeats", 0) < 2:
            raise ValueError(f"insufficient timing repeats in {path}")
        key = (dataset, seed)
        if key in runs:
            raise ValueError(f"duplicate cost file for {key}")
        rows = {row["condition"]: row for row in data.get("latency", [])}
        expected = {"teacher"} | {f"{method}_T{t}" for method in ("baseline", "node", "graph")
                    for t in TIMESTEPS}
        if not expected.issubset(rows):
            raise ValueError(f"missing latency conditions in {path}")
        for condition in expected:
            row = rows[condition]
            raw_ms = row.get("raw_batch_ms")
            if not isinstance(raw_ms, list) or len(raw_ms) != int(data["repeats"]):
                raise ValueError(f"raw repeat values missing/malformed for {condition} in {path}")
            arr = np.asarray(raw_ms, dtype=np.float64)
            if not np.isfinite(arr).all() or np.any(arr <= 0):
                raise ValueError(f"invalid raw timing values for {condition} in {path}")
            if not math.isfinite(float(row.get("per_graph_ms_median", float("nan")))):
                raise ValueError(f"invalid per-graph median for {condition} in {path}")
        runs[key] = {"data": data, "rows": rows, "path": path}
        raw.append(path)
    for dataset in DATASETS:
        if {(d, s) for d, s in runs if d == dataset} != {(dataset, s) for s in (41, 42, 43)}:
            raise ValueError(f"latency plot requires all three seed runs for {dataset}")
    return runs, raw


def make_latency(cost_runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    x = np.arange(len(TIMESTEPS))
    for ax, dataset in zip(axes, DATASETS):
        seed_runs = [cost_runs[(dataset, seed)] for seed in (41, 42, 43)]
        teacher = [float(r["rows"]["teacher"]["per_graph_ms_median"]) for r in seed_runs]
        tm, tsd, _, _ = mean_sd_ci(teacher)
        ax.axhspan(tm - tsd, tm + tsd, color="#9CA3AF", alpha=0.14)
        ax.axhline(tm, color="#4B5563", linestyle=(0, (4, 2)), linewidth=1.25,
                   label="Teacher median ± seed SD")
        for method in ("baseline", "node", "graph"):
            medians = []
            for t in TIMESTEPS:
                values = [float(r["rows"][f"{method}_T{t}"]["per_graph_ms_median"])
                          for r in seed_runs]
                medians.append(values)
            means = np.asarray([np.mean(v) for v in medians])
            sds = np.asarray([np.std(v, ddof=1) for v in medians])
            ax.errorbar(x, means, yerr=sds, color=METHOD_COLORS[method],
                        marker={"baseline": "o", "node": "s", "graph": "^"}[method],
                        capsize=2.3, elinewidth=1.05, linewidth=1.65,
                        label=method.replace("_", " "))
        ax.set_title(dataset)
        ax.set_xticks(x, [f"T={t}" for t in TIMESTEPS])
        ax.set_xlabel("Time steps")
        ax.grid(axis="y")
    axes[0].set_ylabel("Diagnostic core latency per graph (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(bottom=0.24, wspace=0.12)
    save_figure(fig, out, "latency")


def save_figure(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def write_captions(path, has_costs):
    lines = [
        "# Figure captions",
        "",
        "**Accuracy curves.** Validation accuracy for the frozen baseline and calibration conditions across T=1, 2, 4, and 8 on SuperPixel MNIST and CIFAR10. Points show the mean across three teacher seeds and error bars show sample standard deviation across those seeds. The horizontal teacher reference and shaded band show the corresponding mean and sample standard deviation. These are historically exposed official validation examples; they are descriptive and do not represent an untouched test estimate.",
        "",
        "**Paired objective.** Within-seed graph-minus-node validation accuracy difference (percentage points) across T. Small points show individual teacher seeds; large markers show the paired mean and t-based 95% interval with df=2. Circle and square markers distinguish the full 5,000-example validation split from the 4,488-example subset excluding the 512 examples used in the original screen. Both subsets remain historically exposed validation data; the interval is descriptive under a seed-level independence assumption and is not a significance claim.",
    ]
    if has_costs:
        lines.extend(["", "**Diagnostic latency.** Median synchronized software latency per graph for the teacher, zero-offset baseline, node, and graph conditions across T. Points and error bars summarize the three per-seed medians using the mean and sample standard deviation. Underlying per-repeat timing measurements remain preserved in the input cost JSON files and are listed with SHA-256 hashes in the figure data manifest. The core returns logits and six decoded site activations, so these timings describe the current diagnostic implementation; they are not optimized deployment latency, energy measurements, or hardware cost claims."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--costs", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True,
                        help="new figure directory; existing output is never overwritten")
    args = parser.parse_args()
    analysis_dir = args.analysis.resolve()
    aggregate_path = analysis_dir / "aggregate.csv"
    per_seed_path = analysis_dir / "per_seed.csv"
    if not aggregate_path.is_file() or not per_seed_path.is_file():
        raise FileNotFoundError("analysis directory must contain aggregate.csv and per_seed.csv")
    aggregate = read_csv(aggregate_path)
    per_seed = read_csv(per_seed_path)
    validate_analysis(aggregate, per_seed)
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    cost_runs = None
    cost_files = []
    if args.costs is not None:
        cost_runs, cost_files = load_costs(args.costs.resolve())
    out.mkdir(parents=True, exist_ok=False)
    make_accuracy(aggregate, out)
    make_paired(aggregate, per_seed, out)
    if cost_runs is not None:
        make_latency(cost_runs, out)
    write_captions(out / "captions.md", cost_runs is not None)
    manifest = {
        "analysis_directory": str(analysis_dir),
        "input_files": {
            str(aggregate_path): sha256(aggregate_path),
            str(per_seed_path): sha256(per_seed_path),
        },
        "cost_inputs": [{"path": str(path), "sha256": sha256(path)} for path in cost_files],
        "runner_versions": sorted({r.get("runner_script_sha256", "") for r in per_seed if r.get("runner_script_sha256")}),
        "figure_files": sorted(p.name for p in out.glob("*.pdf")) + sorted(p.name for p in out.glob("*.png")),
        "plot_qualification": "historically exposed validation, descriptive teacher-seed variability; diagnostic latency only",
        "raw_latency_repeats_preserved_in_source_json": bool(cost_files),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    (out / "figure_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
