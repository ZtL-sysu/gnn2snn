from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def read(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def row_for(name: str, d: dict, dataset: str, record_level: str, metadata: dict | None = None):
    ann = d.get("ann_test_mean")
    ann_record = "aggregate_only"
    if d.get("ann_per_seed"):
        vals = [float(x["acc_test"]) for x in d["ann_per_seed"]]
        ann = sum(vals) / len(vals)
        ann_record = f"recomputed_from_{len(vals)}_ANN_seed_records"
    results = d["results"]
    out = {"dataset": dataset, "N": (metadata or {}).get("N", ""), "hidden": d.get("hidden", (metadata or {}).get("hidden", "")),
           "ANN_test_mean": ann, "ANN_record_level": ann_record}
    for t in ("8", "16"):
        f = results["ftbc"][t]
        out[f"FTBC_membrane_T{t}"] = f.get("acc_mean", f.get("mean", ""))
        out[f"FTBC_count_T{t}"] = f.get("acc_count_mean", f.get("count_mean", ""))
    out["SNN_record_level"] = record_level
    out["source_file"] = name
    return out

rows = []
for dataset in ("cora", "citeseer", "pubmed"):
    name = f"results__{dataset}_analytic.json"
    d = read(name)
    rows.append(row_for(name, d, dataset, "aggregate_means_only_for_SNN; ANN records retained"))
for name, file, dataset_names in [
    ("results__strengthen_extended_hetero.json", "results__strengthen_extended_hetero.json", ("chameleon", "squirrel")),
    ("results__strengthen_extended_arxiv.json", "results__strengthen_extended_arxiv.json", ("ogbn-arxiv",)),
]:
    d = read(file)
    for dataset in dataset_names:
        rows.append(row_for(name, d[dataset], dataset, "aggregate_means_only_for_ANN_and_SNN", d[dataset]))

out_path = ROOT / "six_dataset_summary.csv"
with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        clean = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
        w.writerow(clean)

manifest = []
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and p.name != "sha256_manifest.json":
        manifest.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT / "sha256_manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_path} ({len(rows)} rows); manifest has {len(manifest)} files")
for r in rows:
    print(r["dataset"], r["ANN_test_mean"], r["FTBC_membrane_T8"], r["FTBC_membrane_T16"], r["SNN_record_level"])
