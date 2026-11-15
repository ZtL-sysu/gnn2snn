#!/usr/bin/env python3
"""Merge every piece of D1 evidence into one machine-readable bundle.

Input : results/{cora,citeseer,pubmed}_fixed.json   (main run, membrane readout)
        results/strengthen_ablate.json              (readout / bias / neuron / theta)
        results/strengthen_direct.json              (directly-trained spiking GCN)
        results/strengthen_extended_hetero.json     (chameleon, squirrel)
        results/strengthen_extended_arxiv.json      (ogbn-arxiv, may be absent yet)
Output: results/strengthen_summary.json  +  a printed markdown-ready report
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
TS = [1, 2, 4, 8, 16]


def load(name, unwrap=None):
    """Stage runs wrap their payload under the stage key ({"ablate": {...}}); unwrap it."""
    p = ROOT / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if unwrap is not None and unwrap in d:
        d = d[unwrap]
    return d


def main():
    out = {"main": {}, "ablate": load("strengthen_ablate.json", "ablate"),
           "direct": load("strengthen_direct.json", "direct"), "extended": {}}
    for ds in ("cora", "citeseer", "pubmed"):
        d = load(f"{ds}_analytic.json") or load(f"{ds}_fixed.json")
        if d:
            out["main"][ds] = d
    for f in ("strengthen_extended_hetero.json", "strengthen_extended_arxiv.json"):
        d = load(f)
        if d:
            out["extended"].update(d)

    # ---------------- print report ----------------
    print("=" * 88)
    print("D1 STRENGTHENING SUMMARY")
    print("=" * 88)

    print("\n[A] Main conversion (membrane readout, 5 seeds, hidden=16)")
    print(f"{'dataset':10s} {'ANN':>7s} {'ftbc T4':>8s} {'ftbc T8':>8s} {'ftbc T16':>9s} "
          f"{'oracle':>7s} {'strong(T<=8)':>13s}")
    for name, d in out["main"].items():
        ann = d["ann_test_mean"]; thr = ann - 0.015
        ft = d["results"]["ftbc"]
        t8 = ft["8"]["acc_mean"]
        print(f"{name:10s} {ann:7.4f} {ft['4']['acc_mean']:8.4f} {t8:8.4f} "
              f"{ft['16']['acc_mean']:9.4f} {d['results']['oracle']['8']['acc_mean']:7.4f} "
              f"{('PASS' if t8 >= thr else 'fail'):>13s}")

    if out["direct"]:
        print("\n[B] vs DIRECTLY-TRAINED spiking GCN (same 2-layer GCN, same T, retrained per T)")
        print(f"{'dataset':10s} {'ANN':>7s} | " + " ".join(f"{'T'+str(t):>8s}" for t in TS))
        for name, d in out["main"].items():
            conv = [d["results"]["ftbc"][str(t)]["acc_mean"] for t in TS]
            dr = out["direct"].get(name, {}).get("results", {})
            dirv = [dr.get(str(t), {}).get("acc_mean", float("nan")) for t in TS]
            print(f"{name:10s} {d['ann_test_mean']:7.4f} conv " + " ".join(f"{v:8.4f}" for v in conv))
            print(f"{'':10s} {'':7s} dir  " + " ".join(f"{v:8.4f}" for v in dirv))
            print(f"{'':10s} {'':7s} gain " + " ".join(f"{a-b:+8.4f}" for a, b in zip(conv, dirv)))

    if out["ablate"]:
        print("\n[C] Ablations at T=8 (membrane readout unless stated)")
        for name, d in out["ablate"].items():
            print(f"  {name} (ANN {d['ann_test_mean']:.4f}):")
            for lab, per in d["ablations"].items():
                print(f"     {lab:22s} T8={per['8']['acc_mean']:.4f}")

    if out["extended"]:
        print("\n[D] Extended datasets (scale + heterophily)")
        print(f"{'dataset':12s} {'N':>7s} {'hom':>6s} {'ANN':>7s} {'ftbc T4':>8s} "
              f"{'ftbc T8':>8s} {'ftbc T16':>9s} {'oracle':>7s}")
        for name, d in out["extended"].items():
            if "error" in d:
                print(f"{name:12s} ERROR {d['error'][:60]}")
                continue
            ft = d["results"]["ftbc"]
            print(f"{name:12s} {d['N']:7d} {d['homophily']:6.3f} {d['ann_test_mean']:7.4f} "
                  f"{ft['4']['acc_mean']:8.4f} {ft['8']['acc_mean']:8.4f} "
                  f"{ft['16']['acc_mean']:9.4f} {d['results']['oracle']['8']['acc_mean']:7.4f}")

    (ROOT / "strengthen_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'strengthen_summary.json'}")


if __name__ == "__main__":
    main()
