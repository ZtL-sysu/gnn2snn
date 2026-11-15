#!/usr/bin/env python3
"""M2 repair: ogbn-arxiv with a COMPETITIVE ANN baseline.

The reviewer objection is fair: at hidden=64 the GCN reaches only 0.6595 test accuracy, so a
skeptic can dismiss the scale experiment as "converting a weak model". Standard ogbn-arxiv GCN
numbers are ~0.715, obtained with hidden=256, no feature row-normalisation (the TF-IDF features
are already normalised) and ~500 epochs of Adam.

This script re-runs the SAME conversion pipeline on a stronger ANN so the scale row is a fair
fidelity test. Key deviation from `train_gcn`: A_hat @ X is constant across epochs and is
pre-computed once (a 169k x 128 x 256 dense product per epoch otherwise dominates runtime).

Run:
  python arxiv_strong.py --seeds 7 --hidden 256 --epochs 400 --out ../results/arxiv_strong_s7.json
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import snn_gnn_experiment as E
import graph_data as G

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def train_gcn_fast(adj_norm, X_np, labels, tr, va, te, seed, hidden=256,
                   lr=0.01, weight_decay=5e-4, epochs=400, dropout=0.5, early_stop=40):
    torch.manual_seed(seed); np.random.seed(seed)
    N, F = X_np.shape
    C = int(labels.max()) + 1
    Y = torch.tensor(labels, dtype=torch.long)
    coo = adj_norm.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    vals = torch.tensor(coo.data, dtype=torch.float32)
    adj_t = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    AX = torch.sparse.mm(adj_t, X_t)                 # constant: precompute once

    def glorot(shape):
        std = 1.0 / math.sqrt(shape[0])
        return torch.FloatTensor(*shape).uniform_(-std, std)

    W1 = torch.nn.Parameter(glorot((F, hidden)))
    W2 = torch.nn.Parameter(glorot((hidden, C)))
    b1 = torch.nn.Parameter(torch.zeros(hidden))
    b2 = torch.nn.Parameter(torch.zeros(C))
    opt = torch.optim.Adam([W1, W2, b1, b2], lr=lr, weight_decay=weight_decay)
    idx_tr = torch.where(torch.tensor(tr))[0]
    idx_va = torch.where(torch.tensor(va))[0]

    best_val, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        opt.zero_grad()
        H1 = torch.relu(AX @ W1 + b1)
        if dropout > 0:
            H1 = torch.dropout(H1, dropout, train=True)
        Z = torch.sparse.mm(adj_t, H1) @ W2 + b2
        loss = torch.nn.functional.cross_entropy(Z[idx_tr], Y[idx_tr])
        loss.backward(); opt.step()
        with torch.no_grad():
            v = float((Z[idx_va].argmax(1) == Y[idx_va]).float().mean())
        if v > best_val:
            best_val, patience = v, 0
            best_state = (W1.detach().clone(), W2.detach().clone(),
                          b1.detach().clone(), b2.detach().clone())
        else:
            patience += 1
            if patience >= early_stop:
                break
    W1, W2, b1, b2 = best_state
    with torch.no_grad():
        H1 = torch.relu(AX @ W1 + b1)
        Z = torch.sparse.mm(adj_t, H1) @ W2 + b2
        pred = Z.argmax(1).numpy()
        acc_test = float((pred[te] == labels[te]).mean())
        B1 = np.asarray(AX.numpy() @ W1.numpy() + b1.numpy(), dtype=np.float32)
        B2 = np.asarray(adj_norm @ (np.maximum(B1, 0) @ W2.numpy()) + b2.numpy(), dtype=np.float32)
    return dict(W1=W1.numpy(), W2=W2.numpy(), b1=b1.numpy(), b2=b2.numpy(),
                acc_test=acc_test, B1=B1, B2=B2, X=X_np, adj_norm=adj_norm,
                labels=labels, test_mask=te, val_mask=va, train_mask=tr,
                epochs_run=ep + 1, best_val=best_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ogbn-arxiv")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--data-root", default=str(ROOT / "results" / ".data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "arxiv_strong_s7.json"))
    a = ap.parse_args()

    d = G.load_ds(a.name, a.data_root)
    adj_norm = E.normalize_adj(d["adj"])
    X_np = d["features"].astype(np.float32).copy()
    if d["row_normalize"]:
        rs = X_np.sum(1); rs[rs == 0] = 1.0
        X_np = X_np / rs[:, None]
    tr, va, te = d["train_mask"], d["val_mask"], d["test_mask"]
    C = int(d["labels"].max()) + 1

    store, ann = [], []
    for sd in range(a.seeds):
        m = train_gcn_fast(adj_norm, X_np, d["labels"], tr, va, te, sd,
                           hidden=a.hidden, epochs=a.epochs)
        ann.append(m["acc_test"]); store.append(m)
        print(f"  seed {sd}: ANN={m['acc_test']:.4f} (ep={m['epochs_run']}, val={m['best_val']:.4f})",
              flush=True)
    ann_mean = float(np.mean(ann))
    print(f"[{a.name}] ANN ({a.seeds} seeds, hidden={a.hidden}) = {ann_mean:.4f} "
          f"+- {np.std(ann):.4f}", flush=True)

    res = dict(dataset=a.name, N=int(X_np.shape[0]), hidden=a.hidden, C=C,
               epochs_budget=a.epochs, ann_test_mean=ann_mean,
               ann_per_seed=[float(x) for x in ann],
               homophily=float(G.edge_homophily(d["adj"], d["labels"])), results={})
    for mode in ("naive", "ftbc", "oracle"):
        res["results"][mode] = {}
        for T in a.ts:
            accs, cnts, spk = [], [], []
            for m in store:
                beta1 = E.ftbc_bias(T) if mode in ("ftbc", "oracle") else 0.0
                r = E.convert(m, T=T, beta1=beta1,
                              mode=("oracle" if mode == "oracle" else "naive"),
                              l1_mode="if", seed=0, theta=1.0, readout="membrane")
                accs.append(r["acc_test"]); cnts.append(r["acc_count"])
                spk.append(r["total_spikes"])
            res["results"][mode][str(T)] = dict(
                acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                acc_count_mean=float(np.mean(cnts)), spike_mean=float(np.mean(spk)),
                acc_per_seed=[float(x) for x in accs])
        print(f"[{a.name}] {mode}: " + " ".join(
            f"T{t}={res['results'][mode][str(t)]['acc_mean']:.4f}" for t in a.ts), flush=True)

    Path(a.out).write_text(json.dumps({a.name: res}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
