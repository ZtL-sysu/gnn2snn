#!/usr/bin/env python3
"""B-conference strengthening suite for D1 (Spiking-GNN conversion).

Adds the pieces a reviewer will ask for beyond the 3 homophilic Planetoid graphs:

  1. `extended`  - run the conversion on ogbn-arxiv (scale) and Chameleon/Squirrel
                   (heterophily), i.e. datasets where neighbour averaging is NOT trivially
                   easy, so the "neighborhood temporal misalignment" claim is really tested.
  2. `direct`    - DIRECTLY-TRAINED spiking GCN competitor (LIF + surrogate-gradient BPTT).
                   This is the natural baseline: same architecture, same T, trained rather
                   than converted. Note it must be re-trained for every T, while the
                   conversion trains ONCE and deploys at any T.
  3. `ablate`    - ablations: readout (membrane vs count), bias (none/scalar-FTBC/
                   per-channel-FTBC), neuron model (IF vs TPP), theta, hidden width.
  4. `plot`      - accuracy-vs-T curves and energy/latency figures.

Run:
  python strengthen.py --stage extended --out ../results/strengthen_extended.json
  python strengthen.py --stage direct  --out ../results/strengthen_direct.json
  python strengthen.py --stage ablate  --out ../results/strengthen_ablate.json
  python strengthen.py --stage plot
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import snn_gnn_experiment as E
import graph_data as G


# ---------------------------------------------------------------- common
def prepare(name, data_root):
    d = G.load_ds(name, data_root)
    adj_norm = E.normalize_adj(d["adj"])
    return d, adj_norm


def train_models(d, adj_norm, seeds, hidden, row_normalize=True):
    store = []
    for sd in seeds:
        m = E.train_gcn(d["features"], d["labels"], adj_norm,
                        d["train_mask"], d["val_mask"], d["test_mask"], sd,
                        hidden=hidden, row_normalize=row_normalize)
        m["seed"] = sd
        m["labels"] = d["labels"]; m["test_mask"] = d["test_mask"]
        m["val_mask"] = d["val_mask"]; m["train_mask"] = d["train_mask"]
        store.append(m)
    return store


# ---------------------------------------------------------------- 1. extended datasets
# Wider networks for the larger/richer graphs: hidden=16 is the standard Kipf setting for
# the small Planetoid graphs, but it makes ogbn-arxiv / heterophilic baselines needlessly
# weak, which would invite the criticism "the method never reaches usable accuracy".
HIDDEN = {"ogbn-arxiv": 128, "chameleon": 64, "squirrel": 64}


def run_extended(names, seeds, ts, data_root, hidden=16):
    out = {}
    for name in names:
        h = HIDDEN.get(name, hidden)
        try:
            d, adj_norm = prepare(name, data_root)
        except Exception as e:
            out[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f"[{name}] LOAD FAILED: {e}", flush=True)
            continue
        hom = G.edge_homophily(d["adj"], d["labels"])
        store = train_models(d, adj_norm, seeds, h, d["row_normalize"])
        ann = float(np.mean([m["acc_test"] for m in store]))
        res = dict(ann_test_mean=ann, N=int(d["features"].shape[0]), hidden=h,
                   C=int(d["labels"].max()) + 1, homophily=float(hom), results={})
        for mode in ("naive", "ftbc", "oracle"):
            res["results"][mode] = {}
            for T in ts:
                accs, cnts, spk = [], [], []
                for m in store:
                    beta1 = E.tune_bias(m["B1"], T, mask=m["train_mask"]) \
                        if mode in ("ftbc", "oracle") else 0.0
                    conv_mode = "oracle" if mode == "oracle" else "naive"
                    r = E.convert(m, T=T, beta1=beta1, mode=conv_mode,
                                  l1_mode="if", seed=0, theta=1.0, readout="membrane")
                    accs.append(r["acc_test"]); cnts.append(r["acc_count"]); spk.append(r["total_spikes"])
                res["results"][mode][str(T)] = dict(
                    acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                    acc_count_mean=float(np.mean(cnts)), spike_mean=float(np.mean(spk)))
        out[name] = res
        print(f"[{name}] N={res['N']} ANN={ann:.4f} homophily={hom:.3f} "
              f"ftbc: " + " ".join(f"T{t}={res['results']['ftbc'][str(t)]['acc_mean']:.3f}" for t in ts),
              flush=True)
    return out


# ---------------------------------------------------------------- 2. direct training
class SpikeFn(torch.autograd.Function):
    """Heaviside forward with triangular surrogate gradient (|u-theta| < width)."""
    @staticmethod
    def forward(ctx, u, theta, width):
        ctx.save_for_backward(u)
        ctx.theta = theta; ctx.width = width
        return (u >= theta).float()

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        d = torch.abs(u - ctx.theta) / ctx.width
        grad_u = grad_out * torch.clamp(1.0 - d, 0.0, 1.0) / ctx.width
        return grad_u, None, None


def train_direct(adj_t, X, labels, tr_idx, va_idx, hidden, C, T, seed, epochs=400,
                 lr=0.01, wd=5e-4, early_stop=30, theta=1.0, width=1.0, readout="membrane",
                 dropout=0.5):
    """End-to-end trained 2-layer spiking GCN (LIF, reset-by-subtraction, surrogate BPTT).

    Layer 1 emits binary spikes each step; layer 2 integrates the (instantaneous) messages
    A(s1_t)W2 + b2. Decision = accumulated layer-2 current (mirrors our membrane readout),
    or accumulated output spikes if readout='count'.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    N, F = X.shape
    def glorot(shape):
        std = 1.0 / math.sqrt(shape[0])
        return torch.FloatTensor(*shape).uniform_(-std, std)
    W1 = torch.nn.Parameter(glorot((F, hidden)))
    W2 = torch.nn.Parameter(glorot((hidden, C)))
    b1 = torch.nn.Parameter(torch.zeros(hidden))
    b2 = torch.nn.Parameter(torch.zeros(C))
    opt = torch.optim.Adam([W1, W2, b1, b2], lr=lr, weight_decay=wd)
    Y = torch.tensor(labels, dtype=torch.long)

    def forward(train=True):
        pre1 = torch.sparse.mm(adj_t, X) @ W1 + b1          # constant input current
        if train and dropout > 0:
            pre1 = torch.dropout(pre1, dropout, train=True)
        u1 = torch.zeros(N, hidden)
        acc2 = torch.zeros(N, C)
        cnt2 = torch.zeros(N, C)
        u2 = torch.zeros(N, C)
        for t in range(T):
            u1 = u1 + pre1
            s1 = SpikeFn.apply(u1, theta, width)
            u1 = u1 - s1 * theta
            cur2 = torch.sparse.mm(adj_t, s1) @ W2 + b2
            u2 = u2 + cur2
            s2 = SpikeFn.apply(u2, theta, width)
            u2 = u2 - s2 * theta
            cnt2 = cnt2 + s2
            acc2 = acc2 + cur2
        return (acc2 if readout == "membrane" else cnt2), cnt2.mean()

    best_val, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        opt.zero_grad()
        logits, _ = forward(True)
        loss = torch.nn.functional.cross_entropy(logits[tr_idx], Y[tr_idx])
        loss.backward(); opt.step()
        with torch.no_grad():
            lg, _ = forward(False)
            v = float((lg[va_idx].argmax(1) == Y[va_idx]).float().mean())
            if v > best_val:
                best_val, patience = v, 0
                best_state = (W1.detach().clone(), W2.detach().clone(),
                              b1.detach().clone(), b2.detach().clone())
            else:
                patience += 1
            if patience >= early_stop:
                break
    W1_, W2_, b1_, b2_ = best_state
    with torch.no_grad():
        W1.data.copy_(W1_); W2.data.copy_(W2_); b1.data.copy_(b1_); b2.data.copy_(b2_)
        lg, _ = forward(False)
    return (W1, W2, b1, b2), lg, best_val


def run_direct(names, seeds, ts, data_root, hidden=16):
    out = {}
    for name in names:
        try:
            d, adj_norm = prepare(name, data_root)
        except Exception as e:
            out[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f"[direct {name}] LOAD FAILED: {e}", flush=True)
            continue
        # L1 row-normalise inside the model here too (same preprocessing as the ANN side)
        import scipy.sparse as sp
        coo = adj_norm.tocoo()
        idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        vals = torch.tensor(coo.data, dtype=torch.float32)
        adj_t = torch.sparse_coo_tensor(idx, vals, (adj_norm.shape[0], adj_norm.shape[1])).coalesce()
        X_np = d["features"].astype(np.float32).copy()
        if d["row_normalize"]:
            rs = X_np.sum(1); rs[rs == 0] = 1.0
            X_np = X_np / rs[:, None]
        X = torch.tensor(X_np)
        tr = torch.tensor(np.where(d["train_mask"])[0], dtype=torch.long)
        va = torch.tensor(np.where(d["val_mask"])[0], dtype=torch.long)
        te = np.where(d["test_mask"])[0]
        C = int(d["labels"].max()) + 1
        res = dict(N=int(X_np.shape[0]), results={})
        for T in ts:
            accs = []
            for sd in seeds:
                _, lg, best_val = train_direct(adj_t, X, d["labels"], tr, va, hidden, C, T, sd)
                pred = lg.argmax(1).numpy()
                accs.append(float((pred[te] == d["labels"][te]).mean()))
            res["results"][str(T)] = dict(acc_mean=float(np.mean(accs)),
                                          acc_std=float(np.std(accs)))
        out[name] = res
        print(f"[direct {name}] " + " ".join(f"T{t}={res['results'][str(t)]['acc_mean']:.3f}" for t in ts),
              flush=True)
    return out


# ---------------------------------------------------------------- 3. ablations
def run_ablate(names, seeds, ts, data_root, hidden=16):
    out = {}
    for name in names:
        d, adj_norm = prepare(name, data_root)
        store = train_models(d, adj_norm, seeds, hidden, d["row_normalize"])
        ann = float(np.mean([m["acc_test"] for m in store]))
        res = dict(ann_test_mean=ann, ablations={})

        def _eval(label, make_beta, l1_mode="if", theta=1.0, readout="membrane"):
            per_T = {}
            for T in ts:
                accs = []
                for m in store:
                    beta1 = make_beta(m, T)
                    r = E.convert(m, T=T, beta1=beta1, mode="naive", l1_mode=l1_mode,
                                  seed=0, theta=theta, readout=readout)
                    accs.append(r["acc_test"])
                per_T[str(T)] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)))
            res["ablations"][label] = per_T

        # (a) readout: membrane vs integer count
        _eval("readout_membrane", lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"]),
              readout="membrane")
        _eval("readout_count", lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"]),
              readout="count")
        # (b) bias: none / scalar FTBC / per-channel FTBC
        _eval("bias_none", lambda m, T: 0.0)
        _eval("bias_ftbc_scalar", lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"]))
        _eval("bias_ftbc_perchannel",
              lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"], per_channel=True))
        # (c) neuron model: IF vs probabilistic TPP
        _eval("neuron_if", lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"]), l1_mode="if")
        _eval("neuron_tpp", lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"]), l1_mode="tpp")
        # (d) threshold sensitivity
        for th in (0.5, 1.0, 2.0):
            _eval(f"theta_{th}", lambda m, T: E.tune_bias(m["B1"], T, mask=m["train_mask"]), theta=th)
        out[name] = res
        print(f"[ablate {name}] ANN={ann:.4f}; T8 =" +
              " ".join(f"{k}={v['8']['acc_mean']:.3f}" for k, v in res["ablations"].items()),
              flush=True)
    return out


# ---------------------------------------------------------------- 4. plots
def make_plots(data_root, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)
    ts = [1, 2, 4, 8, 16]
    made = []
    for name in ("cora", "citeseer", "pubmed"):
        p = Path(data_root) / f"{name}_fixed.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
        for mode, style in (("naive", "o-"), ("ftbc", "s-"), ("oracle", "k--")):
            if mode not in d["results"]:
                continue
            acc = [d["results"][mode][str(t)]["acc_mean"] for t in ts]
            spk = [d["results"][mode][str(t)]["spike_mean"] for t in ts]
            ax[0].plot(ts, acc, style, label=mode)
            ax[1].plot(ts, spk, style, label=mode)
        ax[0].axhline(d["ann_test_mean"], color="gray", ls=":", label="ANN")
        ax[0].set_xscale("log", base=2); ax[0].set_xticks(ts); ax[0].set_xticklabels(ts)
        ax[0].set_xlabel("timesteps T"); ax[0].set_ylabel("test accuracy")
        ax[0].set_title(f"{name}: accuracy vs T"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].set_xscale("log", base=2); ax[1].set_xticks(ts); ax[1].set_xticklabels(ts)
        ax[1].set_xlabel("timesteps T"); ax[1].set_ylabel("total spikes (energy proxy)")
        ax[1].set_title(f"{name}: spikes vs T"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        f = outdir / f"{name}_accuracy_energy.png"
        fig.savefig(f, dpi=150); plt.close(fig); made.append(str(f))
    return made


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "extended", "direct", "ablate", "plot"])
    # resolve relative to THIS file so the suite works from any cwd (a "../results/.data"
    # default silently created duplicate dataset caches outside the run root)
    ap.add_argument("--data-root",
                    default=str(Path(__file__).resolve().parent.parent / "results" / ".data"))
    ap.add_argument("--out", default="../results/strengthen.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--figures", default="../figures")
    ap.add_argument("--names", nargs="*", default=None,
                    help="override the dataset list used by the 'extended' stage")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    data = {}

    if a.stage in ("extended", "all"):
        data["extended"] = run_extended(a.names or ["ogbn-arxiv", "chameleon", "squirrel"],
                                        seeds, a.ts,
                                        a.data_root, a.hidden)
    if a.stage in ("direct", "all"):
        data["direct"] = run_direct(["cora", "citeseer", "pubmed"], seeds, a.ts,
                                    a.data_root, a.hidden)
    if a.stage in ("ablate", "all"):
        data["ablate"] = run_ablate(["cora", "citeseer", "pubmed"], seeds, a.ts,
                                    a.data_root, a.hidden)
    if a.stage in ("plot", "all"):
        figs = make_plots(Path(__file__).resolve().parent.parent / "results", a.figures)
        data["figures"] = figs
        print("figures:", figs, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
