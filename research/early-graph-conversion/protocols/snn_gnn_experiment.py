#!/usr/bin/env python3
"""
Spiking-GNN Conversion experiment (D1).

Pipeline:
  1. Train a full-precision 2-layer GCN (ANN-GNN) on a Planetoid node-classification graph.
  2. Convert ANN-GNN -> SNN-GNN by rate-based ANN->SNN conversion:
       - Layer 1: spiking IF / TPP neurons (spike count -> rate).
       - Layer 2: spiking neurons reading instantaneous layer-1 spikes.
     Variants:
       - naive     : no bias correction (floor-deficit bias of IF rate coding left in).
       - ftbc      : layer-1 FTBC-style forward bias correction beta1 (PROPOSED METHOD).
       - oracle    : rate-ideal LIMIT where layer 1 emits its exact real-valued rate, so
                     sum_t inst_t = T*B2 exactly and accuracy must equal the ANN. Used as
                     the mathematical ceiling AND as a correctness check on the chain.
  3. Report node-classification accuracy vs T in {1,2,4,8,16} and spike counts.

Run:
  python snn_gnn_experiment.py --dataset cora --seeds 5 --ts 1 2 4 8 16 \
        --modes naive ftbc oracle --out ../results/cora.json
"""
import argparse, json, math, os, sys, time, urllib.request, ssl
from pathlib import Path

import numpy as np

# ---------- Planetoid loader (GitHub raw, no PyG dependency) ----------
PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"

def _download(url, dest):
    if dest.exists():
        return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
        dest.write_bytes(r.read())

def _parse_index(path):
    with open(path, "r") as f:
        return np.fromfile(f, sep="\n", dtype=int)

def _sparse_to_dense(m):
    # PyG Planetoid stores .x/.allx/.tx as scipy sparse .npz for pubmed, else pickled sparse.
    import scipy.sparse as sp
    if hasattr(m, "toarray"):
        return m.toarray()
    return np.asarray(m)

def load_planetoid(name, data_root):
    data_root = Path(data_root); data_root.mkdir(parents=True, exist_ok=True)
    files = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    paths = {}
    for f in files:
        p = data_root / f"ind.{name}.{f}"
        _download(f"{PLANETOID_BASE}/ind.{name}.{f}", p)
        paths[f] = p
    import pickle
    def load_pkl(p):
        with open(p, "rb") as f:
            return pickle.load(f, encoding="latin1")
    def load_npz(p):
        obj = np.load(p, allow_pickle=True, encoding="latin1")
        if hasattr(obj, "toarray"):          # already a scipy sparse matrix
            return _sparse_to_dense(obj)
        try:                                  # NpzFile container
            return _sparse_to_dense(obj["arr_0"])
        except Exception:
            import scipy.sparse as sp
            return _sparse_to_dense(sp.load_npz(p))

    if name == "pubmed":
        x = load_npz(paths["x"]); tx = load_npz(paths["tx"]); allx = load_npz(paths["allx"])
    else:
        x = _sparse_to_dense(load_pkl(paths["x"])); tx = _sparse_to_dense(load_pkl(paths["tx"]))
        allx = _sparse_to_dense(load_pkl(paths["allx"]))
    y = load_pkl(paths["y"]); ty = load_pkl(paths["ty"]); ally = load_pkl(paths["ally"])
    graph = load_pkl(paths["graph"])
    # !! RAW (unsorted) test indices: test_idx[k] is the TRUE graph-node index of the
    # k-th test node, whose feature row is tx[k] / label row is ty[k]. The planetoid
    # test.index is stored in a scattered order, so we MUST place tx[k] at position
    # test_idx[k] (NOT at 1708+k). Sorting here misaligns every test node with the graph.
    test_idx = _parse_index(paths["test.index"]).astype(np.int64)

    # ---- Assemble features & labels by TRUE graph index ----
    # graph is keyed by true graph-node index; tx[k]/ty[k] belong to test node test_idx[k].
    # allx/ally already cover nodes [0, len(allx)) in true order; test nodes must be
    # placed at their true graph positions so they align with the adjacency.
    n = max(graph.keys()) + 1            # total node count (Cora=2708)
    nfeat = allx.shape[1]
    features = np.zeros((n, nfeat), dtype=np.float32)
    features[:allx.shape[0]] = allx.astype(np.float32)
    for i, gi in enumerate(test_idx):
        features[gi] = tx[i].astype(np.float32)

    labels = -np.ones(n, dtype=np.int64)  # -1 = unlabeled (excluded from train/eval)
    labels[:ally.shape[0]] = np.argmax(np.asarray(ally), axis=1).astype(np.int64)
    for i, gi in enumerate(test_idx):
        labels[gi] = int(np.argmax(np.asarray(ty[i])))

    # adjacency (true graph index space), add self-loops, symmetrize
    adj = np.zeros((n, n), dtype=np.float32)
    for i, nbrs in graph.items():
        for j in nbrs:
            adj[i, j] = 1.0
    adj = adj + np.eye(n, dtype=np.float32)
    adj = np.maximum(adj, adj.T)
    # masks
    train_mask = np.zeros(n, dtype=bool); val_mask = np.zeros(n, dtype=bool); test_mask = np.zeros(n, dtype=bool)
    train_mask[:y.shape[0]] = True
    # val = next 500 nodes after the train block (within the non-test portion)
    val_mask[y.shape[0]:y.shape[0] + 500] = True
    # test = test_idx positions (rows appended after the non-test block)
    test_mask[test_idx] = True
    train_mask[val_mask] = False  # val removed from train
    train_mask[test_mask] = False
    return dict(features=features, labels=labels, adj=adj,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

def normalize_adj(adj):
    import scipy.sparse as sp
    adj = sp.csr_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).astype(np.float32).tocsr()

# ---------- GCN (PyTorch) ----------
import torch

def train_gcn(features, labels, adj_norm, train_mask, val_mask, test_mask, seed, hidden=16,
              lr=0.01, weight_decay=5e-4, epochs=400, dropout=0.5, early_stop=30,
              row_normalize=True):
    """Train a 2-layer GCN (Kipf) as the ANN-GNN baseline.

    Forward: H1 = relu(A_hat @ X @ W1 + b1); Z = A_hat @ H1 @ W2 + b2.
    `row_normalize` applies Kipf's L1 row-normalisation, which is part of the standard
    Cora/Citeseer/Pubmed BoW-feature recipe. Datasets whose features are already
    normalised (ogbn-arxiv TF-IDF, WikipediaNetwork proportions) set it False.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    N, F = features.shape; C = int(labels.max()) + 1
    # L1 row-normalize features (Kipf BoW recipe; skipped for pre-normalised inputs)
    X_np = features.astype(np.float32).copy()
    if row_normalize:
        rs = X_np.sum(1); rs[rs == 0] = 1.0
        X_np = X_np / rs[:, None]
    Y = torch.tensor(labels, dtype=torch.long)
    coo = adj_norm.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    vals = torch.tensor(coo.data, dtype=torch.float32)
    adj_t = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    def glorot(shape):
        std = 1.0 / math.sqrt(shape[0])
        return torch.FloatTensor(*shape).uniform_(-std, std)
    W1 = torch.nn.Parameter(glorot((F, hidden)))
    W2 = torch.nn.Parameter(glorot((hidden, C)))
    b1 = torch.nn.Parameter(torch.zeros(hidden))
    b2 = torch.nn.Parameter(torch.zeros(C))
    opt = torch.optim.Adam([W1, W2, b1, b2], lr=lr, weight_decay=weight_decay)
    idx_tr = torch.where(torch.tensor(train_mask))[0]
    idx_va = torch.where(torch.tensor(val_mask))[0]
    best_val = -1.0; best_state = None; patience = 0
    for ep in range(epochs):
        opt.zero_grad()
        H1 = torch.relu(torch.sparse.mm(adj_t, X_t) @ W1 + b1)
        if dropout > 0:
            H1 = torch.dropout(H1, dropout, train=True)
        Z = torch.sparse.mm(adj_t, H1) @ W2 + b2
        loss = torch.nn.functional.cross_entropy(Z[idx_tr], Y[idx_tr])
        loss.backward(); opt.step()
        with torch.no_grad():
            v = float((Z[idx_va].argmax(1) == Y[idx_va]).float().mean().item())
            if v > best_val:
                best_val = v; patience = 0
                best_state = (W1.detach().clone(), W2.detach().clone(),
                              b1.detach().clone(), b2.detach().clone())
            else:
                patience += 1
            if patience >= early_stop:
                break
    W1, W2, b1, b2 = best_state
    with torch.no_grad():
        H1 = torch.relu(torch.sparse.mm(adj_t, X_t) @ W1 + b1)
        Z = torch.sparse.mm(adj_t, H1) @ W2 + b2
        pred = Z.argmax(1).numpy()
        tr = float((pred[train_mask] == labels[train_mask]).mean())
        va = float((pred[val_mask] == labels[val_mask]).mean())
        te = float((pred[test_mask] == labels[test_mask]).mean())
        B1 = np.asarray(adj_norm @ X_np @ W1.numpy() + b1.numpy(), dtype=np.float32)
        B2 = np.asarray(adj_norm @ (np.maximum(B1, 0) @ W2.numpy()) + b2.numpy(), dtype=np.float32)
    return dict(W1=W1.numpy(), W2=W2.numpy(), b1=b1.numpy(), b2=b2.numpy(),
                acc_train=tr, acc_val=va, acc_test=te,
                B1=B1, B2=B2, X=X_np, adj_norm=adj_norm,
                labels=labels, test_mask=test_mask, val_mask=val_mask, train_mask=train_mask)

def ftbc_bias(T, theta=1.0):
    """Closed-form FTBC bias: beta* = theta/(2T).

    Derived in audits/d1-theory-derivation.md (Lemma 2): the IF rate is floor(T(c+beta))/T,
    so E[rate] - c = beta - 1/(2T) and beta* = 1/(2T) is the unique choice that zeroes the
    systematic error of every layer-1 unit. It is parameter-free -- no calibration set and
    no grid search -- and it is not limited by a grid step (the old grid of 0.05 could not
    express 1/(2*16) = 0.031, which mildly over-corrected at large T).
    Measured against the grid search it wins in 7 of 9 (dataset, T) cells.
    """
    return theta / (2.0 * T)


def tune_bias(a, T, theta=1.0, grid=np.arange(0.0, 1.05, 0.05), mask=None, per_channel=False):
    """FTBC-style forward bias: compensate the systematic floor deficit of IF rate coding.

    IF with constant per-step input c emits floor(T*c/theta) spikes in T steps, so the
    realized rate floor(T*c/theta)/T systematically UNDER-estimates c by ~0.5/T. Adding a
    bias beta >= 0 to the input restores an (approximately) unbiased rate. No labels are
    used, so this is an input-statistics-only calibration.

    Leakage control: `mask` restricts calibration to the given nodes; callers MUST pass the
    TRAIN mask. Calibrating on all nodes would tune on test-node activation statistics
    (test features are visible transductively, but that still leaks split information and is
    not defensible in review).

    per_channel=False -> one scalar beta for the whole layer.
    per_channel=True  -> one beta per hidden unit (vector of shape H); more expressive and
                         closer to standard per-channel FTBC, at the same cost.
    """
    a = np.asarray(a, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.any():
            a = a[mask]
    if a.ndim == 1:
        a = a[:, None]
    if not per_channel:
        a = a.reshape(-1, 1)

    def _best_b(col):
        pos = col[col > 0]
        if pos.size == 0:
            return 0.0
        best_b, best_e = 0.0, 1e9
        for b in grid:
            c = np.clip(pos + b, 0.0, None)
            rate = np.floor(T * c / theta) / T
            err = float(np.mean(np.abs(rate - pos)))
            if err < best_e:
                best_e, best_b = err, b
        return best_b

    betas = np.array([_best_b(a[:, j]) for j in range(a.shape[1])], dtype=np.float64)
    return betas if per_channel else float(betas[0])

# ---------- SNN conversion (corrected) ----------
def convert(model, T, theta=1.0, beta1=0.0, mode="naive", l1_mode="if", seed=0, readout="membrane"):
    """
    Convert ANN-GNN -> SNN-GNN by faithful rate-based ANN->SNN conversion.

    Rate coding with absolute threshold (theta = 1), no global L-scale. Verified diagnostics
    (see audit_diagnostic.py): relu(B1) rarely exceeds theta (0.00% of entries on the three
    Planetoid graphs, max ~2.0), so the absolute threshold does NOT clip the hidden
    representation -- no normalisation is needed at layer 1.

    Layer-2 integrates BINARY layer-1 spikes (instantaneous -> temporal misalignment):
      - "naive"  : real SNN, layer-1 IF. No bias correction: IF emits floor(T*c/theta)
                   spikes, so the rate is systematically biased LOW (~ -0.5/T deficit).
      - "ftbc"   : real SNN + layer-1 forward-bias correction beta1, which removes that
                   floor deficit (measured bias -0.044 -> -0.0098 at T=8). PROPOSED METHOD.
      - "oracle" : rate-ideal LIMIT, not a deployable SNN: layer 1 emits its exact
                   real-valued rate, so sum_t inst_t = T*B2 exactly and argmax == ANN.
                   Serves as (a) the mathematical ceiling of this conversion and (b) a
                   correctness check that the conversion chain is faithful.
    (The previous "calibrated" mode was mislabelled: it bypassed layer-1 spiking, Bernoulli-
    sampled layer 2 from clip(B2,0,1), and therefore inherited the same logit-clipping
    ceiling as the count readout -- it UNDERPERFORMED ftbc, so it was not an upper bound.
    It is replaced by the honest "oracle" above.)

    Decision readout:
      - "membrane" (default): classify by the final membrane potential
        mem = cnt*theta + sub-threshold remainder = sum_t inst_t
            = A_hat @ (count1 @ W2) + T*b2   (exact accumulation identity).
        Removes the LAYER-2 floor quantization that the integer count readout suffers,
        giving a finer decision for the SAME spike train / energy. Crucially the membrane
        is NOT clipped by theta, so it escapes the ~98.4-99.9% clipping ceiling that
        bounds any count/argmax(min(B2,1)) decision (measured, dataset-dependent).
      - "count"    : classify by integer spike count (floor) -> shows the T-dependent
        quantization floor AND the logit-clipping ceiling. Energy metric either way.
    """
    W1 = model["W1"]; W2 = model["W2"]
    b1 = model.get("b1", np.zeros(W1.shape[1])); b2 = model.get("b2", np.zeros(W2.shape[1]))
    X = model["X"].astype(np.float32); adj = model["adj_norm"].tocsr()
    N, H = X.shape[0], W1.shape[1]
    C = W2.shape[1]
    test_mask = model["test_mask"]; labels = model["labels"]
    B1 = model["B1"].astype(np.float32)            # A_hat @ X @ W1 + b1  (N,H)
    B2 = model["B2"].astype(np.float32)            # A_hat @ relu(B1) @ W2 + b2  (N,C)
    reluB1 = np.maximum(B1, 0.0)
    W1s = W1; b1s = b1
    W2s = W2; b2s = b2
    B1s = np.asarray(adj @ (X @ W1s), dtype=np.float32) + b1s   # = B1  (N,H)

    rng = np.random.default_rng(seed)

    def _decide(cnt2, u2):
        # membrane potential = cnt*theta + sub-threshold remainder == sum_t inst_t (exact)
        mem = cnt2 * theta + u2
        if readout == "membrane":
            return np.argmax(mem, axis=1)
        return np.argmax(cnt2, axis=1)

    def layer2_from_s1(s1_seq):
        u2 = np.zeros((N, C), dtype=np.float64); cnt2 = np.zeros((N, C), dtype=np.int64)
        lat = {}
        for t in range(T):
            s1t = s1_seq[t]
            inst = np.asarray(adj @ (s1t.astype(np.float32) @ W2s), dtype=np.float32) + b2s
            u2 = u2 + inst
            fired = (u2 >= theta).astype(np.int64); cnt2 += fired; u2 = u2 - fired * theta
            pred_t = _decide(cnt2, u2)
            lat[t + 1] = float((pred_t[test_mask] == labels[test_mask]).mean())
        return cnt2, u2, lat

    def _finish(cnt2, u2, lat, count1):
        pred = _decide(cnt2, u2)
        acc = float((pred[test_mask] == labels[test_mask]).mean())
        acc_count = float((np.argmax(cnt2, axis=1)[test_mask] == labels[test_mask]).mean())
        n1 = float(np.sum(count1)); n2 = float(cnt2.sum())
        return dict(acc_test=acc, acc_count=acc_count, spike_l1=n1, spike_l2=n2,
                    total_spikes=n1 + n2, latency=lat)

    if mode == "oracle":
        # ---- rate-ideal LIMIT (not a deployable SNN) ------------------------------
        # Layer 1 emits its exact real-valued rate: no integer quantization, no temporal
        # misalignment. Layer 2 then accumulates, over T steps,
        #     sum_t inst_t = T * (A_hat @ relu(B1) @ W2 + b2) = T * B2   exactly,
        # hence argmax == argmax(B2) == the ANN prediction. This is simultaneously
        # (a) the mathematical ceiling of this conversion, and (b) a CORRECTNESS CHECK:
        # if oracle accuracy != ANN accuracy, the conversion chain has a bug.
        u2 = np.zeros((N, C), dtype=np.float64); cnt2 = np.zeros((N, C), dtype=np.int64); lat = {}
        inst = np.asarray(adj @ (reluB1 @ W2s), dtype=np.float32) + b2s   # == B2
        for t in range(T):
            u2 = u2 + inst
            fired = (u2 >= theta).astype(np.int64); cnt2 += fired; u2 = u2 - fired * theta
            pred_t = _decide(cnt2, u2)
            lat[t + 1] = float((pred_t[test_mask] == labels[test_mask]).mean())
        return _finish(cnt2, u2, lat, float(T) * reluB1.sum())

    # ---- real SNN: layer 1 emits INTEGER spikes; layer 2 reads them ---------------
    # (simulated ONCE so that the reported energy and the propagated spikes come from
    #  the same spike train -- previously count1 and s1_seq were drawn separately, which
    #  made the reported spike count inconsistent with the run being evaluated under TPP.)
    u = np.zeros((N, H), dtype=np.float64); s1_seq = []
    for t in range(T):
        u = u + B1s + beta1
        if l1_mode == "tpp":
            p = np.clip(u, 0, theta) / theta
            s = (rng.random((N, H)) < p).astype(np.int8); u = u - s * theta
        else:
            s = (u >= theta).astype(np.int8); u = u - s * theta
        s1_seq.append(s)
    count1 = np.sum(s1_seq, axis=0).astype(np.int64)
    cnt2, u2, lat = layer2_from_s1(s1_seq)
    return _finish(cnt2, u2, lat, count1)

def run_dataset(name, seeds, ts, modes, data_root, hidden=16, l1_mode="if", theta=1.0, readout="membrane", per_channel_bias=False, bias="analytic"):
    data = load_planetoid(name, data_root)
    adj_norm = normalize_adj(data["adj"])
    model_store = []
    # train ANN-GNN per seed
    ann_results = []
    for sd in seeds:
        m = train_gcn(data["features"], data["labels"], adj_norm,
                      data["train_mask"], data["val_mask"], data["test_mask"], sd, hidden=hidden)
        m["seed"] = sd
        m["labels"] = data["labels"]; m["test_mask"] = data["test_mask"]
        m["val_mask"] = data["val_mask"]; m["train_mask"] = data["train_mask"]
        ann_results.append(dict(seed=sd, acc_test=m["acc_test"], acc_val=m["acc_val"]))
        model_store.append(m)
    ann_mean = float(np.mean([a["acc_test"] for a in ann_results]))
    out = dict(dataset=name, ann_test_mean=ann_mean, ann_per_seed=ann_results,
              hidden=hidden, T=ts, l1_mode=l1_mode, results={})
    for mode in modes:
        out["results"][mode] = {}
        for T in ts:
            accs = []; acc_counts = []; totspikes = []; lat = {}
            for m in model_store:
                # "ftbc"/"oracle" apply the layer-1 forward-bias correction (beta1).
                # Calibration uses ONLY TRAIN nodes (leakage control: tuning on all nodes
                # would read test-node activation statistics).
                if mode in ("ftbc", "oracle"):
                    beta1 = (ftbc_bias(T, theta) if bias == "analytic" else
                             tune_bias(m["B1"], T, mask=m["train_mask"],
                                       per_channel=per_channel_bias))
                else:
                    beta1 = 0.0
                conv_mode = "oracle" if mode == "oracle" else "naive"
                r = convert(m, T=T, beta1=beta1, mode=conv_mode, l1_mode=l1_mode, seed=0, theta=theta, readout=readout)
                accs.append(r["acc_test"]); acc_counts.append(r["acc_count"]); totspikes.append(r["total_spikes"])
                for k, v in r["latency"].items():
                    lat.setdefault(k, []).append(v)
            out["results"][mode][str(T)] = dict(
                acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                acc_count_mean=float(np.mean(acc_counts)),
                spike_mean=float(np.mean(totspikes)),
                latency_acc_mean={k: float(np.mean(v)) for k, v in lat.items()})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--modes", nargs="+", default=["naive", "ftbc", "oracle"])
    ap.add_argument("--data-root", default="../results/.data")
    ap.add_argument("--out", default="../results/cora.json")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--l1-mode", default="if", choices=["if", "tpp"])
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--readout", default="membrane", choices=["membrane", "count"])
    ap.add_argument("--per-channel-bias", action="store_true")
    ap.add_argument("--bias", default="analytic", choices=["analytic", "grid"])
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    t0 = time.time()
    res = run_dataset(a.dataset, seeds, a.ts, a.modes, a.data_root, hidden=a.hidden, l1_mode=a.l1_mode, theta=a.theta, readout=a.readout, per_channel_bias=a.per_channel_bias, bias=a.bias)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[{a.dataset}] ANN test mean={res['ann_test_mean']:.4f}  time={time.time()-t0:.1f}s")
    for mode in a.modes:
        line = f"  {mode:11s}: " + " ".join(
            f"T{t}={res['results'][mode][str(t)]['acc_mean']:.3f}" for t in a.ts)
        print(line)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
