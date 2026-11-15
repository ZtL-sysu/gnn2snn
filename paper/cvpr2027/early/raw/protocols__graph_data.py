#!/usr/bin/env python3
"""Unified graph-data hub for the D1 strengthening experiments.

Everything is kept SPARSE: the original load_planetoid() materialised an N x N dense
adjacency, which already cost ~1.5 GB on PubMed and is simply impossible on ogbn-arxiv
(169k nodes -> 2.8e10 entries). Here adjacency is built directly as scipy CSR, so the
same conversion code scales to 100k+ node graphs without change.

Each loader returns a dict compatible with snn_gnn_experiment.run_dataset():
    features (N,F float32), labels (N,), adj (raw CSR with self-loops, symmetrised),
    train_mask, val_mask, test_mask      -> then normalize_adj(adj) before use.

Dataset notes (why each was added):
  cora/citeseer/pubmed : baseline homophilic Planetoid (already in the core run).
  ogbn-arxiv (169k)    : SCALE + standard OGB benchmark; tests that the method does not
                         depend on small-graph artefacts and survives a real citation graph
                         with train/val/test given by the official time-based split.
  chameleon (2.3k)     : HETEROPHILIC (low homophily). Rate coding averages neighbour
                         messages, which is easy under high homophily and hard otherwise;
                         this is the honest stress test for the conversion.
  squirrel             : second heterophilic point (same family, another size).
"""
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------- helpers
def _to_csr(n, rows, cols, add_self_loops=True, symmetrize=True):
    """Build the binary adjacency, reproducing the reference load_planetoid() semantics.

    Two pitfalls that silently changed the graph if ignored:
      * CSR construction SUMS duplicate (i,j) coordinates. The reference implementation
        assigns adj[i,j] = 1.0 in a loop, which deduplicates. Duplicate neighbour entries
        therefore made some edge weights 2 here and 1 there, changing the normalised
        degree -> measurable accuracy drift (CiteSeer ANN 0.6974 vs 0.7016). We binarise.
      * Order is +self_loops THEN symmetrise, matching the reference exactly.
    """
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    adj.data = np.ones_like(adj.data)          # collapse duplicate entries to weight 1
    if add_self_loops:
        adj = adj + sp.eye(n, dtype=np.float32, format="csr")
    if symmetrize:
        adj = adj.maximum(adj.T)
    adj.sort_indices()
    return adj


def _masks_from_idx(n, train_idx, val_idx, test_idx):
    out = []
    for idx in (train_idx, val_idx, test_idx):
        m = np.zeros(n, dtype=bool)
        idx = np.asarray(idx, dtype=np.int64)
        m[idx[idx >= 0]] = True
        out.append(m)
    tr, va, te = out
    tr = tr & ~va & ~te
    va = va & ~te
    return tr, va, te


# ---------------------------------------------------------------- Planetoid (sparse)
PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"


def load_planetoid_sparse(name, data_root):
    """Same rows/labels/splits as snn_gnn_experiment.load_planetoid but CSR adjacency."""
    import ssl, urllib.request
    data_root = Path(data_root); data_root.mkdir(parents=True, exist_ok=True)
    files = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    paths = {}
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    for f in files:
        p = data_root / f"ind.{name}.{f}"
        if not p.exists():
            with urllib.request.urlopen(f"{PLANETOID_BASE}/ind.{name}.{f}", timeout=60, context=ctx) as r:
                p.write_bytes(r.read())
        paths[f] = p

    def load_pkl(p):
        with open(p, "rb") as f:
            return pickle.load(f, encoding="latin1")

    def load_npz(p):
        obj = np.load(p, allow_pickle=True, encoding="latin1")
        if hasattr(obj, "toarray"):
            return obj.toarray()
        try:
            arr = obj["arr_0"]
            return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
        except Exception:
            return sp.load_npz(p).toarray()

    if name == "pubmed":
        x = load_npz(paths["x"]); tx = load_npz(paths["tx"]); allx = load_npz(paths["allx"])
    else:
        def _dense(obj):
            return obj.toarray() if hasattr(obj, "toarray") else np.asarray(obj)
        x = _dense(load_pkl(paths["x"]))
        tx = _dense(load_pkl(paths["tx"]))
        allx = _dense(load_pkl(paths["allx"]))

    y = load_pkl(paths["y"]); ty = load_pkl(paths["ty"]); ally = load_pkl(paths["ally"])
    graph = load_pkl(paths["graph"])
    test_idx = np.fromfile(open(paths["test.index"], "r"), sep="\n", dtype=int)

    n = max(graph.keys()) + 1
    nfeat = allx.shape[1]
    features = np.zeros((n, nfeat), dtype=np.float32)
    features[:allx.shape[0]] = allx.astype(np.float32)
    for i, gi in enumerate(test_idx):
        features[gi] = tx[i].astype(np.float32)

    labels = -np.ones(n, dtype=np.int64)
    labels[:ally.shape[0]] = np.argmax(np.asarray(ally), axis=1).astype(np.int64)
    for i, gi in enumerate(test_idx):
        labels[gi] = int(np.argmax(np.asarray(ty[i])))

    rows, cols = [], []
    for i, nbrs in graph.items():
        for j in nbrs:
            rows.append(i); cols.append(j)
    adj = _to_csr(n, rows, cols)

    train_mask = np.zeros(n, dtype=bool); val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:y.shape[0]] = True
    val_mask[y.shape[0]:y.shape[0] + 500] = True
    test_mask[test_idx] = True
    train_mask[val_mask] = False
    train_mask[test_mask] = False
    return dict(features=features, labels=labels, adj=adj,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


# ---------------------------------------------------------------- ogbn-arxiv
def load_ogbn_arxiv(data_root):
    """ogbn-arxiv read straight from OGB's RAW csv.gz files.

    Why not use the `ogb` package here: after the first successful extraction it deletes
    arxiv.zip, and PyTorch >= 2.6 also refuses to unpickle ogb's cached processed file
    (`weights_only=True` by default). Both make `NodePropPredDataset` fail intermittently
    depending on cache state. The raw files are plain, well-specified CSVs, so parsing them
    directly is deterministic, offline and reproducible.
    Verified against OGB's declared sizes: 169343 nodes / 1166243 edges / 128 features /
    40 classes, official time-based split 90941 / 29799 / 48603.
    """
    import pandas as pd
    base = Path(data_root) / "ogb" / "ogbn_arxiv"
    raw = base / "raw"
    if not (raw / "edge.csv.gz").exists():
        # fall back to the official loader (it will download) if raw files are absent
        from ogb.nodeproppred import NodePropPredDataset
        return load_ogbn_arxiv_via_ogb(data_root)
    n = int(pd.read_csv(raw / "num-node-list.csv.gz", header=None).iloc[0, 0])
    edge = pd.read_csv(raw / "edge.csv.gz", header=None).to_numpy(dtype=np.int64)
    if int(pd.read_csv(raw / "num-edge-list.csv.gz", header=None).iloc[0, 0]) != edge.shape[0]:
        raise ValueError("ogbn-arxiv edge file row count does not match its declared size")
    features = pd.read_csv(raw / "node-feat.csv.gz", header=None).to_numpy(dtype=np.float32)
    labels = pd.read_csv(raw / "node-label.csv.gz", header=None).to_numpy(dtype=np.int64).ravel()
    if features.shape[0] != n or labels.shape[0] != n:
        raise ValueError(f"ogbn-arxiv row counts disagree: N={n} "
                         f"features={features.shape[0]} labels={labels.shape[0]}")
    sp_dir = base / "split" / "time"
    tr = pd.read_csv(sp_dir / "train.csv.gz", header=None).to_numpy(dtype=np.int64).ravel()
    va = pd.read_csv(sp_dir / "valid.csv.gz", header=None).to_numpy(dtype=np.int64).ravel()
    te = pd.read_csv(sp_dir / "test.csv.gz", header=None).to_numpy(dtype=np.int64).ravel()
    adj = _to_csr(n, edge[:, 0], edge[:, 1])
    tr_m, va_m, te_m = _masks_from_idx(n, tr, va, te)
    return dict(features=features, labels=labels, adj=adj,
                train_mask=tr_m, val_mask=va_m, test_mask=te_m)


def load_ogbn_arxiv_via_ogb(data_root):
    """Original ogb-based path, only used when the raw files are not present locally."""
    import torch
    from ogb.nodeproppred import NodePropPredDataset
    _orig_load = torch.load

    def _load_permissive(*args, **kwargs):
        kwargs.setdefault("weights_only", False)   # our own local ogb cache
        return _orig_load(*args, **kwargs)

    torch.load = _load_permissive
    try:
        ds = NodePropPredDataset(name="ogbn-arxiv", root=str(Path(data_root) / "ogb"))
        g, y = ds[0]
    finally:
        torch.load = _orig_load
    ei = g["edge_index"]; n = int(g["num_nodes"])
    adj = _to_csr(n, np.asarray(ei[0], dtype=np.int64), np.asarray(ei[1], dtype=np.int64))
    si = ds.get_idx_split()
    tr, va, te = _masks_from_idx(n, si["train"], si["valid"], si["test"])
    return dict(features=np.asarray(g["node_feat"], dtype=np.float32),
                labels=np.asarray(y, dtype=np.int64).ravel(), adj=adj,
                train_mask=tr, val_mask=va, test_mask=te)


# ---------------------------------------------------------------- WikipediaNetwork
def load_wikipedia(name, data_root, split=0):
    """Heterophilic Wikipedia page-page graphs (chameleon / squirrel)."""
    from torch_geometric.datasets import WikipediaNetwork
    ds = WikipediaNetwork(root=str(Path(data_root) / "pyg"), name=name)
    d = ds[0]
    n = int(d.num_nodes)
    ei = d.edge_index.numpy()
    adj = _to_csr(n, ei[0].astype(np.int64), ei[1].astype(np.int64))
    # The benchmark ships 10 standard 48/32/20 random splits as [N, 10] masks.
    def get_split(key):
        m = getattr(d, key, None)
        if m is None:
            return None
        m = m.numpy()
        return m[:, split] if m.ndim == 2 else m
    tr, va, te = get_split("train_mask"), get_split("val_mask"), get_split("test_mask")
    if tr is None:  # fall back to standard 48/32/20 if the attribute is absent
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        a, b = int(0.48 * n), int(0.80 * n)
        tr = np.zeros(n, bool); va = np.zeros(n, bool); te = np.zeros(n, bool)
        te[perm[b:]] = True; va[perm[a:b]] = True; tr[perm[:a]] = True
    return dict(features=np.asarray(d.x, dtype=np.float32),
                labels=np.asarray(d.y, dtype=np.int64).ravel(), adj=adj,
                train_mask=tr, val_mask=va, test_mask=te)


REGISTRY = {
    "cora": lambda root: load_planetoid_sparse("cora", root),
    "citeseer": lambda root: load_planetoid_sparse("citeseer", root),
    "pubmed": lambda root: load_planetoid_sparse("pubmed", root),
    "ogbn-arxiv": load_ogbn_arxiv,
    "chameleon": lambda root: load_wikipedia("chameleon", root),
    "squirrel": lambda root: load_wikipedia("squirrel", root),
}

# L1 row-normalisation is part of the Planetoid BoW recipe; the others ship normalised
# features and must NOT be re-normalised.
ROW_NORMALIZE = {"cora": True, "citeseer": True, "pubmed": True,
                 "ogbn-arxiv": False, "chameleon": False, "squirrel": False}


def load_ds(name, data_root="../results/.data"):
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name}; known: {sorted(REGISTRY)}")
    d = REGISTRY[name](data_root)
    d["row_normalize"] = ROW_NORMALIZE[name]
    return d


def edge_homophily(adj, labels):
    """Standard edge homophily: fraction of edges joining same-label endpoints."""
    labels = np.asarray(labels)
    adj = adj.tocsr()
    ind, indptr = adj.indices, adj.indptr
    same = tot = 0.0
    for i in range(adj.shape[0]):
        nb = ind[indptr[i]:indptr[i + 1]]
        nb = nb[nb != i]
        if nb.size == 0:
            continue
        same += float((labels[nb] == labels[i]).sum()); tot += nb.size
    return same / max(tot, 1e-9)


if __name__ == "__main__":
    import sys, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../results/.data")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    names = a.only or list(REGISTRY)
    for nm in names:
        try:
            d = load_ds(nm, a.root)
            hom = edge_homophily(d["adj"], d["labels"])
            print(f"{nm:12s} N={d['features'].shape[0]:7d} F={d['features'].shape[1]:5d} "
                  f"E={int(d['adj'].nnz):9d} C={int(d['labels'].max())+1:3d} "
                  f"train={int(d['train_mask'].sum()):6d} homophily={hom:.3f}")
        except Exception as e:
            print(f"{nm:12s} FAILED: {type(e).__name__}: {str(e)[:120]}")
            sys.stdout.flush()
