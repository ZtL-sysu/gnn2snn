"""Fixed-capacity feature, ridge and gating utilities for two-event swaps."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

MODES = ("nongraph", "graph", "shuffled", "toppin")
BASE_FEATURES = 13
CONDITIONING_FEATURES = 3
CHANNELS = 146
FEATURES = 162
OUTPUTS = 8
SHUFFLE_SEED = 20261118


def _graph_rng(graph_id: int) -> np.random.Generator:
    seed_bytes = hashlib.sha256(f"{SHUFFLE_SEED}:{int(graph_id)}".encode("ascii")).digest()[:16]
    return np.random.default_rng(int.from_bytes(seed_bytes, "little"))


def _validate_graph_inputs(edge_index: torch.Tensor, ptr: torch.Tensor, graph_ids: list[int], mode: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not isinstance(edge_index, torch.Tensor) or edge_index.device.type != "cpu" or edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a CPU LongTensor[2,E]")
    if not isinstance(ptr, torch.Tensor) or ptr.device.type != "cpu" or ptr.dtype != torch.long or ptr.ndim != 1:
        raise ValueError("ptr must be a CPU LongTensor[G+1]")
    if len(ptr) < 2 or int(ptr[0]) != 0 or bool((ptr[1:] < ptr[:-1]).any()):
        raise ValueError("ptr must start at zero and be nondecreasing")
    n = int(ptr[-1])
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
        raise ValueError("edge_index contains out-of-range node ids")
    gids = [int(x) for x in graph_ids]
    if len(gids) != len(ptr) - 1 or len(set(gids)) != len(gids):
        raise ValueError("graph_ids must be unique and match graph count")
    src, dst = edge_index.numpy()
    # Prevent cross-graph edges, which would invalidate per-graph cache isolation.
    node_graph = np.empty(n, dtype=np.int64)
    p = ptr.numpy()
    for gi, (lo, hi) in enumerate(zip(p[:-1], p[1:])):
        node_graph[int(lo):int(hi)] = gi
    if len(src) and not np.array_equal(node_graph[src], node_graph[dst]):
        raise ValueError("edge_index contains a cross-graph edge")
    return edge_index.numpy(), p, gids


def graph_cache(edge_index: torch.Tensor, ptr: torch.Tensor, graph_ids: list[int], mode: str) -> dict[str, Any]:
    """Cache only graph descriptors required by one of the fixed predictor arms.

    ``A[v,u] = 1/sqrt(max(outdegree[u],1)*max(indegree[v],1))`` follows the
    normalized directed message operator.  ``mass[u]=sum_v A[v,u]`` and
    ``mass2[u]=sum_v mass[v]*A[v,u]`` are the one- and two-hop readout masses.
    Shuffled mode permutes these two vectors together within each graph using
    a graph-ID-seeded fixed RNG; it never changes ``edge_index``.
    """
    edges, p, gids = _validate_graph_inputs(edge_index, ptr, graph_ids, mode)
    src, dst = edges
    n, gcount = int(p[-1]), len(p) - 1
    outdegree = np.bincount(src, minlength=n).astype(np.int64, copy=False)
    cache: dict[str, Any] = {
        "mode": mode,
        "graph_ids": np.asarray(gids, dtype=np.int64),
        "ptr": p.astype(np.int64, copy=True),
        "outdegree": outdegree.copy(),
        "graph_size": np.diff(p).astype(np.int64, copy=True),
    }

    if mode in ("graph", "shuffled", "toppin"):
        indegree = np.bincount(dst, minlength=n).astype(np.int64, copy=False)
        cache["indegree"] = indegree.copy()

    if mode in ("graph", "shuffled"):
        mass = np.zeros(n, dtype=np.float64)
        # Compute each graph's A column sums independently.
        for lo, hi in zip(p[:-1], p[1:]):
            lo, hi = int(lo), int(hi)
            mask = (src >= lo) & (src < hi)
            s, d = src[mask], dst[mask]
            weight = 1.0 / np.sqrt(np.maximum(outdegree[s], 1) * np.maximum(cache["indegree"][d], 1))
            np.add.at(mass, s, weight)
        mass2 = np.zeros(n, dtype=np.float64)
        for lo, hi in zip(p[:-1], p[1:]):
            lo, hi = int(lo), int(hi)
            mask = (src >= lo) & (src < hi)
            s, d = src[mask], dst[mask]
            weight = 1.0 / np.sqrt(np.maximum(outdegree[s], 1) * np.maximum(cache["indegree"][d], 1))
            np.add.at(mass2, s, mass[d] * weight)
        cache["mass"] = mass
        cache["mass2"] = mass2
        if mode == "shuffled":
            shuffled_mass = mass.copy()
            shuffled_mass2 = mass2.copy()
            for gi, (lo, hi) in enumerate(zip(p[:-1], p[1:])):
                lo, hi = int(lo), int(hi)
                permutation = _graph_rng(gids[gi]).permutation(hi - lo)
                shuffled_mass[lo:hi] = mass[lo:hi][permutation]
                shuffled_mass2[lo:hi] = mass2[lo:hi][permutation]
            cache["shuffled_mass"] = shuffled_mass
            cache["shuffled_mass2"] = shuffled_mass2

    if mode == "toppin":
        indegree = cache["indegree"]
        reciprocal_mean = np.zeros(n, dtype=np.float64)
        if len(src):
            reciprocal = 1.0 / np.maximum(indegree[src], 1)
            sums = np.zeros(n, dtype=np.float64)
            np.add.at(sums, dst, reciprocal)
            nonzero = indegree > 0
            reciprocal_mean[nonzero] = sums[nonzero] / indegree[nonzero]
        cache["mean_reciprocal_in_neighbor_indegree"] = reciprocal_mean

    arrays = [v for v in cache.values() if isinstance(v, np.ndarray)]
    cache["cache_nbytes"] = int(sum(v.nbytes for v in arrays))
    return cache


def _as_np_array(value: Any, name: str, ndim: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be CPU-resident")
        value = value.numpy()
    a = np.asarray(value)
    if a.ndim != ndim or not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{name} must be a numeric {ndim}-D array")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def pair_features(events: np.ndarray, membrane: np.ndarray, theta: np.ndarray,
                  pairs: np.ndarray, ptr: np.ndarray, cache: dict[str, Any],
                  mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Construct 13 base + 3 mode features + 146 channel-one-hot values."""
    if mode not in MODES or cache.get("mode") != mode:
        raise ValueError("feature mode must match a cache made for the same mode")
    q = _as_np_array(events, "events", 2).astype(np.float64, copy=False)
    m = _as_np_array(membrane, "membrane", 2).astype(np.float64, copy=False)
    th = _as_np_array(theta, "theta", 1).astype(np.float64, copy=False)
    pp = np.asarray(pairs)
    if pp.ndim != 3 or pp.shape[2] != 3 or not np.issubdtype(pp.dtype, np.integer):
        raise ValueError("pairs must be integer array[K,G,3] with absolute node indices")
    p = np.asarray(ptr)
    if p.ndim != 1 or len(p) != pp.shape[1] + 1 or p[0] != 0 or p[-1] != q.shape[0] or np.any(p[1:] < p[:-1]):
        raise ValueError("ptr shape/order must match pair graph axis and event node count")
    if q.shape != m.shape or q.shape[1] != CHANNELS or th.shape != (CHANNELS,):
        raise ValueError(f"events/membrane must be [N,{CHANNELS}], theta must be [{CHANNELS}]")
    if np.any(th <= 0):
        raise ValueError("theta entries must be positive")
    if len(cache.get("outdegree", [])) != q.shape[0] or not np.array_equal(cache.get("ptr"), p):
        raise ValueError("cache node/graph layout does not match features")

    kcount, gcount = pp.shape[:2]
    features = np.zeros((kcount, gcount, FEATURES), dtype=np.float64)
    valid = np.zeros((kcount, gcount), dtype=bool)
    outdegree = cache["outdegree"]
    for k in range(kcount):
        for gi in range(gcount):
            u, v, ch = (int(x) for x in pp[k, gi])
            if (u, v, ch) == (-1, -1, -1):
                continue
            if min(u, v, ch) < 0:
                raise ValueError(f"pair {k},{gi} has a partial negative sentinel")
            lo, hi = int(p[gi]), int(p[gi + 1])
            if not (lo <= u < hi and lo <= v < hi) or u == v or ch >= CHANNELS:
                raise ValueError(f"pair {k},{gi} has invalid graph/node/channel coordinates")
            qu, qv = q[u, ch], q[v, ch]
            if qu == 0 or qv == 0 or qu == qv:
                raise ValueError(f"pair {k},{gi} must reference distinct nonzero event symbols")
            if outdegree[u] != outdegree[v]:
                raise ValueError(f"pair {k},{gi} has unequal outdegree")
            mu, mv, scale = m[u, ch], m[v, ch], th[ch]
            d = (qv - qu) / scale
            md = (mu - mv) / scale
            graph_q = q[lo:hi, ch]
            graph_m = m[lo:hi, ch]
            base = np.asarray([
                d,
                md,
                d * d,
                d * md,
                qu / scale,
                qv / scale,
                (mu + mv) / scale,
                float(outdegree[u]),
                graph_q.mean() / scale,
                np.count_nonzero(graph_q) / len(graph_q),
                graph_m.mean() / scale,
                np.log1p(len(graph_q)),
                (qu + qv) / scale,
            ], dtype=np.float64)
            if mode == "nongraph":
                extra = np.asarray([
                    d * (mu * mu - mv * mv) / (scale * scale),
                    d * (mu + mv) / scale,
                    d * (abs(mu) - abs(mv)) / scale,
                ])
            elif mode in ("graph", "shuffled"):
                mass = cache["mass"] if mode == "graph" else cache["shuffled_mass"]
                mass2 = cache["mass2"] if mode == "graph" else cache["shuffled_mass2"]
                extra = np.asarray([
                    d * (mass[u] - mass[v]),
                    d * (mass[u] * mu - mass[v] * mv) / scale,
                    d * (mass2[u] - mass2[v]),
                ])
            else:
                indeg = cache["indegree"]
                reciprocal = cache["mean_reciprocal_in_neighbor_indegree"]
                extra = np.asarray([
                    d * (indeg[u] - indeg[v]),
                    d * (reciprocal[u] - reciprocal[v]),
                    d * (indeg[u] * mu - indeg[v] * mv) / scale,
                ])
            start = BASE_FEATURES
            features[k, gi, :BASE_FEATURES] = base
            features[k, gi, start:start + CONDITIONING_FEATURES] = extra
            features[k, gi, start + CONDITIONING_FEATURES + ch] = 1.0
            valid[k, gi] = True
    return features, valid


def ridge_fit(X2D: np.ndarray, y2D8outputs: np.ndarray, alpha: float = 0.001) -> dict[str, Any]:
    """Fit standardized multi-output ridge with an unpenalized intercept."""
    x = _as_np_array(X2D, "X2D", 2).astype(np.float64, copy=False)
    y = _as_np_array(y2D8outputs, "y2D8outputs", 2).astype(np.float64, copy=False)
    if x.shape[0] == 0 or x.shape[1] != FEATURES or y.shape != (x.shape[0], OUTPUTS):
        raise ValueError(f"ridge expects X[n,{FEATURES}] and y[n,{OUTPUTS}] with n>0")
    if not np.isfinite(alpha) or float(alpha) != 0.001:
        raise ValueError("this fixed protocol requires ridge alpha=0.001")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ y
    coef = np.linalg.solve(gram, rhs)
    return {"mean": mean, "std": std, "coef": coef, "alpha": float(alpha),
            "n_features": FEATURES, "n_outputs": OUTPUTS, "n_samples": int(len(x)),
            "intercept_unpenalized": True}


def ridge_predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    x = _as_np_array(X, "X", 3 if np.asarray(X).ndim == 3 else 2).astype(np.float64, copy=False)
    if x.shape[-1] != np.asarray(model["mean"]).size:
        raise ValueError("predictor feature dimension mismatch")
    mean, std, coef = model["mean"], model["std"], model["coef"]
    z = (x - mean) / std
    design = np.concatenate((z, np.ones(z.shape[:-1] + (1,), dtype=np.float64)), axis=-1)
    pred = design @ coef
    if not np.isfinite(pred).all():
        raise ValueError("ridge prediction is nonfinite")
    return pred


def choose_pairs(pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return candidate index+1 per graph, or zero for no swap.

    Output heads: T1 score, T2 score, then six predicted cost changes.
    A candidate is eligible only when T1>0, T2>=0, and all costs<=0.
    """
    p = _as_np_array(pred, "pred", 3).astype(np.float64, copy=False)
    v = np.asarray(valid, dtype=bool)
    if p.shape[2] != OUTPUTS or p.shape[:2] != v.shape:
        raise ValueError(f"pred must be [K,G,{OUTPUTS}] and valid [K,G]")
    eligible = v & (p[:, :, 0] > 0) & (p[:, :, 1] >= 0) & np.all(p[:, :, 2:] <= 0, axis=2)
    chosen = np.zeros(p.shape[1], dtype=np.int64)
    for gi in range(p.shape[1]):
        ids = np.flatnonzero(eligible[:, gi])
        if len(ids):
            best = ids[np.argmax(p[ids, gi, 0])]
            chosen[gi] = int(best) + 1
    return chosen
