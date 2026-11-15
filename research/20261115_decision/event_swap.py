"""Support- and alphabet-preserving first-tick event swap primitives."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

Swap = tuple[int, int, int]


def _validate_base(events: torch.Tensor, ptr: torch.Tensor, outdegree: torch.Tensor) -> None:
    if not isinstance(events, torch.Tensor) or events.ndim != 2:
        raise ValueError("events must be a Tensor[N,C]")
    if not events.is_floating_point():
        raise ValueError("events must have a floating dtype")
    if not isinstance(ptr, torch.Tensor) or ptr.device != events.device or ptr.dtype != torch.long or ptr.ndim != 1:
        raise ValueError("ptr must be a LongTensor[G+1] on the events device")
    if len(ptr) < 2 or int(ptr[0]) != 0 or int(ptr[-1]) != len(events):
        raise ValueError("ptr must start at 0 and end at the event node count")
    if bool((ptr[1:] < ptr[:-1]).any()):
        raise ValueError("ptr must be nondecreasing")
    if not isinstance(outdegree, torch.Tensor) or outdegree.device != events.device or outdegree.dtype != torch.long or outdegree.shape != (len(events),):
        raise ValueError("outdegree must be a LongTensor[N] on the events device")
    if bool((outdegree < 0).any()):
        raise ValueError("outdegree values must be nonnegative")
    if not bool(torch.isfinite(events).all()):
        raise ValueError("events contains nonfinite values")


def _rng(seed: int, graph_id: int) -> np.random.Generator:
    # Stable unsigned split lets independent graph RNG streams work for any
    # signed Python integer graph ID without relying on batch position.
    material = f"{int(seed)}:{int(graph_id)}".encode("ascii")
    entropy = int.from_bytes(hashlib.sha256(material).digest()[:16], "little")
    return np.random.default_rng(entropy)


def propose_swaps(
    events: torch.Tensor,
    ptr: torch.Tensor,
    outdegree: torch.Tensor,
    graph_ids: Sequence[int],
    count: int = 8,
    seed: int = 20261117,
) -> list[list[Swap | None]]:
    """Propose candidate-major, per-graph event swaps without reading labels.

    Each non-None tuple is ``(global_node_u, global_node_v, channel)``.
    Eligible nodes share a graph, channel and exact outdegree, and carry
    different nonzero event symbols.  Node-pair products are never enumerated;
    bounded random sampling is done over amplitude groups.
    """
    _validate_base(events, ptr, outdegree)
    if events.device.type != "cpu":
        raise ValueError("propose_swaps requires CPU events, ptr and outdegree")
    graphs = len(ptr) - 1
    if len(graph_ids) != graphs:
        raise ValueError(f"graph_ids length {len(graph_ids)} != graph count {graphs}")
    ids = [int(x) for x in graph_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("graph_ids must be unique within a batch")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if count == 0:
        return []

    per_graph: list[list[Swap | None]] = []
    for gi, graph_id in enumerate(ids):
        lo, hi = int(ptr[gi]), int(ptr[gi + 1])
        rng = _rng(seed, graph_id)
        # Group only occupied (channel, outdegree, amplitude) cells.  The
        # symbol alphabet is small for multi-level event codes; storage is O(NC).
        grouped: dict[tuple[int, int], dict[float, list[int]]] = defaultdict(lambda: defaultdict(list))
        for node in range(lo, hi):
            degree = int(outdegree[node])
            for channel in range(events.shape[1]):
                value = float(events[node, channel])
                if value != 0.0:
                    grouped[(channel, degree)][value].append(node)
        buckets = [(key, values) for key, values in grouped.items() if len(values) >= 2]
        chosen: list[Swap | None] = []
        seen: set[tuple[int, int, int]] = set()
        # Bounded retries avoid materializing any node-pair cross products.
        max_trials = max(256, count * 128)
        for _ in range(max_trials):
            if len(chosen) >= count or not buckets:
                break
            bucket_index = int(rng.integers(0, len(buckets)))
            (channel, _degree), values = buckets[bucket_index]
            amplitudes = list(values)
            pick = rng.choice(len(amplitudes), size=2, replace=False)
            amp_a, amp_b = amplitudes[int(pick[0])], amplitudes[int(pick[1])]
            u = values[amp_a][int(rng.integers(0, len(values[amp_a])))]
            v = values[amp_b][int(rng.integers(0, len(values[amp_b])))]
            if u > v:
                u, v = v, u
            pair = (u, v, channel)
            if pair not in seen:
                seen.add(pair)
                chosen.append(pair)
        chosen.extend([None] * (count - len(chosen)))
        per_graph.append(chosen)

    # Public contract is candidate-major: candidate k has one entry per graph.
    return [[per_graph[g][k] for g in range(graphs)] for k in range(count)]


def apply_swaps(
    events: torch.Tensor,
    pairs: Sequence[Swap | None],
    ptr: torch.Tensor,
    outdegree: torch.Tensor,
) -> torch.Tensor:
    """Return a same-device copy with validated per-graph swaps applied.

    CUDA inputs are supported for the runner; scalar validity checks may
    synchronize, and no GPU tensor is converted through NumPy.
    """
    _validate_base(events, ptr, outdegree)
    graphs = len(ptr) - 1
    if len(pairs) != graphs:
        raise ValueError(f"pairs length {len(pairs)} != graph count {graphs}")
    plan: list[Swap] = []
    occupied: set[tuple[int, int]] = set()
    for gi, item in enumerate(pairs):
        if item is None:
            continue
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(f"graph {gi}: swap must be None or (node_u,node_v,channel)")
        u, v, channel = item
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) for x in item):
            raise ValueError(f"graph {gi}: swap coordinates must be integers")
        u, v, channel = int(u), int(v), int(channel)
        lo, hi = int(ptr[gi]), int(ptr[gi + 1])
        if not (lo <= u < hi and lo <= v < hi):
            raise ValueError(f"graph {gi}: nodes must both lie within [{lo},{hi})")
        if u == v:
            raise ValueError(f"graph {gi}: swap nodes must be distinct")
        if not 0 <= channel < events.shape[1]:
            raise ValueError(f"graph {gi}: channel {channel} out of range")
        if int(outdegree[u]) != int(outdegree[v]):
            raise ValueError(f"graph {gi}: swap nodes have different outdegree")
        a, b = events[u, channel], events[v, channel]
        if bool(a == 0) or bool(b == 0):
            raise ValueError(f"graph {gi}: both event symbols must be nonzero")
        if bool(a == b):
            raise ValueError(f"graph {gi}: event symbols must differ")
        for node in (u, v):
            cell = (node, channel)
            if cell in occupied:
                raise ValueError(f"graph {gi}: a node/channel cell appears in multiple swaps")
            occupied.add(cell)
        plan.append((u, v, channel))

    changed = events.clone()
    # Validate the whole plan before applying any swap; caller input is untouched.
    for u, v, channel in plan:
        tmp = changed[u, channel].clone()
        changed[u, channel] = changed[v, channel]
        changed[v, channel] = tmp
    return changed
