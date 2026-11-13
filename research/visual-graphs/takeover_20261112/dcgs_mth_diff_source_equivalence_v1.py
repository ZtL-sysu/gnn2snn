"""CPU source-equivalence oracle and fixed-operator GCN unit prototype.

The reference is the pinned author module ``neurons.relu_neuron.MTH_diff``.
Pass the clean checkout root through --author-repo; it is imported read-only.
The companion pure-Python stepper is a literal transcription of that module's
step-mode-s recurrence, including its 0.75 threshold margin and negative-spike
guard. It is intentionally separate from the paper-equation oracle.
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PINNED_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
Vector = List[float]
Matrix = List[List[float]]


@dataclass
class State:
    bias: Vector
    exp_in: Vector
    exp_out: Vector
    membrane: Optional[Vector] = None
    cumulative_out: Optional[Vector] = None
    active_ticks: int = 0


def _thresholds(theta: Sequence[float], n_levels: int) -> List[Vector]:
    if n_levels < 1 or any(x <= 0 for x in theta):
        raise ValueError("positive theta and n_levels required")
    return [[x / (2**k) for x in theta] for k in range(n_levels)]


def source_exact_stepper(frames: Sequence[Sequence[float]], theta: Sequence[float], n_levels: int) -> List[Dict]:
    """Pure-Python transcription of MTH_diff(step_mode='s'), one frame at a time.

    ``frames[0]`` is a non-emitting baseline/init call. Remaining frames are
    the actual current slices. Returned state is captured after each tick.
    """
    if len(frames) < 1:
        raise ValueError("include at least the baseline frame")
    width = len(frames[0])
    if len(theta) != width or any(len(x) != width for x in frames):
        raise ValueError("frame/theta widths must match")
    ladder = _thresholds(theta, n_levels)
    state = State(list(map(float, frames[0])), list(map(float, frames[0])), [0.0] * width)
    traces: List[Dict] = []

    for frame in frames[1:]:
        if state.membrane is None:
            state.membrane = [0.0] * width
            state.cumulative_out = [0.0] * width
        t = state.active_ticks + 1
        x = list(map(float, frame))

        # Match relu_neuron.py::MTH_diff step_mode='s' operation order.
        mem = [state.membrane[j] + x[j] - state.bias[j] + state.exp_in[j] - state.exp_out[j]
               for j in range(width)]
        cumulative = [state.cumulative_out[j] + state.exp_out[j] for j in range(width)]
        pos_false = [[1 if mem[j] >= 0.75 * ladder[k][j] else 0 for j in range(width)]
                     for k in range(n_levels)]
        neg_false = [[1 if (mem[j] <= -0.75 * ladder[k][j] and
                            cumulative[j] >= ladder[k][j]) else 0 for j in range(width)]
                     for k in range(n_levels)]
        pos_true = [[pos_false[k][j] - (pos_false[k - 1][j] if k else 0)
                     for j in range(width)] for k in range(n_levels)]
        neg_true = [[neg_false[k][j] - (neg_false[k - 1][j] if k else 0)
                     for j in range(width)] for k in range(n_levels)]
        spike = [sum(ladder[k][j] * (pos_true[k][j] - neg_true[k][j])
                     for k in range(n_levels)) for j in range(width)]
        membrane = [mem[j] - spike[j] for j in range(width)]
        cumulative = [cumulative[j] + spike[j] for j in range(width)]
        exp_in = [state.exp_in[j] + (x[j] - state.bias[j]) / t for j in range(width)]
        exp_out = [state.exp_out[j] + spike[j] / t for j in range(width)]

        state = State(state.bias, exp_in, exp_out, membrane, cumulative, t)
        traces.append({"output": spike, "membrane": membrane, "cumulative_out": cumulative,
                       "exp_in": exp_in, "exp_out": exp_out,
                       "positive_events": sum(v > 0 for v in spike),
                       "negative_events": sum(v < 0 for v in spike)})
    return traces


def load_author_module(author_repo: Path):
    commit = subprocess.check_output(["git", "-C", str(author_repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected pinned author commit {PINNED_COMMIT}; found {commit}")
    sys.path.insert(0, str(author_repo.resolve()))
    from neurons.relu_neuron import MTH_diff  # type: ignore[import-not-found]
    import torch
    if torch.cuda.is_available():
        # CPU is enforced below regardless of CUDA presence.
        pass
    return torch, MTH_diff, commit


def author_source_trace(torch, mth_cls, frames: Sequence[Sequence[float]], theta: Sequence[float], n_levels: int):
    module = mth_cls(T=0, thresh=torch.tensor(theta, dtype=torch.float32, device="cpu"),
                     step_mode="s", num_thresholds=n_levels).cpu()
    trace = []
    with torch.no_grad():
        module(torch.tensor(frames[0], dtype=torch.float32, device="cpu"))  # source T=-1 init call
        for frame in frames[1:]:
            out = module(torch.tensor(frame, dtype=torch.float32, device="cpu"))
            trace.append({
                "output": out.detach().cpu().tolist(),
                "membrane": module.mem.detach().cpu().tolist(),
                "cumulative_out": module.cumulative_out.detach().cpu().tolist(),
                "exp_in": module.exp_in.detach().cpu().tolist(),
                "exp_out": module.exp_out.detach().cpu().tolist(),
            })
    return trace


def _assert_trace_equal(name: str, got: Sequence[Dict], ref: Sequence[Dict], atol: float = 3e-6) -> Dict:
    if len(got) != len(ref):
        raise AssertionError(f"{name}: tick count {len(got)} != {len(ref)}")
    fields = ("output", "membrane", "cumulative_out", "exp_in", "exp_out")
    max_abs = {field: 0.0 for field in fields}
    event_counts = {"positive": 0, "negative": 0, "zero": 0}
    for tick, (a, b) in enumerate(zip(got, ref), start=1):
        for field in fields:
            if len(a[field]) != len(b[field]):
                raise AssertionError(f"{name} tick {tick} {field} width mismatch")
            err = max((abs(float(x) - float(y)) for x, y in zip(a[field], b[field])), default=0.0)
            max_abs[field] = max(max_abs[field], err)
            if err > atol:
                raise AssertionError(f"{name} tick {tick} {field}: max error {err} > {atol}")
        for x in a["output"]:
            event_counts["positive" if x > 0 else "negative" if x < 0 else "zero"] += 1
    return {"ticks_compared": len(got), "max_abs_error_by_state": max_abs,
            "signed_event_counts": event_counts}


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or any(len(row) != len(b) for row in a):
        raise ValueError("matrix shape mismatch")
    cols = len(b[0])
    if any(len(row) != cols for row in b):
        raise ValueError("ragged matrix")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(cols)] for i in range(len(a))]


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("residual branch shapes differ")
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def _flatten(x: Matrix) -> Vector:
    return [float(v) for row in x for v in row]


def _unflatten(x: Sequence[float], rows: int, cols: int) -> Matrix:
    if len(x) != rows * cols:
        raise ValueError("flat width mismatch")
    return [list(map(float, x[i * cols:(i + 1) * cols])) for i in range(rows)]


def fixed_gcn_mth_unit(
    torch,
    mth_cls,
    input_frames: Sequence[Matrix],
    adjacency: Matrix,
    weight: Matrix,
    *,
    theta_in: Sequence[float],
    theta_out: Sequence[float],
    n_levels: int,
    residual_skip: bool = False,
    linear_bias_first_current: Optional[Sequence[float]] = None,
    bn_eval_scale: Optional[Sequence[float]] = None,
    bn_eval_shift: Optional[Sequence[float]] = None,
) -> dict:
    """Two MTH_diff nonlinear units around fixed P@H@W, optional skip.

    Input frame 0 is init/baseline (no event); later frames are current slices.
    P and W are fixed over the sequence. Bias is explicitly first-current only.
    BN policy is explicit: pass eval scale/shift to fold into the fixed affine
    operator, with shift mapped once to first current. No dynamic edge gates.
    """
    if len(input_frames) < 2:
        raise ValueError("need baseline plus at least one active frame")
    n_nodes = len(input_frames[0])
    in_width = len(input_frames[0][0])
    out_width = len(weight[0]) if weight else 0
    if any(len(x) != n_nodes or any(len(row) != in_width for row in x) for x in input_frames):
        raise ValueError("input frame shapes differ")
    if len(adjacency) != n_nodes or any(len(row) != n_nodes for row in adjacency):
        raise ValueError("adjacency must be a fixed square node operator")
    if len(weight) != in_width or len(theta_in) != in_width or len(theta_out) != out_width:
        raise ValueError("weight/threshold shapes mismatch")
    if residual_skip and out_width != in_width:
        raise ValueError("identity residual requires equal feature widths")
    if (bn_eval_scale is None) != (bn_eval_shift is None):
        raise ValueError("BN fold requires both eval scale and shift")
    if bn_eval_scale is not None and (len(bn_eval_scale) != out_width or len(bn_eval_shift) != out_width):
        raise ValueError("BN affine vectors must match output channels")
    if linear_bias_first_current is not None and len(linear_bias_first_current) != out_width:
        raise ValueError("first-current bias width mismatch")

    # Layer 1 MTH_diff, thresholds shared by channel and tiled over nodes.
    flat_in = [_flatten(x) for x in input_frames]
    theta1 = [float(theta_in[c]) for _node in range(n_nodes) for c in range(in_width)]
    layer1_source = author_source_trace(torch, mth_cls, flat_in, theta1, n_levels)
    layer1_port = source_exact_stepper(flat_in, theta1, n_levels)
    layer1_match = _assert_trace_equal("layer1", layer1_port, layer1_source)
    spikes1 = [_unflatten(s["output"], n_nodes, in_width) for s in layer1_source]

    # Fixed graph convolution current. Eval BN scale folds into W; its shift,
    # ordinary linear bias, or both are injected only into the first current.
    currents = []
    for tick, h in enumerate(spikes1):
        current = matmul(matmul(adjacency, h), weight)
        if residual_skip:
            current = mat_add(current, h)
        if bn_eval_scale is not None:
            current = [[current[i][j] * float(bn_eval_scale[j]) for j in range(out_width)]
                       for i in range(n_nodes)]
        if tick == 0:
            first_bias = list(linear_bias_first_current or [0.0] * out_width)
            if bn_eval_shift is not None:
                first_bias = [first_bias[j] + float(bn_eval_shift[j]) for j in range(out_width)]
            current = [[current[i][j] + first_bias[j] for j in range(out_width)] for i in range(n_nodes)]
        currents.append(current)

    # The downstream source unit gets its own zero baseline and the mapped
    # graph-current frames. This matches the leading init call in source code.
    zero_baseline = [[0.0] * out_width for _ in range(n_nodes)]
    downstream_frames = [_flatten(zero_baseline)] + [_flatten(x) for x in currents]
    theta2 = [float(theta_out[c]) for _node in range(n_nodes) for c in range(out_width)]
    layer2_source = author_source_trace(torch, mth_cls, downstream_frames, theta2, n_levels)
    layer2_port = source_exact_stepper(downstream_frames, theta2, n_levels)
    layer2_match = _assert_trace_equal("layer2", layer2_port, layer2_source)

    return {"layer1_source_match": layer1_match, "layer2_source_match": layer2_match,
            "layer1_spikes": spikes1,
            "graph_current": currents,
            "layer2_spikes": [_unflatten(s["output"], n_nodes, out_width) for s in layer2_source],
            "layer2_states": layer2_source}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path, default=Path(os.environ["DCGS_AUTHOR_REPO"])
                        if os.environ.get("DCGS_AUTHOR_REPO") else None,
                        help="clean h-z-h-cell/ANN-to-SNN-DCGS checkout at pinned commit")
    args = parser.parse_args()
    if args.author_repo is None:
        parser.error("provide --author-repo or DCGS_AUTHOR_REPO; do not substitute another commit")
    torch, mth_cls, commit = load_author_module(args.author_repo)

    theta = [1.0, 0.75]
    cases = {
        "zero": ([[0.0, 0.0]] + [[0.0, 0.0] for _ in range(8)], theta),
        "static_signed_channels": ([[0.35, -0.2]] + [[0.35, -0.2] for _ in range(8)], theta),
        "dynamic_cross_zero": ([[0.0, 0.0], [0.9, 0.6], [0.9, 0.6], [0.1, -0.8],
                                [-0.9, 0.4], [-0.4, -0.8], [0.2, 0.0], [0.6, -0.1]], theta),
        "abrupt_drop_then_negative": ([[0.0, 0.0], [1.1, 0.8], [1.1, 0.8], [0.0, 0.0],
                                        [0.0, 0.0], [-1.0, -0.7], [-1.0, -0.7], [0.2, 0.1]], theta),
    }
    case_results = {}
    for name, (frames, th) in cases.items():
        ref = author_source_trace(torch, mth_cls, frames, th, n_levels=3)
        got = source_exact_stepper(frames, th, n_levels=3)
        case_results[name] = _assert_trace_equal(name, got, ref)

    # Two-layer nonlinear graph block: MTH_diff -> fixed directed P@H@W
    # (plus optional identity residual) -> MTH_diff. All operations CPU-only.
    graph_frames = [
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # baseline init frame
        [[0.9, 0.6], [0.2, -0.4], [-0.6, 0.8]],
        [[0.9, 0.6], [0.2, -0.4], [-0.6, 0.8]],
        [[0.1, -0.8], [-0.9, 0.4], [0.2, 0.0]],  # abrupt/sign changes
        [[-0.9, 0.4], [-0.4, -0.8], [0.6, -0.1]],
        [[0.2, 0.0], [0.5, -0.1], [-0.2, 0.3]],
    ]
    p = [[0.5, 0.5, 0.0], [0.0, 0.25, 0.75], [1.0, 0.0, 0.0]]  # fixed directed P
    w = [[1.0, -0.25], [0.3, 0.8]]
    block_results = {}
    for residual in (False, True):
        key = "fixed_directed_gcn_residual" if residual else "fixed_directed_gcn"
        block = fixed_gcn_mth_unit(torch, mth_cls, graph_frames, p, w,
                                   theta_in=[1.0, 0.75], theta_out=[1.25, 0.9],
                                   n_levels=3, residual_skip=residual,
                                   linear_bias_first_current=[0.03, -0.02],
                                   bn_eval_scale=[0.9, 1.1], bn_eval_shift=[0.04, -0.03])
        block_results[key] = {"layer1": block["layer1_source_match"],
                              "layer2": block["layer2_source_match"],
                              "active_ticks": len(block["layer2_spikes"])}

    print(json.dumps({"pinned_author_commit": commit,
                      "torch": torch.__version__, "device": "cpu",
                      "source_equivalence_cases": case_results,
                      "fixed_operator_graph_blocks": block_results,
                      "scope": "source step-mode MTH_diff equivalence only; synthetic sequences"}, indent=2))


if __name__ == "__main__":
    main()
