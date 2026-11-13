"""CPU synthetic checks for the shared-consumer affine-base reference modes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch_geometric.data import Batch, Data

HERE = Path(__file__).resolve().parent
DEFAULT_AUTHOR_REPO = Path(r"C:\Users\xhf\AppData\Local\Temp\dcgs-5e0e4c73")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakePrefixCoder:
    def __init__(self, reference, mode):
        self.mode = mode
        self.T = 0
        self.calls = 0
        self.reference = reference

    def step(self, current, tick):
        self.calls += 1
        self.T = tick
        if self.mode == "identity":
            return current.clone(), torch.zeros_like(current)
        if self.mode == "zero":
            return torch.zeros_like(current), torch.zeros_like(current)
        raise AssertionError(self.mode)


def _factory(mode):
    return lambda reference, threshold, neuron_class, site: FakePrefixCoder(reference, mode)


def _batch(with_label=True):
    graph = Data(
        x=torch.tensor([[0.2, 0.4], [0.8, -0.1], [0.3, 0.7], [0.1, 0.2]], dtype=torch.float32),
        pos=torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.6, 0.3]], dtype=torch.float32),
        edge_attr=torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long))
    if with_label:
        graph.y = torch.tensor([3])
    return Batch.from_data_list([graph])


def _thresholds(module, mode, model):
    sites = module.site_specs(mode, model)
    return [torch.full((width,), 0.35, dtype=torch.float32) for _, width, _ in sites]


def _check_source_prefix_decoder(impl, neuron_class):
    threshold = torch.tensor([0.31, 0.27, 0.22], dtype=torch.float32)
    series = [torch.tensor([[0.2, 0.1, 0.0], [0.4, 0.3, 0.1]]),
              torch.tensor([[0.3, 0.0, 0.2], [0.5, 0.1, 0.0]]),
              torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.0, 0.4]]),
              torch.tensor([[0.4, 0.2, 0.1], [0.0, 0.2, 0.3]])]
    encoder = None
    expected_module = None
    previous = torch.zeros_like(series[0])
    expected = torch.zeros_like(previous)
    for tick, current in enumerate(series, 1):
        if encoder is None:
            encoder = impl.SourcePrefixEncoder(current, threshold, neuron_class, "prefix-check")
            expected_module = neuron_class(T=0, thresh=threshold, step_mode="s", num_thresholds=8)
            warm = expected_module(torch.zeros_like(current))
            if bool(warm.any()) or int(expected_module.T) != 0:
                raise AssertionError("direct source zero latch changed")
        source_input = tick * (current - previous)
        event = expected_module(source_input)
        expected = expected + event / float(tick)
        decoded, observed_event = encoder.step(current, tick)
        if not torch.equal(observed_event, event) or not torch.equal(decoded, expected):
            raise AssertionError("source prefix delta/Σevent/t decoder mismatch")
        previous = current
    if encoder.active_calls != len(series) or int(encoder.module.T) != len(series):
        raise AssertionError("source prefix sequence did not reset/count active ticks")


def _check_zero_consumer_closed_form(impl, model, batch, mode, trace):
    base_h, base_e = impl._raw_embeddings(model, batch)
    expected_h, expected_e = base_h.clone(), base_e.clone()
    invalid_h, invalid_e = None, None
    for i, layer in enumerate(model.layers):
        if mode == "generic":
            consumer_h = torch.zeros_like(base_h)
            consumer_e = torch.zeros_like(base_e)
        elif mode == "first_analog" and i == 0:
            consumer_h, consumer_e = base_h, base_e
        elif mode == "centered":
            consumer_h, consumer_e = base_h, base_e
        else:
            consumer_h = torch.zeros_like(base_h)
            consumer_e = torch.zeros_like(base_e)
        branch_h, branch_e = impl._graph_layer(layer, consumer_h, consumer_e,
                                               batch.edge_index)
        expected_h = expected_h + branch_h
        expected_e = expected_e + branch_e
        # Deliberately wrong decoded-state skip alternatives must be separated
        # from the raw-prefix closed form by this synthetic fixture.
        if i == 0:
            invalid_h, invalid_e = consumer_h + branch_h, consumer_e + branch_e
        else:
            invalid_h, invalid_e = consumer_h + branch_h, consumer_e + branch_e
    last = trace[0]["layers"][-1]
    if not torch.allclose(last["raw_node_output"], expected_h, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero-consumer node raw skip differs from base+sum(branches)")
    if not torch.allclose(last["raw_edge_output"], expected_e, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero-consumer edge raw skip differs from base+sum(branches)")
    if torch.allclose(invalid_h, expected_h, atol=2e-6, rtol=2e-6):
        raise AssertionError("fixture cannot catch decoded-state node skip substitution")
    if torch.allclose(invalid_e, expected_e, atol=2e-6, rtol=2e-6):
        raise AssertionError("fixture cannot catch decoded-state edge skip substitution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    impl = _load_module("gated_affine_base_v1_under_test", HERE / "gated_affine_base_v1.py")
    from gated_teacher_model import GatedTeacher

    MTH_diff, MTH_diff_line, provenance = impl.require_author_sources(args.author_repo)
    if provenance["commit"] != impl.PINNED_COMMIT:
        raise AssertionError("unexpected source commit")
    torch.manual_seed(919)
    batch = _batch(with_label=False)
    model = GatedTeacher(node_dim=4, edge_dim=1, hidden_dim=8, out_dim=8,
                         n_classes=3, layers=4, dropout=0.0, batch_norm=True,
                         residual=True, readout="mean").eval()
    with torch.no_grad():
        teacher_logits = model(batch)

    def forbidden_forward(*_a, **_kw):
        raise AssertionError("converted path called the full teacher forward oracle")
    model.forward = forbidden_forward

    identity_results = []
    for mode in ("generic", "first_analog", "centered"):
        for T in (1, 2, 4, 8):
            result = impl.affine_mode_forward(
                model, batch, _thresholds(impl, mode, model), T, mode,
                MTH_diff, MTH_diff_line, capture_trace=True,
                encoder_factory=_factory("identity"))
            if not torch.allclose(result["logits"], teacher_logits, atol=2e-5, rtol=2e-5):
                raise AssertionError("identity-code %s T=%d teacher mismatch" % (mode, T))
            if len(result["tick_logits"]) != T or result["head_calls"] != T:
                raise AssertionError("capture_trace head call accounting mismatch")
            identity_results.append((mode, T))
    # Normal execution evaluates the matched analog head once after final T.
    normal = impl.affine_mode_forward(
        model, batch, _thresholds(impl, "centered", model), 4, "centered",
        MTH_diff, MTH_diff_line, capture_trace=False,
        encoder_factory=_factory("identity"))
    if normal["head_calls"] != 1 or normal["tick_logits"]:
        raise AssertionError("normal head must execute once")

    source_results = []
    for mode in ("generic", "first_analog", "centered"):
        T = 4
        thresholds = _thresholds(impl, mode, model)
        a = impl.affine_mode_forward(model, batch, thresholds, T, mode,
                                     MTH_diff, MTH_diff_line, capture_trace=True)
        b = impl.affine_mode_forward(model, batch, thresholds, T, mode,
                                     MTH_diff, MTH_diff_line, capture_trace=True)
        if not torch.equal(a["logits"], b["logits"]):
            raise AssertionError("fresh-run source reset mismatch for %s" % mode)
        expected_sites = 8 if mode == "generic" else 6
        if a["source_warmup_calls"] != expected_sites or a["active_unit_calls"] != expected_sites * T:
            raise AssertionError("wrong source warmup/active calls for %s" % mode)
        if a["source_ticks"] != [T] * expected_sites:
            raise AssertionError("source active tick counter mismatch for %s" % mode)
        if mode in ("first_analog", "centered"):
            first_layer = a["trace"][0]["layers"][0]
            if not torch.equal(first_layer["consumer_node_input"], first_layer["raw_node_input"]):
                raise AssertionError("first graph layer is not exact/analog in %s" % mode)
        for name, stats in zip(a["site_names"], a["site_stats"]):
            levels = sum(stats["signed_threshold_level_events"].values())
            if levels != stats["positive"] + stats["negative"]:
                raise AssertionError("site event level partition mismatch")
            rows = int(batch.x.shape[0]) if name.endswith(".node") else int(batch.edge_index.shape[1])
            if not 0 <= stats["nonzero_rows"] <= rows * T:
                raise AssertionError("observed nonzero row count outside rows*T bound")
        source_results.append(mode)

    centered_zero = impl.affine_mode_forward(
        model, batch, _thresholds(impl, "centered", model), 1, "centered",
        MTH_diff, MTH_diff_line, capture_trace=True,
        encoder_factory=_factory("zero"))
    layer1 = centered_zero["trace"][0]["layers"][1]
    if torch.allclose(layer1["raw_node_input"], layer1["consumer_node_input"]):
        raise AssertionError("zero coder replaced the RAW centered skip with its decoded input")
    if torch.count_nonzero(layer1["consumer_node_input"] - centered_zero["trace"][0]["base_node"]) != 0:
        raise AssertionError("zero coder consumer input should be the retained base only")
    if any(stats["nonzero_rows"] != 0 for stats in centered_zero["site_stats"]):
        raise AssertionError("zero fake encoder must emit zero nonzero rows")
    for mode in ("generic", "first_analog", "centered"):
        zero_run = impl.affine_mode_forward(
            model, batch, _thresholds(impl, mode, model), 1, mode,
            MTH_diff, MTH_diff_line, capture_trace=True,
            encoder_factory=_factory("zero"))
        _check_zero_consumer_closed_form(impl, model, batch, mode, zero_run["trace"])

    # Prefix-only capture must agree with independent fresh runs at every
    # checkpoint tick and must not add duplicate head calls with full tracing.
    prefix_results = []
    for mode in ("generic", "first_analog", "centered"):
        thresholds = _thresholds(impl, mode, model)
        long = impl.affine_mode_forward(model, batch, thresholds, 8, mode,
                                        MTH_diff, MTH_diff_line,
                                        capture_prefix_stats=True,
                                        capture_prefix_logits=True)
        if set(long["prefix_site_stats"]) != {"1", "2", "4", "8"}:
            raise AssertionError("prefix site stats missing locked tick keys")
        if set(long["prefix_logits"]) != {"1", "2", "4", "8"} or long["head_calls"] != 4:
            raise AssertionError("prefix-only head capture shape/call accounting mismatch")
        if any(value.shape != (int(batch.num_graphs), 3)
               for value in long["prefix_logits"].values()):
            raise AssertionError("prefix logits must remain graph-count by class-count only")
        for prefix, prefix_stats in long["prefix_site_stats"].items():
            for name, stats in zip(long["site_names"], prefix_stats):
                rows = int(batch.x.shape[0]) if name.endswith(".node") else int(batch.edge_index.shape[1])
                if not 0 <= stats["nonzero_rows"] <= rows * int(prefix):
                    raise AssertionError("prefix nonzero row count outside rows*T bound")
        for prefix in (1, 2, 4, 8):
            short = impl.affine_mode_forward(model, batch, thresholds, prefix, mode,
                                             MTH_diff, MTH_diff_line)
            if not torch.allclose(long["prefix_logits"][str(prefix)], short["logits"],
                                  atol=1e-6, rtol=1e-6):
                raise AssertionError("prefix logit mismatch for %s T%d" % (mode, prefix))
            if long["prefix_site_stats"][str(prefix)] != short["site_stats"]:
                raise AssertionError("prefix site stats mismatch for %s T%d" % (mode, prefix))
        # Prefix snapshots are value copies, not aliases of final counters.
        long["site_stats"][0]["nonzero_rows"] += 100000
        if long["prefix_site_stats"]["1"][0]["nonzero_rows"] > int(batch.x.shape[0]):
            raise AssertionError("prefix site stats are not deep-copied")
        both = impl.affine_mode_forward(model, batch, thresholds, 8, mode,
                                        MTH_diff, MTH_diff_line, capture_trace=True,
                                        capture_prefix_stats=True,
                                        capture_prefix_logits=True)
        if both["head_calls"] != 8:
            raise AssertionError("trace+prefix capture duplicated analog head calls")
        if any(not torch.equal(both["prefix_logits"][str(t)], both["tick_logits"][t - 1])
               for t in (1, 2, 4, 8)):
            raise AssertionError("combined trace/prefix logits differ or duplicate")
        prefix_results.append(mode)
        prefix4 = impl.affine_mode_forward(model, batch, thresholds, 4, mode,
                                           MTH_diff, MTH_diff_line,
                                           capture_prefix_logits=True)
        if set(prefix4["prefix_logits"]) != {"1", "2", "4"} or prefix4["head_calls"] != 3:
            raise AssertionError("T4 prefix logits should use exactly three head calls")

    labeled = _batch(with_label=True)
    expect = lambda fn, key: _expect_error(fn, key)
    expect(lambda: impl.affine_mode_forward(model, labeled, _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line), "labels")
    bad = _batch(with_label=False)
    bad.x[0, 0] = float("nan")
    expect(lambda: impl.affine_mode_forward(model, bad, _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line), "finite")
    bad_thresholds = _thresholds(impl, "centered", model)
    bad_thresholds[0] = torch.ones(7)
    expect(lambda: impl.affine_mode_forward(model, batch, bad_thresholds, 1, "centered",
                                            MTH_diff, MTH_diff_line), "threshold")

    _check_source_prefix_decoder(impl, MTH_diff)
    _check_source_prefix_decoder(impl, MTH_diff_line)
    overflow = impl.SourcePrefixEncoder(torch.zeros((1, 2)), torch.tensor([0.2, 0.2]),
                                        MTH_diff_line, "overflow-check")
    overflow.step(torch.full((1, 2), 3.0e38), 1)
    expect(lambda: overflow.step(torch.full((1, 2), -3.0e38), 2), "finite")
    poisoned = impl.SourcePrefixEncoder(torch.zeros((1, 2)), torch.tensor([0.2, 0.2]),
                                        MTH_diff_line, "persistent-state-check")
    poisoned.step(torch.zeros((1, 2)), 1)
    poisoned.module.exp_in[0, 0] = float("inf")
    expect(lambda: poisoned.step(torch.zeros((1, 2)), 2), "persistent state")

    bad_edge_index = _batch(with_label=False)
    bad_edge_index.edge_index = bad_edge_index.edge_index.float()
    expect(lambda: impl.affine_mode_forward(model, bad_edge_index,
                                            _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line),
           "integer dtype")
    bad_edge_index = _batch(with_label=False)
    bad_edge_index.edge_index = bad_edge_index.edge_index.bool()
    expect(lambda: impl.affine_mode_forward(model, bad_edge_index,
                                            _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line),
           "integer dtype")
    bad_edges = _batch(with_label=False)
    bad_edges.edge_attr = torch.zeros((7, 1))
    expect(lambda: impl.affine_mode_forward(model, bad_edges,
                                            _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line),
           "row count")

    saved_node_mean = model.layers[0].bn_node_h.running_mean.clone()
    model.layers[0].bn_node_h.running_mean[0] = float("nan")
    expect(lambda: impl.affine_mode_forward(model, batch,
                                            _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line),
           "post-BN")
    model.layers[0].bn_node_h.running_mean.copy_(saved_node_mean)
    saved_edge_mean = model.layers[0].bn_node_e.running_mean.clone()
    model.layers[0].bn_node_e.running_mean[0] = float("nan")
    expect(lambda: impl.affine_mode_forward(model, batch,
                                            _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line),
           "post-BN")
    model.layers[0].bn_node_e.running_mean.copy_(saved_edge_mean)
    bad_thresholds = _thresholds(impl, "centered", model)
    bad_thresholds[0][0] = float("inf")
    expect(lambda: impl.affine_mode_forward(model, batch, bad_thresholds, 1, "centered",
                                            MTH_diff, MTH_diff_line), "threshold")
    bad_thresholds = _thresholds(impl, "centered", model)
    bad_thresholds[0][0] = float("nan")
    expect(lambda: impl.affine_mode_forward(model, batch, bad_thresholds, 1, "centered",
                                            MTH_diff, MTH_diff_line), "threshold")

    row_stats = impl._site_stats(torch.tensor([[0., 0.], [1., 0.], [0., -1.]]), torch.ones(2))
    assert row_stats["nonzero_rows"] == 2, row_stats

    child_bn = model.layers[0].bn_node_h
    child_bn.train()
    expect(lambda: impl.affine_mode_forward(model, batch, _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line), "eval")
    child_bn.eval()
    original_fc = model.MLP_layer.FC_layers[0]
    model.MLP_layer.FC_layers[0] = torch.nn.Sequential(torch.nn.Dropout(.5), original_fc)
    expect(lambda: impl.affine_mode_forward(model, batch, _thresholds(impl, "centered", model),
                                            1, "centered", MTH_diff, MTH_diff_line), "eval")
    model.MLP_layer.FC_layers[0] = original_fc

    print(json.dumps({"device": "cpu", "official_dataset_loaded": False,
                      "teacher_forward_disabled_during_modes": True,
                      "identity_matches": identity_results,
                      "source_modes_reset_and_partitions": source_results,
                      "source_warmup_sites": {"generic": 8, "first_analog": 6, "centered": 6},
                      "active_calls_per_mode_T4": {"generic": 32, "first_analog": 24, "centered": 24},
                      "zero_event_raw_skip_control": "passed",
                      "closed_form_raw_skip_modes": ["generic", "first_analog", "centered"],
                      "source_prefix_capture_modes": prefix_results,
                      "source_delta_prefix_decoder": "MTH_diff and MTH_diff_line, t*delta and sum(event/t) verified",
                      "negative_checks": ["labels", "NaN input", "threshold shape", "non-finite threshold",
                                          "integer/bool edge indices", "edge row mismatch",
                                          "source delta overflow", "persistent source state",
                                          "post-BN node/edge nonfinite"]}, indent=2))


def _expect_error(fn, key):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        if key.lower() not in str(exc).lower():
            raise AssertionError("unexpected error: %s" % exc) from exc
        return
    raise AssertionError("expected an error containing %r" % key)


if __name__ == "__main__":
    main()
