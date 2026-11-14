import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '20261114_story'))
from test_calibration_core import TinyFolded, synthetic_batch

from calibration_core import CalibratedGCN
from percentile_core import PercentileGCN, channelwise_quantile_thresholds


def test_eight_level_percentile_baseline_is_bitwise_legacy():
    torch.manual_seed(13)
    batch = synthetic_batch()
    model = TinyFolded().eval()
    thresholds = [[0.2] * 8] * 4 + [[0.2] * 4, [0.2] * 2]
    old = CalibratedGCN(model, thresholds).eval()
    new = PercentileGCN(model, thresholds, num_levels=8).eval()
    for ticks in (1, 2, 4, 8):
        assert torch.equal(old(batch, ticks)[0], new(batch, ticks)[0])


def test_four_level_percentile_baseline_matches_author_mth_diff():
    torch.manual_seed(13)
    batch = synthetic_batch()
    model = TinyFolded().eval()
    thresholds = [[0.2] * 8] * 4 + [[0.2] * 4, [0.2] * 2]
    candidate = PercentileGCN(model, thresholds, num_levels=4).eval()
    # Compare the copied vectorized recurrence against the pinned author unit.
    sys.path.insert(0, 'C:/code/gnn2snn-dcgs-reference')
    from neurons.relu_neuron import MTH_diff
    from percentile_core import source_mth_diff_step

    theta = torch.tensor([0.2, 0.4])
    currents = [torch.tensor([[0.8, 0.1], [-0.3, 0.5]]),
                torch.tensor([[-0.3, 0.5], [0.2, -0.1]]),
                torch.tensor([[0.1, -0.5], [0.4, 0.2]])]
    source = MTH_diff(T=0, thresh=theta, step_mode='s', num_thresholds=4)
    source(torch.zeros_like(currents[0]))
    state = None
    for tick, current in enumerate(currents, 1):
        expected = source(current)
        actual, state = source_mth_diff_step(current, theta, state, tick, 4)
        assert torch.equal(actual, expected)
    assert candidate(batch, 2)[0].shape == (batch.num_graphs, model.mlp[4].out_features)


def test_channelwise_quantile_uses_sample_axis_and_positive_floor():
    sites = [torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]) for _ in range(6)]
    thresholds = channelwise_quantile_thresholds(sites, quantile=0.5, floor=1e-6)
    assert len(thresholds) == 6
    assert torch.equal(thresholds[0], torch.tensor([2.0, 3.0]))
    zeros = [torch.zeros(3, 2) for _ in range(6)]
    assert torch.equal(channelwise_quantile_thresholds(zeros)[0], torch.full((2,), 1e-6))

def test_repeated_evaluation_preserves_cached_labels():
    from run_percentile import evaluation_batch
    batch=synthetic_batch()
    for _ in range(2):
        stripped,labels=evaluation_batch(batch,'cpu')
        assert torch.equal(labels,batch.y.view(-1))
        assert stripped.y is None
