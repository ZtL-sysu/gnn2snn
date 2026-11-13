import torch
from torch_geometric.data import Batch, Data

from gated_frozen_gate_prototype import (
    _quantize_activation,
    extract_teacher_gates,
    frozen_gate_forward,
    target_normalize_gates,
)
from gated_teacher_model import GatedTeacher


def _synthetic_batch():
    # Directed and deliberately asymmetric: 0->1 and 2->1, but no reverses.
    return Batch.from_data_list([Data(
        x=torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.3], [0.3, 0.5, 0.9]]),
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]]),
        edge_index=torch.tensor([[0, 2, 1], [1, 1, 0]]),
        edge_attr=torch.tensor([[0.1], [0.7], [0.4]]),
    )])


def test_unquantized_frozen_gate_path_matches_teacher_without_labels():
    torch.manual_seed(9)
    model = GatedTeacher(node_dim=5, edge_dim=1, hidden_dim=4, out_dim=4,
                         n_classes=3, layers=2, batch_norm=True, residual=True).eval()
    batch = _synthetic_batch()
    assert batch.y is None
    with torch.no_grad():
        expected = model(batch)
        controller = extract_teacher_gates(model, batch)
        actual, diagnostics = frozen_gate_forward(model, batch, controller)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert diagnostics["quantized"] is False
    assert len(controller.gates) == len(controller.edge_states) == 2


def test_target_normalization_respects_directed_target_and_channels():
    gates = torch.tensor([[0.2, 0.4], [0.8, 0.2], [0.6, 0.3]])
    dst = torch.tensor([1, 1, 0])
    alpha = target_normalize_gates(gates, dst, num_nodes=3, eps=0.0)
    expected = torch.tensor([[0.2, 2 / 3], [0.8, 1 / 3], [1.0, 1.0]])
    torch.testing.assert_close(alpha, expected)
    # Incoming weights sum to one separately for target 1's two edges/channels.
    torch.testing.assert_close(alpha[:2].sum(dim=0), torch.ones(2))
    # There is no implicit reverse edge: target 0 has only the third row.
    torch.testing.assert_close(alpha[2], torch.ones(2))


def test_single_site_count_quantization_error_and_saturation():
    x = torch.tensor([[0.35, 1.25]])
    theta = torch.tensor([1.0, 2.0])
    quantized, counts = _quantize_activation(x, theta, T=4)
    torch.testing.assert_close(counts, torch.tensor([[1.0, 2.0]]))
    assert torch.all(torch.abs(quantized - x) <= theta / (2 * 4) + 1e-7)
    saturated, saturated_counts = _quantize_activation(torch.tensor([[1.4]]), [1.0], T=4)
    torch.testing.assert_close(saturated_counts, torch.tensor([[4.0]]))
    torch.testing.assert_close(saturated, torch.tensor([[1.0]]))


def test_staged_path_quantizes_each_activation_site_and_keeps_analog_logits():
    torch.manual_seed(23)
    model = GatedTeacher(node_dim=5, edge_dim=1, hidden_dim=4, out_dim=4,
                         n_classes=3, layers=2, batch_norm=True, residual=True).eval()
    batch = _synthetic_batch()
    controller = extract_teacher_gates(model, batch)
    thresholds = [torch.ones(4) for _ in range(2)] + [torch.ones(2), torch.ones(1)]
    with torch.no_grad():
        exact, _ = frozen_gate_forward(model, batch, controller)
        staged, diagnostics = frozen_gate_forward(model, batch, controller, thresholds, T=2)
    assert staged.shape == exact.shape == (1, 3)
    assert diagnostics["quantized"] is True
    assert len(diagnostics["sites"]) == 4
    for site, width in zip(diagnostics["sites"], (4, 4, 2, 1)):
        assert site["counts"].shape[-1] == width
        assert bool((site["counts"] >= 0).all()) and bool((site["counts"] <= 2).all())
        assert site["spikes"] == int(site["counts"].sum().item())
    assert not torch.allclose(staged, exact)
