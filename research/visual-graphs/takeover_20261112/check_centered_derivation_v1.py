"""Float64 algebra/source checks only; no official data or accuracy experiment."""
import hashlib
import json
from pathlib import Path
import torch
from dcgs_fixed_residual_gcn_event_forward_v1 import require_author_source

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    torch.manual_seed(717)
    unit, _, provenance = require_author_source(Path(r'C:\Users\xhf\AppData\Local\Temp\dcgs-5e0e4c73'))
    dtype = torch.float64
    theta = torch.logspace(-3, 3, 7, dtype=dtype)
    neuron = unit(T=0, thresh=theta, step_mode='s', num_thresholds=8).cpu()
    shape = (13, 7)
    assert not neuron(torch.zeros(shape, dtype=dtype)).any() and neuron.T == 0
    previous = torch.zeros(shape, dtype=dtype)
    decoded = torch.zeros_like(previous)
    base = torch.randn_like(previous)
    W = torch.randn(5, 7, dtype=dtype)
    bias = torch.randn(5, dtype=dtype)
    projected = base @ W.T + bias
    maxima = dict(membrane=0.0, cumulative=0.0, input=0.0, projection=0.0)
    minimum_rate = 0.0
    for t in range(1, 9):
        # Abrupt signed currents and disparate channel scales stress the source identity.
        target = torch.randn(shape, dtype=dtype) * theta * (2 if t % 2 else .2)
        event = neuron(t * (target - previous))
        previous = target
        decoded = decoded + event / t
        projected = projected + (event / t) @ W.T
        errors = {'membrane': (neuron.mem - t * (target - decoded)).abs().max(),
                  'cumulative': (neuron.cumulative_out - t * decoded).abs().max(),
                  'input': (neuron.exp_in - target).abs().max(),
                  'projection': (projected - ((base + decoded) @ W.T + bias)).abs().max()}
        for k, value in errors.items():
            maxima[k] = max(maxima[k], float(value))
        minimum_rate = min(minimum_rate, float(decoded.min()))
    assert max(maxima.values()) < 1e-10 and minimum_rate >= -1e-10

    # Nonlinear own-state GatedGCN functions, independent of a benchmark forward.
    n, d, L = 5, 7, 4
    src = torch.tensor([0, 1, 0, 2, 3, 3])
    dst = torch.tensor([1, 2, 2, 1, 4, 3])
    E = src.numel()
    h0, e0 = torch.randn(n, d, dtype=dtype), torch.randn(E, d, dtype=dtype)
    layers = [{k: (torch.randn(d, d, dtype=dtype)*.15, torch.randn(d, dtype=dtype)*.1)
               for k in ('A', 'B', 'C', 'D', 'E')} for _ in range(L)]
    node_bn = [(torch.randn(d, dtype=dtype), torch.randn(d, dtype=dtype)) for _ in range(L)]
    edge_bn = [(torch.randn(d, dtype=dtype), torch.randn(d, dtype=dtype)) for _ in range(L)]

    def branches(h, e, layer, i):
        def linear(k, x):
            w, b = layer[k]
            return x @ w.T + b
        z = linear('C', e) + linear('D', h)[src] + linear('E', h)[dst]
        g = torch.sigmoid(z)
        num, den = torch.zeros_like(h), torch.zeros_like(h)
        num.index_add_(0, dst, g * linear('B', h)[src])
        den.index_add_(0, dst, g)
        j = linear('A', h) + num / (den + 1e-6)
        return torch.relu(j * node_bn[i][0] + node_bn[i][1]), torch.relu(z * edge_bn[i][0] + edge_bn[i][1])

    h, e, q, r = h0.clone(), e0.clone(), torch.zeros_like(h0), torch.zeros_like(e0)
    centered_error, negative_count = 0.0, 0
    for i, layer in enumerate(layers):
        bh, be = branches(h, e, layer, i)
        h, e = h + bh, e + be
        ch, ce = branches(h0 + q, e0 + r, layer, i)
        q, r = torch.relu(q + ch), torch.relu(r + ce)
        negative_count += int((q < 0).sum() + (r < 0).sum())
        centered_error = max(centered_error, float((h - (h0 + q)).abs().max()),
                             float((e - (e0 + r)).abs().max()))
    assert centered_error < 1e-10 and negative_count == 0
    # Explicit boundaries: signed updates and a doubled skip invalidate the naive transform.
    assert -1.0 != max(0.0, -1.0)
    assert 2.0 * 1.0 + 2.0 != 1.0 + 2.0
    report = {'status': 'passed', 'scope': 'synthetic_float64_representation_and_source_identities_only',
              'official_data_loaded': False, 'T': 8, 'layers': L,
              'source': provenance, 'source_state_max_abs_errors': maxima,
              'minimum_decoded_rate': minimum_rate, 'centered_gated_layer_max_abs_error': centered_error,
              'signed_update_and_nonidentity_skip_counterexamples': True,
              'accuracy_energy_novelty_claim': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'derivation_sha256': hashlib.sha256((HERE.parent/'DERIVATION_PACKAGE.md').read_bytes()).hexdigest()}
    output = HERE / 'centered_math_v1.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
