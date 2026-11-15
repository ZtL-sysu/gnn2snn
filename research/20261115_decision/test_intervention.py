import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from intervention import replace_event, adjacent_amplitude


def test_replacement_updates_all_recurrent_states():
    original = torch.tensor([[1., 0.]])
    replacement = torch.tensor([[.5, 0.]])
    state = {k: torch.tensor([[3., 4.]]) for k in ('mem', 'cumulative_out', 'exp_out')}
    replace_event(state, original, replacement, 2)
    assert torch.equal(state['mem'], torch.tensor([[3.5, 4.]]))
    assert torch.equal(state['cumulative_out'], torch.tensor([[2.5, 4.]]))
    assert torch.equal(state['exp_out'], torch.tensor([[2.75, 4.]]))


def test_adjacent_original_alphabet_preserves_nonzero_and_bounds():
    theta = torch.tensor([1., 2., 1.])
    event = torch.tensor([[1., 1., 1/128]])
    up = adjacent_amplitude(event, theta, 1)
    down = adjacent_amplitude(event, theta, -1)
    assert torch.equal(up, torch.tensor([[1., 2., 1/64]]))
    assert torch.equal(down, torch.tensor([[.5, .5, 1/128]]))
    assert torch.equal(up != 0, event != 0)
    assert torch.equal(down != 0, event != 0)
