import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prefix_codec import prefix_alternative, prefix_bidirectional, _SOURCE_MTH_STEP
from run_objectives import core


def _state(theta, offset=0.0):
    return core._new_site_state(theta, theta * offset, (1, theta.numel()),
                                theta.device, theta.dtype)


def test_hand_example_from_real_source_step():
    theta = torch.tensor([1.0])
    state = _state(theta)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[0.8]]), theta, 1)
    assert torch.equal(q1, torch.tensor([[1.0]]))
    alt, eligible, diag = prefix_alternative(state, q1, theta)
    assert eligible.item()
    assert torch.equal(alt, torch.tensor([[0.5]]))
    assert torch.equal(diag["base_q2"], torch.tensor([[-0.5]]))
    assert torch.equal(diag["alternative_q2"], torch.tensor([[0.5]]))
    assert torch.equal(alt + diag["selected_q2"] / 2, torch.tensor([[0.75]]))
    assert torch.equal(q1 + diag["base_q2"] / 2, torch.tensor([[0.75]]))


def test_random_real_state_exact_prefix_support_budget_and_no_mutation():
    torch.manual_seed(917)
    theta = torch.tensor([1.0, 0.75, 1.5, 0.5])
    offset = torch.empty((1, 4)).uniform_(-0.1, 0.1)
    state = _state(theta, 0.0)
    state["mem"].copy_(theta * offset)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[0.79, 0.61, 1.21, 0.39]]), theta, 1)
    original = {key: value.clone() for key, value in state.items()}
    alt, eligible, diag = prefix_alternative(state, q1, theta)
    assert all(torch.equal(state[key], value) for key, value in original.items())
    assert torch.equal(alt != 0, q1 != 0)
    assert torch.equal(alt[eligible], q1[eligible] / 2)
    assert torch.equal(diag["selected_q2"][eligible], diag["alternative_q2"][eligible])
    assert torch.equal(q1[eligible] + diag["base_q2"][eligible] / 2,
                       alt[eligible] + diag["selected_q2"][eligible] / 2)
    old_events = (q1 != 0).to(torch.int64) + (diag["base_q2"] != 0).to(torch.int64)
    new_events = (alt != 0).to(torch.int64) + (diag["selected_q2"] != 0).to(torch.int64)
    old_amp = q1.abs() + diag["base_q2"].abs()
    new_amp = alt.abs() + diag["selected_q2"].abs()
    assert torch.all(new_events <= old_events)
    assert torch.all(new_amp <= old_amp)
    assert torch.allclose(diag["max_decode_error"], torch.tensor(0.0))


def test_minimum_level_and_real_threshold_boundaries():
    theta = torch.tensor([1.0])
    # At the lowest source level, halving falls outside the original alphabet.
    state = _state(theta)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[1 / 128]]), theta, 1)
    assert q1.item() == 1 / 128
    _, eligible, _ = prefix_alternative(state, q1, theta)
    assert not eligible.item()

    # Values immediately around the 0.75 threshold use the source's actual
    # floating point comparisons, with no epsilon in the eligibility rule.
    for z, expected in ((0.75, True), (0.74999994, False), (0.8125, True), (0.81250006, False)):
        state = _state(theta)
        q1 = _SOURCE_MTH_STEP(state, torch.tensor([[z]]), theta, 1)
        _, eligible, diag = prefix_alternative(state, q1, theta)
        assert eligible.item() is expected, (z, q1.item(), diag["base_q2"].item(), diag["alternative_q2"].item())


def test_state_recurrence_has_two_delta_membrane_shift_and_original_state_is_untouched():
    theta = torch.tensor([1.0])
    state = _state(theta)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[0.8]]), theta, 1)
    alt = {key: value.clone() for key, value in state.items()}
    from intervention import replace_event
    replace_event(alt, q1, q1 / 2, 1)
    assert torch.equal(alt["exp_in"], state["exp_in"])
    torch.testing.assert_close(alt["mem"] - state["mem"], q1 / 2, rtol=0, atol=0)
    torch.testing.assert_close(alt["exp_out"] - state["exp_out"], -q1 / 2, rtol=0, atol=0)
    base_mem2 = state["mem"] + state["exp_in"] - state["exp_out"]
    alt_mem2 = alt["mem"] + alt["exp_in"] - alt["exp_out"]
    torch.testing.assert_close(alt_mem2 - base_mem2, q1, rtol=0, atol=0)


def test_captured_source_step_is_frozen_implementation():
    assert _SOURCE_MTH_STEP is core._mth_step
    assert Path(core.__file__).name == "calibration_core.py"
    assert callable(_SOURCE_MTH_STEP)


def test_double_hand_example_from_real_source_step_and_amplitude_increase():
    theta = torch.tensor([1.0])
    state = _state(theta)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[0.72]]), theta, 1)
    assert torch.equal(q1, torch.tensor([[0.5]]))
    alt, eligible, diag = prefix_alternative(state, q1, theta, direction="double")
    assert eligible.item()
    assert torch.equal(alt, torch.tensor([[1.0]]))
    assert torch.equal(diag["base_q2"], torch.tensor([[0.5]]))
    assert torch.equal(diag["selected_q2"], torch.tensor([[-0.5]]))
    assert torch.equal(q1 + diag["base_q2"] / 2, torch.tensor([[0.75]]))
    assert torch.equal(alt + diag["selected_q2"] / 2, torch.tensor([[0.75]]))
    assert diag["new_local_amplitude_sum"] > diag["old_local_amplitude_sum"]


def test_bidirectional_exact_prefix_and_equal_two_tick_event_count():
    theta = torch.tensor([1.0, 1.0, 1.0, 1.0])
    state = _state(theta)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[0.8, 0.72, 0.61, 0.0]]), theta, 1)
    original = {key: value.clone() for key, value in state.items()}
    alt, eligible, diag = prefix_bidirectional(state, q1, theta)
    assert all(torch.equal(state[key], value) for key, value in original.items())
    assert not torch.any(diag["eligible_halve"] & diag["eligible_double"])
    assert torch.equal(eligible, diag["eligible_halve"] | diag["eligible_double"])
    assert torch.equal(alt[diag["eligible_halve"]], q1[diag["eligible_halve"]] / 2)
    assert torch.equal(alt[diag["eligible_double"]], q1[diag["eligible_double"]] * 2)
    old_q2 = diag["base_q2"]
    assert torch.equal(q1 + old_q2 / 2, alt + diag["selected_q2"] / 2)
    old_count = (q1 != 0).to(torch.int64) + (old_q2 != 0).to(torch.int64)
    new_count = (alt != 0).to(torch.int64) + (diag["selected_q2"] != 0).to(torch.int64)
    assert torch.equal(old_count, new_count)
    old_amp = q1.abs() + old_q2.abs()
    new_amp = alt.abs() + diag["selected_q2"].abs()
    assert new_amp[diag["eligible_double"]].sum() > old_amp[diag["eligible_double"]].sum()
    assert new_amp[diag["eligible_halve"]].sum() < old_amp[diag["eligible_halve"]].sum()


def test_direction_validation_and_zero_event_fallback():
    theta = torch.tensor([1.0])
    state = _state(theta)
    q1 = _SOURCE_MTH_STEP(state, torch.tensor([[0.0]]), theta, 1)
    alt, eligible, _ = prefix_alternative(state, q1, theta, direction="double")
    assert torch.equal(alt, q1)
    assert not eligible.any()
    try:
        prefix_alternative(state, q1, theta, direction="up")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown direction was accepted")
