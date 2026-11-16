"""Exact next-tick prefix replacement diagnostic for the frozen MTH recurrence.

This is a first-site prefix primitive only. It makes no claim about matching
the rest of the graph network after the first active site.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_objectives import core
from intervention import replace_event

# Keep the frozen implementation even when a runner later patches core._mth_step
# to instrument the online forward.
_SOURCE_MTH_STEP = core._mth_step


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()}


@torch.no_grad()
def prefix_alternative(state_after_t1: dict[str, torch.Tensor],
                       event1: torch.Tensor,
                       theta: torch.Tensor,
                       direction: str = "halve") -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Try a same-alphabet half/double q1 replacement with exact t=2 match.

    Inputs are the first site's state immediately after the original t=1
    ``_mth_step``. The source recurrence is queried on cloned states with zero
    current input at t=2. Only coordinates meeting every exact source-level,
    recurrence, finite-value, and decoded-prefix condition are changed.
    """
    if event1.ndim < 2 or theta.ndim != 1 or event1.shape[-1] != theta.numel():
        raise ValueError("event1 must have shape [..., C] and theta shape [C]")
    if not isinstance(event1, torch.Tensor) or not isinstance(theta, torch.Tensor):
        raise TypeError("event1 and theta must be tensors")
    if event1.device != theta.device or event1.dtype != theta.dtype:
        raise ValueError("event1 and theta must share device and dtype")
    if "mem" not in state_after_t1 or state_after_t1["mem"].shape != event1.shape:
        raise ValueError("state_after_t1 mem shape must match event1")
    for key in ("cumulative_out", "exp_in", "exp_out", "threshold_steps"):
        if key not in state_after_t1:
            raise ValueError(f"state_after_t1 missing {key}")

    if direction not in ("halve", "double"):
        raise ValueError("direction must be 'halve' or 'double'")
    candidate = event1 / 2 if direction == "halve" else event1 * 2
    zero = torch.zeros_like(event1)
    base_state = _clone_state(state_after_t1)
    base_q2 = _SOURCE_MTH_STEP(base_state, zero, theta, 2)

    alt_state = _clone_state(state_after_t1)
    replace_event(alt_state, event1, candidate, 1)
    alt_q2 = _SOURCE_MTH_STEP(alt_state, zero, theta, 2)

    levels = theta.unsqueeze(0) / (2 ** torch.arange(8, device=theta.device, dtype=theta.dtype)).unsqueeze(1)
    candidate_is_level = (candidate.unsqueeze(0) == levels.reshape((8,) + (1,) * (event1.ndim - 1) + (theta.numel(),))).any(0)
    finite = torch.isfinite(event1) & torch.isfinite(candidate) & torch.isfinite(base_q2) & torch.isfinite(alt_q2)
    finite = finite & torch.isfinite(theta).reshape((1,) * (event1.ndim - 1) + (theta.numel(),))
    in_range = (candidate >= theta / 128) & (candidate <= theta)
    if direction == "halve":
        expected_base, expected_alternative = -candidate, candidate
    else:
        expected_base, expected_alternative = event1, -event1
    eligible = ((event1 > 0) & candidate_is_level & in_range &
                (base_q2 == expected_base) & (alt_q2 == expected_alternative) & finite)
    old_decode = event1 + base_q2 / 2
    new_decode = candidate + alt_q2 / 2
    eligible = eligible & torch.isfinite(old_decode) & torch.isfinite(new_decode) & (old_decode == new_decode)
    alternative = torch.where(eligible, candidate, event1)
    selected_q2 = torch.where(eligible, alt_q2, base_q2)
    selected_decode = alternative + selected_q2 / 2
    decode_error = (old_decode - selected_decode).abs()
    max_decode_error = decode_error.max() if decode_error.numel() else event1.new_zeros(())
    diagnostics = {
        "base_q2": base_q2,
        "alternative_q2": alt_q2,
        "selected_q2": selected_q2,
        "max_decode_error": max_decode_error,
        "candidate_decode_error": (old_decode - new_decode).abs(),
        "old_local_amplitude_sum": event1.abs().sum(),
        "new_local_amplitude_sum": alternative.abs().sum(),
        "eligible_count": eligible.sum(),
        "eligible_fraction": eligible.to(event1.dtype).mean(),
        "direction_is_double": torch.tensor(direction == "double", device=event1.device),
    }
    return alternative, eligible, diagnostics


@torch.no_grad()
def prefix_bidirectional(state_after_t1: dict[str, torch.Tensor],
                         event1: torch.Tensor,
                         theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Merge mutually exclusive exact halve/double one-step alternatives."""
    halve, eh, dh = prefix_alternative(state_after_t1, event1, theta, direction="halve")
    double, ed, dd = prefix_alternative(state_after_t1, event1, theta, direction="double")
    if torch.any(eh & ed):
        raise RuntimeError("halve and double eligibility overlap")
    eligible = eh | ed
    alternative = torch.where(ed, double, halve)
    selected_q2 = torch.where(ed, dd["selected_q2"], dh["selected_q2"])
    base_q2 = dh["base_q2"]
    old_decode = event1 + base_q2 / 2
    selected_decode = alternative + selected_q2 / 2
    max_decode_error = (old_decode - selected_decode).abs().max()
    diagnostics = {
        "base_q2": base_q2,
        "selected_q2": selected_q2,
        "eligible_halve": eh,
        "eligible_double": ed,
        "max_decode_error": max_decode_error,
        "old_local_amplitude_sum": event1.abs().sum(),
        "new_local_amplitude_sum": alternative.abs().sum(),
        "halve_amplitude_delta": (halve.abs() - event1.abs()).sum(),
        "double_amplitude_delta": (double.abs() - event1.abs()).sum(),
        "eligible_count": eligible.sum(),
        "eligible_fraction": eligible.to(event1.dtype).mean(),
    }
    return alternative, eligible, diagnostics
