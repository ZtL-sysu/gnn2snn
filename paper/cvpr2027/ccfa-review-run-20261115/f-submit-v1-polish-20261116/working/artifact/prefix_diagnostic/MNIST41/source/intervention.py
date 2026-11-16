"""Diagnostic hard-event intervention, preserving all affected recurrent state."""
import torch


def replace_event(state, original, replacement, t):
    delta = replacement - original
    state['mem'] = state['mem'] - delta
    state['cumulative_out'] = state['cumulative_out'] + delta
    state['exp_out'] = state['exp_out'] + delta / float(t)


def adjacent_amplitude(event, theta, direction):
    if direction not in (-1, 1):
        raise ValueError('direction must be -1 or +1')
    magnitude = event.abs()
    value = (magnitude * (2. if direction == 1 else .5)).clamp(min=theta / 128, max=theta)
    return torch.where(event != 0, value * event.sign(), event)
