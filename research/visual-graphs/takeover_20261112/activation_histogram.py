"""Streaming, bounded-memory per-channel and global activation histograms."""
import math
import torch


class ActivationHistogram:
    def __init__(self, maxima, bins=512):
        assert maxima.ndim == 1 and bins > 1 and bool((maxima >= 0).all())
        self.maxima = maxima.detach()
        self.bins = bins
        self.global_max = maxima.max()
        self.channel = torch.zeros((maxima.numel(), bins), dtype=torch.int64, device=maxima.device)
        self.global_counts = torch.zeros(bins, dtype=torch.int64, device=maxima.device)
        self.rows = 0

    def update(self, activation):
        if activation.ndim != 2 or activation.shape[1] != self.maxima.numel():
            raise ValueError("Unexpected activation shape")
        if not bool(torch.isfinite(activation).all()) or bool((activation < 0).any()):
            raise ValueError("Expected finite nonnegative activations")
        if bool((activation > self.maxima[None, :] + 1e-5).any()):
            raise ValueError("Second-pass activation exceeds first-pass bound")
        values = activation.detach()
        idx = torch.floor(values / self.maxima.clamp(min=1e-12) * self.bins).long().clamp(0, self.bins-1)
        offsets = torch.arange(self.maxima.numel(), device=values.device) * self.bins
        counts = torch.bincount((idx + offsets).reshape(-1), minlength=self.maxima.numel()*self.bins)
        self.channel += counts.reshape_as(self.channel)
        glob_idx = torch.floor(values / self.global_max.clamp(min=1e-12) * self.bins).long().clamp(0, self.bins-1)
        self.global_counts += torch.bincount(glob_idx.reshape(-1), minlength=self.bins)
        self.rows += values.shape[0]

    def thresholds(self, quantile=0.999, floor=1e-6):
        if not (0 < quantile <= 1) or self.rows <= 0:
            raise ValueError("Invalid quantile or empty histogram")
        channel_target = math.ceil(self.rows * quantile)
        hit = self.channel.cumsum(dim=1) >= channel_target
        idx = hit.to(torch.int64).argmax(dim=1)
        channel = ((idx+1) * self.maxima / self.bins).clamp(min=floor)
        global_target = math.ceil(self.rows * self.maxima.numel() * quantile)
        glob_idx = (self.global_counts.cumsum(0) >= global_target).to(torch.int64).argmax()
        global_threshold = ((glob_idx+1) * self.global_max / self.bins).clamp(min=floor)
        return channel, global_threshold
